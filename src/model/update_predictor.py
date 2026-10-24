import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import NUM_PREDS_COORDS, NUM_PREDS_COV_PARAMS, init


def logit(logits: float) -> float:
    """Inverse of sigmoid."""
    return math.log(logits / (1.0 - logits))


class PhaseModulatedPE(nn.Module):
    """
    Positional encoding for 3D query points, based on Phase-Modulated
    Positional Encoding (PMPE) [1].

    The encoding concatenates raw input coordinates with two spectral branches
    that cover complementary frequency bands. All frequencies and phases are
    learnable; the initialisation only provides a well-spread, aperiodic
    starting point.

    Fourier branch (`num_fourier_freq` sin/cos pairs):
        A geometric frequency progression from ~pi*tau^(-1/N) to ~pi/tau with
        sin/cos quadrature pairs (phases 0, pi/2). This covers the high-
        frequency, multi-scale band and follows the geometric init suggested
        by [2] (IJCV 2025). A unit-frequency pair (omega = pi) is prepended
        to bridge into the low-frequency branch.

    Phase-mod branch (`num_phase_mod_freq` sin/cos pairs):
        In the original PMPE [1] this branch used a single fixed carrier
        (pi/2) with a smooth phase progression to break symmetry. A single
        carrier produces a phase-independent diagonal band in the dot-product
        kernel (sum_i sin(w p + phi_i) sin(w p' + phi_i) = L/2 cos(w(p-p'))
        + ...), which no phase progression can remove. Here the carrier is
        replaced by a low-discrepancy frequency sequence in [pi/4, pi] so the
        sum of distinct cosines localises near the diagonal instead. Phases
        use a golden-ratio Weyl sequence with quadrature pairing, retaining
        the anti-resonance / symmetry-breaking role of the original design
        but with provably minimal discrepancy (no rational resonances).

    The two branches are concatenated rather than summed. The original PMPE
    summed them pairwise, which coupled the branches at each channel index
    and -- once both carried diverse frequencies -- became an arbitrary
    constraint requiring a per-channel scale to undo. Concatenation gives the
    downstream linear layer full freedom to mix all channels.

    Golden-ratio Weyl sequences:
        Both frequency and phase low-discrepancy sequences use the golden ratio
        (sqrt(5)-1)/2, the most irrational number (worst Diophantine
        approximant), giving optimal 1D circle coverage with no rational
        resonances. An irrational offset of sqrt(2)/2 is added before the
        mod-1 reduction. This circle translation preserves low discrepancy
        for any offset, but choosing an irrational that is Q-linearly
        independent from the golden ratio (sqrt(2) and sqrt(5) are
        incommensurate) ensures the offset introduces no accidental rational
        structure. Practically, it rotates the sequence so the first few
        elements do not always land near the same points, which matters for
        small sequence lengths.

    Args:
        in_dims: Number of input dimensions (e.g. 3 for xyz).
        num_fourier_freq: Number of geometric Fourier frequency pairs (high band).
        num_phase_mod_freq: Number of low-discrepancy phase-mod frequency pairs
            (low band).
        tau: Bandwidth parameter for the Fourier geometric progression; smaller
            tau extends the progression to higher frequencies.
        concat_raw_input: If True, prepend the raw input coordinates to the
            output for global linearity/injectivity.

    References:
        1. "Cube: A Roblox View of 3D Intelligence" - ArXiv 2025 (original PMPE).
        2. "Mitigating Knowledge Discrepancies among Multiple Datasets for
           Task-agnostic Unified Face Alignment" - IJCV 2025 (geometric init).
    """

    def __init__(self, in_dims: int, num_fourier_freq: int, num_phase_mod_freq, tau=0.008, concat_raw_input=True):
        super().__init__()
        self.in_dims = in_dims
        self.tau = tau
        self.concat_raw_input = concat_raw_input

        self.fourier_len = num_fourier_freq * 2
        self.phase_mod_len = num_phase_mod_freq * 2
        self.total_len = self.fourier_len + self.phase_mod_len

        self.out_dims = in_dims * self.total_len
        if concat_raw_input:
            self.out_dims += in_dims

        # Both fourier features and phase modulation parameters together.
        self.omega = nn.Parameter(torch.empty(in_dims, self.total_len))
        self.phi = nn.Parameter(torch.zeros(self.total_len))

        with torch.no_grad():
            # Fourier branch
            levels = torch.arange(1, self.fourier_len // 2, dtype=torch.float64)  # 1..(fourier_len/2)-1
            self.omega[:, :2] = torch.pi  # Unit frequency for the first two (sin, cos) channels
            # - Geometric progression: (1/tau)^(1/(fourier_len/2)) to (1/tau)
            omega_levels = (1.0 / tau) ** (levels / (self.fourier_len // 2))  # (num_frequencies,)
            self.omega[:, 2 : self.fourier_len] = torch.pi * omega_levels.repeat_interleave(2)  # (L,), [f1, f1, f2, f2, ...]

            # - Sin/cos phase pairing: [0, π/2, 0, π/2, ...]
            phi = torch.pi / 2.0 * (torch.arange(self.fourier_len, dtype=torch.float32) % 2)  # (L,) as [0, π/2, 0, π/2, ...]
            self.phi[: self.fourier_len].copy_(phi)

            # Phase-mod branch.
            golden = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618, most irrational number
            sqrt2_inv = math.sqrt(2.0) / 2.0  # ≈ 0.707, for low-discrepancy phase offsets
            i = torch.arange(1, self.phase_mod_len + 1, dtype=torch.float32)
            # - Frequencies in [π/4, π], low band, distinct per channel
            freq_ld = (i * golden + sqrt2_inv) % 1.0  # low-discrepancy in [0, 1)
            self.omega[:, self.fourier_len :] = torch.pi * (0.25 + 0.75 * freq_ld)  # map to [π/4, π]

            # - Golden-ratio low-discrepancy sequence for phases.
            i = torch.arange(1, self.phase_mod_len // 2 + 1, dtype=torch.float32)
            phi_tilde = 2 * torch.pi * (((i * golden + sqrt2_inv) % 1.0) - 0.5)
            phi_tilde = torch.stack(
                [phi_tilde, phi_tilde + torch.pi / 2.0], dim=-1
            ).flatten()  # (phase_mod_len,), [phi1, phi1 + pi/2, phi2, phi2 + pi/2, ...]
            self.phi[self.fourier_len :].copy_(phi_tilde)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., in_dims)
        Returns:
            shape (..., out_dims)
        """
        y = (x[..., None] * self.omega + self.phi).sin()  # (..., in_dims, total_len)
        y = y.flatten(-2)  # (..., in_dims * total_len)

        if self.concat_raw_input:
            return torch.cat([x, y], dim=-1)  # (..., in_dims + in_dims * total_len)
        else:
            return y  # (..., in_dims * total_len)


class HeadedLinear(nn.Module):
    """
    Multi-headed linear layer, multiple linears in parallel with a single matmul/add.
    """

    def __init__(
        self, in_dim: int, out_dim: int, heads: int, init_out_dim: int | None = None, gain=1.0, head_dim: int = -2
    ) -> None:
        """
        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            heads: Number of parallel heads.
            init_out_dim: Output dimension for weight initialization (default: out_dim).
            gain: Gain factor for weight initialization.
            head_dim: Axis of the input tensor that holds the heads (default: -2). Only -2 and -3 are supported.
                Recommended to use -3 which requires no axis shuffling.
        """

        super().__init__()

        self.weight = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads, out_dim))
        # Axis of the input tensor that holds the heads (default: second-to-last).
        self.head_dim = head_dim

        # Xavier normal init
        init_out_dim = init_out_dim if init_out_dim is not None else out_dim
        std = math.sqrt(2.0 / (in_dim + init_out_dim)) * gain
        nn.init.normal_(self.weight, std=std)

    def init_glu(self) -> None:
        # Kaiming normal init
        in_dim = self.weight.shape[1]
        half_in_dim = in_dim // 2
        std = 1.0 / math.sqrt(in_dim)
        nn.init.normal_(self.weight, std=std)
        with torch.no_grad():
            self.weight[:, half_in_dim:] *= 0.1  # Scale down the GLU gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., heads, ..., in_dim) where `heads` is at dimension `head_dim`.
        Returns:
            (..., heads, ..., out_dim)
        """
        if self.head_dim == -2:
            # bias = self.bias.unsqueeze(-2)  # (heads, 1, out_dim)
            # x = x.unsqueeze(-2)  # (..., heads, 1, in_dim)
            # x = x @ self.weight  # (..., heads, 1, out_dim)
            # return (x + bias).squeeze(-2)  # (..., heads, out_dim)

            # Equalize batch ranks to avoid matmul broadcast expansion.
            leading_shape = x.shape[:-2]
            x_by_head = x.movedim(-2, 0)  # (H, ..., I)
            if leading_shape:
                x_by_head = x_by_head.flatten(1, -2)  # (H, P, I), P = prod(leading_shape)
            else:
                x_by_head = x_by_head.unsqueeze(1)  # (H, 1, I)

            y = torch.matmul(x_by_head, self.weight)  # (H, P, O)

            if leading_shape:
                y = y.unflatten(1, leading_shape)  # (H, ..., O)
            else:
                y = y.squeeze(1)  # (H, O)

            return y.movedim(0, -2) + self.bias  # (..., H, O)

        elif self.head_dim == -3:
            # x: (..., H, R, I) x (H, I, O) -> (..., H, R, O)
            return torch.matmul(x, self.weight) + self.bias.unsqueeze(-2)  # (..., H, R, O) + (H, 1, O) -> (..., H, R, O)
        else:
            # Currently not implemented.
            assert False, f"Unsupported head_dim {self.head_dim}, only -2 and -3 are supported."


class LowRankWriteReadMixer(nn.Module):
    """
    Low-rank write-mix-read module with optional dispersion descriptor (spread/RMS) branch.
    """

    def __init__(
        self,
        read_dim: int,
        write_dim: int,
        value_in_dim: int,
        value_dim: int,
        out_dim: int = 256,
        rank: int = 8,
        heads: int = 2,
        hidden_dim: int = 64,
        enable_spread: bool = False,
    ) -> None:
        """
        Args:
            query_dim: Dimension of the query input (read features).
            key_dim: Dimension of the key input (write features).
            value_in_dim: Dimension of the value input (value features).
            value_dim: Dimension of the projected value.
            out_dim: Output dimension after the write-mix-read operation.
            rank: Number of basis slots.
            heads: Number of parallel heads of the write-mix-read operation.
                `value_dim` must be divisible by `heads`. The effective slot dimension per
                head is `D = value_dim // heads`.
            hidden_dim: Hidden dimension for the routing MLPs.
            enable_spread: Whether to enable the dispersion descriptor. If True enables spread/RMS
                calculation, slot normalization (energy decomposition), and dispersion
                branch residual.
        """

        super().__init__()
        self.enable_spread = enable_spread
        self.rank = rank
        self.out_dim = out_dim
        self.heads = heads

        # Asymmetric routing: context features decide what each landmark contributes
        # to shared basis slots, while identity + temporal state decide what each
        # landmark reads back from those slots.
        self.write_route = nn.Sequential(nn.Linear(write_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * rank * heads))
        nn.init.kaiming_normal_(self.write_route[0].weight, nonlinearity="relu")  # type: ignore
        nn.init.zeros_(self.write_route[0].bias)  # type: ignore

        nn.init.kaiming_uniform_(self.write_route[2].weight, nonlinearity="linear")  # type: ignore
        with torch.no_grad():
            self.write_route[2].weight *= math.sqrt(3.0)  # type: ignore
        nn.init.zeros_(self.write_route[2].bias)  # type: ignore

        self.read_route = nn.Sequential(nn.Linear(read_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, rank * heads))
        nn.init.kaiming_normal_(self.read_route[0].weight, nonlinearity="relu")  # type: ignore
        nn.init.zeros_(self.read_route[0].bias)  # type: ignore
        nn.init.kaiming_uniform_(self.read_route[2].weight, nonlinearity="linear")  # type: ignore
        nn.init.zeros_(self.read_route[2].bias)  # type: ignore

        self.wr_temperature = nn.Parameter(torch.empty(1, 1, 3 * heads * rank))
        nn.init.normal_(self.wr_temperature, mean=0.0, std=0.5)

        assert value_dim % heads == 0, "value_dim must be divisible by heads"
        self.head_dim = value_dim // heads

        self.value_proj = nn.Linear(value_in_dim, value_dim)
        nn.init.xavier_normal_(self.value_proj.weight, gain=1.0)
        nn.init.zeros_(self.value_proj.bias)
        self.read_proj = nn.Linear(read_dim, value_dim)
        nn.init.xavier_normal_(self.read_proj.weight, gain=1.0)
        nn.init.zeros_(self.read_proj.bias)

        # When enable_spread=False, we only have 3 residual branches (read, attn, mlp)
        # instead of 4 (spread, read, attn, mlp).
        residual_count = 4 if enable_spread else 3
        self.residual_weights = nn.Parameter(
            torch.randn((residual_count, 1, heads, 1, 1)) * 0.025
        )  # (1, H, 1, 1) for broadcasting across (B, H, R, D)

        if enable_spread:
            self.basis_spread_proj = nn.Sequential(
                HeadedLinear(self.head_dim + 1, 2 * self.head_dim, heads, head_dim=-3, init_out_dim=self.head_dim),
                nn.GLU(),
            )
            self.basis_spread_proj[0].init_glu()  # type: ignore
        else:
            self.basis_spread_proj = nn.Identity()

        self.basis_read_proj = nn.Sequential(
            HeadedLinear(self.head_dim, 2 * self.head_dim, heads, head_dim=-3, init_out_dim=self.head_dim),
            nn.GLU(),
        )
        self.basis_read_proj[0].init_glu()  # type: ignore

        # Lightweight self-attention across the low-rank basis slots. This lets
        # learned topology factors exchange information without forming an
        # O(N^2) landmark-to-landmark attention matrix.
        self.attn_qk_token_dim = 32
        self.attn_v_token_dim = 64

        # Fused projection: produces [q, k, v] in one MatMul chain.
        self.attn_qkv_proj = HeadedLinear(
            self.head_dim,
            2 * self.attn_qk_token_dim + self.attn_v_token_dim,
            heads,
            init_out_dim=self.attn_qk_token_dim,
            head_dim=-3,
        )
        self.attn_out_proj = HeadedLinear(self.attn_v_token_dim, self.head_dim, heads, head_dim=-3)

        self.basis_mlp = nn.Sequential(
            HeadedLinear(self.head_dim, 2 * self.head_dim, heads=heads, head_dim=-3, gain=nn.init.calculate_gain("relu")),
            nn.SiLU(),
            HeadedLinear(2 * self.head_dim, self.head_dim, heads=heads, head_dim=-3),
        )

        self.local_proj = nn.Sequential(
            nn.Linear(value_in_dim, value_dim // 2),
            nn.SiLU(),
        )
        nn.init.kaiming_normal_(self.local_proj[0].weight, nonlinearity="relu")  # type: ignore
        nn.init.zeros_(self.local_proj[0].bias)  # type: ignore

        self.local_head_proj = HeadedLinear(self.head_dim // 2, 2 * self.head_dim, heads, head_dim=-3, gain=0.1)
        torch.nn.init.constant_(self.local_head_proj.bias[:, : self.head_dim], 1.0)
        torch.nn.init.constant_(self.local_head_proj.bias[:, self.head_dim :], 0.0)

        self.out_proj = nn.Linear(value_dim, out_dim)

    @staticmethod
    def _weighted_basis(write: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Compute a convex combination of the value tensor weighted by the write tensor.

        Calculates (B, H, R, N) x (B, H, N, D) -> (B, H, R, D) using batched matmul.
        A weighted sum of the N value tensors (each of dimension D), for R slots, H heads, and B batches.

        Args:
            write: Tensor of shape (B, N, H, R)
            value: Tensor of shape (B, N, H, D)
        Returns:
            Tensor of shape (B, H, R, D)
        """
        return torch.matmul(write.movedim(1, -1), value.movedim(1, 2))  # (B, H, R, D)

    @staticmethod
    def _read_from_basis(read: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Read from the basis slots for N landmarks with a convex combination.

        Calculates (B, H, N, R) x (B, H, R, D) -> (B, H, N, D) using batched matmul.
        A weighted sum of the R basis slots (each of dimension D), for N landmarks, H heads, and B batches.

        Args:
            read: Tensor of shape (B, N, H, R)
            basis: Tensor of shape (B, H, R, D)
        Returns:
            Tensor of shape (B, H, N, D)
        """
        return torch.matmul(read.transpose(1, 2), basis)  # (B, H, N, D)

    def forward(
        self,
        read_feat: torch.Tensor,
        write_feat: torch.Tensor,
        value_feat: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Tensor of shape (B, N, query_dim)
            key: Tensor of shape (B, N, key_dim)
            value: Tensor of shape (B, N, value_in_dim)
            key_padding_mask: Optional tensor of shape (B, N) indicating which landmarks are masked.

        Returns:
            Tensor of shape (B, N, value_dim) after write-mix-read.
        """
        wr_temperature = torch.exp2(self.wr_temperature / 3.0)  # (1, 1, 3 * H * R)
        w_temp = wr_temperature[:, :, :2 * self.heads * self.rank]  # (1, 1, 2 * H * R)
        r_temp = wr_temperature[:, :, 2 * self.heads * self.rank:]  # (1, 1, H * R)

        write_logits: torch.Tensor = self.write_route(write_feat)  # (B, N, 3 * H * R)
        write_logits = write_logits / w_temp  # (B, N, 2 * H * R)
        write_logits = write_logits.unflatten(dim=-1, sizes=(2, self.heads, self.rank))  # (B, N, 2, H, R)


        # For each basis slot, distribute over landmarks.
        if key_padding_mask is not None:
            # Prevent masked landmarks from writing into global basis slots.
            write_logits = write_logits.masked_fill(key_padding_mask[..., None, None, None], torch.finfo(write_logits.dtype).min)

        # Softmax over landmarks.
        write = write_logits.softmax(dim=1)  # (B, N, 2, H, R)

        # We assume that not all landmarks are masked when the model is deployed
        # for efficiency.
        if key_padding_mask is not None and not torch.compiler.is_exporting():
            # Prevent NaN propagation when all landmarks are masked.
            all_masked = key_padding_mask.all(dim=1, keepdim=True)
            write = torch.where(all_masked[..., None, None, None], 0.0, write)

        # Aggregate rank-R global basis vectors.
        write_v = write[:, :, 0]  # (B, N, H, R)
        v: torch.Tensor = self.value_proj(value_feat).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D)
        write_r = write[:, :, 1]  # (B, N, H, R)
        r: torch.Tensor = self.read_proj(read_feat).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D)

        if self.enable_spread:
            # Fuse mean E[v] and second moment E[v²] into a single matmul,
            # then split back into basis (E[v]) and basis_second_moment (E[v²]).
            vv = torch.cat([v, v.square()], dim=-1)  # (B, N, H, 2D)
            vv = self._weighted_basis(write_v, vv)  # (B, H, R, 2D)
            basis, basis_second_moment = vv.chunk(2, dim=-1)  # each (B, H, R, D)

            # We get RMS for free since RMS = sqrt(E(v^2))
            second_moment_mean_rms = basis_second_moment.mean(dim=-1, keepdim=True).clamp(min=1e-6).sqrt()  # (B, H, R, 1)

            # Calculate the spread: std(v) = sqrt(E(v^2) - E(v)^2)
            basis_spread = (basis_second_moment - basis.square()).clamp(min=1e-6).sqrt()  # (B, H, R, D)

            # This is the coherence: fraction of energy in the mean.
            basis = basis / second_moment_mean_rms  # (B, H, R, D)
            # This is the dispersion: fraction of energy in the spread.
            basis_spread = basis_spread / second_moment_mean_rms  # (B, H, R, D)
            # With the nice identity: Σ basis^2 + basis_residual^2 = D.

            # Include the absolute scale, so the model can use it.
            basis_spread = torch.concat([basis_spread, second_moment_mean_rms], dim=-1)  # (B, H, R, D + 1)

            # Condition basis slots with spread information.
            residual_w = self.residual_weights  # (3, 1, H, 1, 1)
            basis = basis + residual_w[0] * self.basis_spread_proj(basis_spread)  # (B, H, R, D)
        else:
            basis = self._weighted_basis(write_v, v)  # (B, H, R, D)
            residual_w = self.residual_weights

        # Also write read_feat to slots
        basis_read = self._weighted_basis(write_r, r)  # (B, H, R, D)
        basis = basis + residual_w[-3] * self.basis_read_proj(basis_read)  # (B, H, R, D)

        # Mixing across basis slots.
        # This is O(R^2 D) with small R, so it preserves linear scaling in N.
        qkv: torch.Tensor = self.attn_qkv_proj(basis)  # (B, H, R, 3*Daq + Dav)
        r, k, v = qkv.split([self.attn_qk_token_dim, self.attn_qk_token_dim, self.attn_v_token_dim], dim=-1)

        attn_out = F.scaled_dot_product_attention(r, k, v)  # (B, H, R, Dav)
        attn_out = self.attn_out_proj(attn_out)  # (B, H, R, D)
        basis = basis + residual_w[-2] * attn_out  # (B, H, R, D)

        # Residual MLP.
        basis_mlp_out = self.basis_mlp(basis)  # (B, H, R, D)
        basis = basis + residual_w[-1] * basis_mlp_out  # (B, H, R, D)

        # Each landmark makes a convex combination of the basis slots.
        read_logits: torch.Tensor = self.read_route(read_feat)  # (B, N, H * R)
        read_logits = read_logits / r_temp  # (B, N, H * R)
        read_logits: torch.Tensor = read_logits.unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, R)

        # Softmax over slots.
        read = read_logits.softmax(dim=-1)  # (B, N, H, R)
        msg = self._read_from_basis(read, basis)  # (B, H, N, D)

        if self.training and key_padding_mask is not None:
            # Prevent the masked landmarks from optimizing the basis slots.
            masked_read = read * key_padding_mask[..., None, None].to(dtype=read.dtype)
            # Stop the gradients back into the basis slots for the masked landmarks.
            msg = msg + LowRankWriteReadMixer._read_from_basis(masked_read, basis.detach() - basis)

        # Bypass Write-Mix-Read for local features from processed values.
        local: torch.Tensor = self.local_proj(value_feat).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D // 2)
        local = local.transpose(1, 2)  # (B, H, N, D // 2)
        local = self.local_head_proj(local)  # (B, H, N, 2 * D)
        local_factor, local_offset = local.chunk(2, dim=-1)  # each (B, H, N, D)

        # Mix global and local features via FiLM.
        mixed = msg * local_factor + local_offset  # (B, H, N, D)

        # Concat heads, and out-project.
        mixed = mixed.transpose(1, 2).flatten(-2, -1)
        mixed = self.out_proj(mixed)

        return mixed


class GRU(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        init(self.hidden_proj, nonlinearity="tanh")

    def forward(
        self,
        hidden_state: torch.Tensor,
        hidden_state_in: torch.Tensor,
        candidate: torch.Tensor,
        update: torch.Tensor,
        reset: torch.Tensor,
    ) -> torch.Tensor:
        candidate = torch.tanh(reset * self.hidden_proj(hidden_state_in) + candidate)
        if torch.compiler.is_exporting():
            next_hidden = (1.0 - update) * hidden_state + update * candidate  # (B, N, hidden_dim)
        else:
            next_hidden = torch.lerp(hidden_state, candidate, update)  # (B, N, hidden_dim)
        return next_hidden
