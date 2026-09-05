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
        query_dim: int,
        key_dim: int,
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
        self.write_route = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.SiLU(),
        )
        nn.init.kaiming_normal_(self.write_route[0].weight, nonlinearity="relu")  # type: ignore
        nn.init.zeros_(self.write_route[0].bias)  # type: ignore

        self.read_route = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.SiLU())
        nn.init.kaiming_normal_(self.read_route[0].weight, nonlinearity="relu")  # type: ignore
        nn.init.zeros_(self.read_route[0].bias)  # type: ignore

        self.write_proj = nn.Linear(hidden_dim, 2 * rank * heads)
        nn.init.kaiming_uniform_(self.write_proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.write_proj.bias)

        self.write_temperature = nn.Parameter(torch.empty(1, 1, 2 * heads * rank))
        nn.init.normal_(self.write_temperature, mean=0.0, std=0.5)
        self.read_proj = nn.Linear(hidden_dim, rank * heads)

        assert value_dim % heads == 0, "value_dim must be divisible by heads"
        self.head_dim = value_dim // heads

        self.value_proj = nn.Linear(value_in_dim, value_dim)
        nn.init.xavier_normal_(self.value_proj.weight, gain=1.0)
        nn.init.zeros_(self.value_proj.bias)
        self.query_proj = nn.Linear(query_dim, value_dim)
        nn.init.xavier_normal_(self.query_proj.weight, gain=1.0)
        nn.init.zeros_(self.query_proj.bias)

        # When enable_spread=False, we only have 3 residual branches (query, attn, mlp)
        # instead of 4 (spread, query, attn, mlp).
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

        self.basis_query_proj = nn.Sequential(
            HeadedLinear(self.head_dim, 2 * self.head_dim, heads, head_dim=-3, init_out_dim=self.head_dim),
            nn.GLU(),
        )
        self.basis_query_proj[0].init_glu()  # type: ignore

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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
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
        write_route = self.write_route(key)  # (B, N, Dh)
        read_route = self.read_route(query)  # (B, N, Dh)

        write_logits: torch.Tensor = self.write_proj(write_route)  # (B, N, 2 * H * R)
        write_logits = write_logits / torch.exp2(self.write_temperature / 3.0)  # (B, N, 2 * H * R)
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
        v: torch.Tensor = self.value_proj(value).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D)
        write_q = write[:, :, 1]  # (B, N, H, R)
        q: torch.Tensor = self.query_proj(query).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D)

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

        # Also write query/hidden state to basis slots
        basis_query = self._weighted_basis(write_q, q)  # (B, H, R, D)
        basis = basis + residual_w[-3] * self.basis_query_proj(basis_query)  # (B, H, R, D)

        # Mixing across basis slots.
        # This is O(R^2 D) with small R, so it preserves linear scaling in N.
        qkv: torch.Tensor = self.attn_qkv_proj(basis)  # (B, H, R, 3*Daq + Dav)
        q, k, v = qkv.split([self.attn_qk_token_dim, self.attn_qk_token_dim, self.attn_v_token_dim], dim=-1)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, H, R, Dav)
        attn_out = self.attn_out_proj(attn_out)  # (B, H, R, D)
        basis = basis + residual_w[-2] * attn_out  # (B, H, R, D)

        # Residual MLP.
        basis_mlp_out = self.basis_mlp(basis)  # (B, H, R, D)
        basis = basis + residual_w[-1] * basis_mlp_out  # (B, H, R, D)

        # Each landmark makes a convex combination of the basis slots.
        read_logits: torch.Tensor = self.read_proj(read_route).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, R)
        # Softmax over slots.
        read = read_logits.softmax(dim=-1)  # (B, N, H, R)
        msg = self._read_from_basis(read, basis)  # (B, H, N, D)

        if self.training and key_padding_mask is not None:
            # Prevent the masked landmarks from optimizing the basis slots.
            masked_read = read * key_padding_mask[..., None, None].to(dtype=read.dtype)
            # Stop the gradients back into the basis slots for the masked landmarks.
            msg = msg + LowRankWriteReadMixer._read_from_basis(masked_read, basis.detach() - basis)

        # Bypass Write-Mix-Read for local features from processed values.
        local: torch.Tensor = self.local_proj(value).unflatten(dim=-1, sizes=(self.heads, -1))  # (B, N, H, D // 2)
        local = local.transpose(1, 2)  # (B, H, N, D // 2)
        local = self.local_head_proj(local)  # (B, H, N, 2 * D)
        local_factor, local_offset = local.chunk(2, dim=-1)  # each (B, H, N, D)

        # Mix global and local features via FiLM.
        mixed = msg * local_factor + local_offset  # (B, H, N, D)

        # Concat heads, and out-project.
        mixed = mixed.transpose(1, 2).flatten(-2, -1)
        mixed = self.out_proj(mixed)

        return mixed


class UpdatePredictor(nn.Module):
    def __init__(
        self,
        context_feat_dim: int,
        corr_feat_dim: int,
        mixer_rank: int = 8,
        mixer_heads: int = 4,
        mixer_hidden_dim: int = 128,
        mixer_value_dim: int = 256,
        hidden_state_dim: int = 128,
        query_enc_dim: int = 128,
        enable_spread: bool = False,
    ):
        """
        Predict landmark updates and covariance matrix based on landmark features.
        Uses low rank mixing + leaky-integrator (GRU) update block.

        Args:
            context_feat_dim: Dimension of the context features for each landmark.
            corr_feat_dim: Dimension of the correlation features for each landmark.
            mixer_rank: Number of basis slots for low rank mixing.
            mixer_heads: Number of parallel mixer heads.
            mixer_hidden_dim: Hidden dimension for the routing MLPs in the mixer.
            evidence_dim: Dimension of the evidence features produced by the mixer.
            mixer_value_dim: Total mixing dimension per slot (must be divisible by `mixer_heads`).
            hidden_state_dim: Hidden dimension for the recurrent latent state.
            pred_hidden_dim: Hidden dimension for the prediction head.
            out_dim: Output dimension for the predicted landmarks.
            query_enc_dim: Dimension of the query encodings for each landmark.
        """
        super().__init__()

        self.context_feat_dim = context_feat_dim
        self.corr_feat_dim = corr_feat_dim

        self.hidden_state_dim = hidden_state_dim
        self.lmk_feat_dim = context_feat_dim + corr_feat_dim

        self.query_input_dim = hidden_state_dim + query_enc_dim
        self.hidden_norm = nn.RMSNorm(hidden_state_dim)

        self.mixer = LowRankWriteReadMixer(
            query_dim=self.query_input_dim,
            key_dim=context_feat_dim,
            value_in_dim=corr_feat_dim,
            value_dim=mixer_value_dim,
            out_dim=hidden_state_dim * 2,
            rank=mixer_rank,
            heads=mixer_heads,
            hidden_dim=mixer_hidden_dim,
            enable_spread=enable_spread,
        )

        self.gate_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_state_dim, 3 * hidden_state_dim), nn.Sigmoid())
        nn.init.kaiming_normal_(self.gate_proj[-2].weight, nonlinearity="sigmoid")  # type: ignore
        with torch.no_grad():
            # Scale up, since we split into 3 gates.
            self.gate_proj[-2].weight.mul_(math.sqrt(3.0))  # type: ignore

        # reset, update, gain
        gate_bias: torch.Tensor = self.gate_proj[-2].bias  # type: ignore
        # reset gate
        nn.init.constant_(gate_bias[:hidden_state_dim], 0.0)
        # update gate
        # Start with a lower update gate bias, so the hidden state is preserved at the start.
        nn.init.constant_(gate_bias[hidden_state_dim : 2 * hidden_state_dim], logit(0.4))
        # gain gate
        # Start with a high gain, so candidate is perserved.
        nn.init.constant_(gate_bias[2 * hidden_state_dim :], logit(0.9))

        self.hidden_proj = nn.Linear(hidden_state_dim, hidden_state_dim)
        init(self.hidden_proj, nonlinearity="tanh")

        self.pred_head = nn.Sequential(
            HeadedLinear(hidden_state_dim, 128, heads=2, head_dim=-3),
            nn.GLU(),
        )
        self.pred_head[0].init_glu()  # type: ignore

        self.coord_proj = nn.Linear(64, NUM_PREDS_COORDS)
        self.cov_proj = nn.Linear(64, NUM_PREDS_COV_PARAMS)
        nn.init.xavier_uniform_(self.coord_proj.weight, gain=0.1)  # type: ignore
        nn.init.constant_(self.coord_proj.bias, 0.0)  # type: ignore
        nn.init.xavier_uniform_(self.cov_proj.weight, gain=0.1)  # type: ignore
        nn.init.constant_(self.cov_proj.bias, 0.0)  # type: ignore

        if not torch.compiler.is_exporting():
            self.cand_mult: torch.Tensor | None = None
            self.cand_offset: torch.Tensor | None = None
            self.last_hidden_state: torch.Tensor | None = None

    def forward(
        self,
        context_feat: torch.Tensor,
        corr_feat: torch.Tensor,
        hidden_state: torch.Tensor,
        query_encodings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            corr_feat: Correlation features for each landmark, shape (batch_size, num_queries, feature_dim).
            context_feat: Context features for each landmark, shape (batch_size, num_queries, feature_dim).
            hidden_state: Previous hidden state for each landmark, shape (batch_size, num_queries, hidden_dim).
            query_encodings: Encoded query point features, shape (num_queries, query_enc_dim).
            mask: Optional boolean mask for write-mix-read (batch_size, num_queries), True = ignore.
        Returns:
            A tuple containing:
             - Landmark updates (dx, dy), shape (batch_size, num_queries, NUM_PREDS_COORDS).
             - Covariance matrix params, shape (batch_size, num_queries, NUM_PREDS_COV_PARAMS).
             - Updated hidden state h_t, shape (batch_size, num_queries, hidden_dim).
        """

        hidden_normed = self.hidden_norm(hidden_state)  # (B, N, hidden_dim)
        hidden_state_feat = torch.cat([hidden_normed, query_encodings], dim=-1)  # (B, N, hidden_dim + query_enc_dim)

        # Low-rank mixing of correlation features with context and hidden state.
        mixed = self.mixer(
            query=hidden_state_feat,
            key=context_feat,
            value=corr_feat,
            key_padding_mask=mask,
        )

        candidate, gate_feat = mixed.chunk(2, dim=-1)  # each (B, N, hidden_dim)

        gate = self.gate_proj(gate_feat)  # (B, N, 3 * hidden_dim)
        reset, update, gain = gate.chunk(3, dim=-1)  # each (B, N, hidden_dim)

        candidate = torch.tanh(reset * self.hidden_proj(hidden_normed) + candidate) * gain  # (B, N, hidden_dim)

        if torch.compiler.is_exporting():
            next_hidden = (1.0 - update) * hidden_state + update * candidate  # (B, N, hidden_dim)
        else:
            next_hidden = torch.lerp(hidden_state, candidate, update)  # (B, N, hidden_dim)

        pred_input = next_hidden.unsqueeze(-3)  # (B, 1, N, hidden_dim)
        output = self.pred_head(pred_input)  # (B, 2, N, 64)

        coord_output, cov_output = output.unbind(dim=-3)  # each (B, N, 64)
        coord_output = self.coord_proj(coord_output)
        cov_output = self.cov_proj(cov_output)

        return coord_output, cov_output, next_hidden
