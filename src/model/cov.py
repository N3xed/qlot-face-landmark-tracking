import torch
import math
from dataclasses import dataclass
from typing import ClassVar, Literal

NUM_PREDS_COV_PARAMS = 4


@dataclass
class Cov:
    """
    A covariance represented as a 2D Anisotropic Radial Laplace distribution.

    The coordinate error e = p* - p is modeled as a 2D Radial Laplace density
    defined relative to a 2D base (inlier) scale matrix B = diag(scale_x, scale_y):

       p(e) = 1 / (2 * pi * sqrt(det(Sigma))) * exp(-sqrt(e^T Sigma^-1 e))

    where:
       Sigma = B @ R(theta) @ diag(b_1^2, b_2^2) @ R(theta)^T @ B^T

    In the base-scale-whitened frame e_tilde = B^-1 e = (e_x / scale_x, e_y / scale_y),
    the scale matrix is R(theta) @ diag(b_1^2, b_2^2) @ R(theta)^T >= I_2.

    So that the eigenvalues b_i^2 are at least 1 relative to the base (inlier) scale
    (preventing variance collapse and recovering pure unit-scale L2 loss at s_1 = s_2 = 0),
    we parameterize:

       b_i = exp(s_i), where s_i are clamped to [0, s_max].

    The rotation angle theta is parameterized as an unconstrained vector (u_x, u_y):

       u_x' = 1 + u_x
       u_y' = u_y
       cos theta = u_x' / sqrt(u_x'^2 + u_y'^2 + norm_eps)
       sin theta = u_y' / sqrt(u_x'^2 + u_y'^2 + norm_eps)

    Attributes:
        params: Tensor of shape (..., 4) containing:
            - s_1, s_2: Log semi-axes of the uncertainty ellipse relative to `xy_scale_total`.
            - u_x, u_y: Unnormalized direction vector offset from (1, 0) to compute theta.
        xy_scale: Coordinate-frame scale `(scale_x, scale_y)` along the x- and y-axes
            (e.g., `(1.0, 1.0)` in pixel space or `(2 / W, 2 / H)` in `[-1, 1]` normalized space).
            Combined with the class constant `base_scale` to form `xy_scale_total`.
    """

    base_scale: ClassVar[float] = 0.25

    params: torch.Tensor  # Shape (..., 4) containing s_1, s_2, u_x, u_y
    xy_scale: tuple[float, float] = (1.0, 1.0)

    norm_eps: float = 1e-6
    eps: float = 1e-4

    def __post_init__(self) -> None:
        if isinstance(self.xy_scale, (int, float)):
            s = float(self.xy_scale)
            self.xy_scale = (s, s)
        else:
            self.xy_scale = (float(self.xy_scale[0]), float(self.xy_scale[1]))

    @property
    def xy_scale_total(self) -> tuple[float, float]:
        return (self.xy_scale[0] * self.base_scale, self.xy_scale[1] * self.base_scale)

    @property
    def s1(self) -> torch.Tensor:
        return self.params[..., 0]

    @property
    def s2(self) -> torch.Tensor:
        return self.params[..., 1]

    @property
    def ux(self) -> torch.Tensor:
        return 1.0 + self.params[..., 2]

    @property
    def uy(self) -> torch.Tensor:
        return self.params[..., 3]

    def nll_loss(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood of the predicted landmarks under the Radial Laplace model.

        Args:
            pred_xy: Predicted landmark coordinates of shape [B, K, 2].
            gt_xy: Ground-truth landmark coordinates of shape [B, K, 2].
        Returns:
            Negative log-likelihood of shape [B, K].
        """
        scale_x, scale_y = self.xy_scale_total
        e = gt_xy - pred_xy
        ex = e[..., 0] / scale_x
        ey = e[..., 1] / scale_y

        # Anisotropic 2x2 Sigma >= I_2 in base-scale-whitened frame
        s1 = self.s1
        s2 = self.s2

        # Continuous 2D rotation via L2-normalized (ux, uy), initialized at (1, 0)
        ux = self.ux
        uy = self.uy
        inv_norm = torch.rsqrt(ux.square() + uy.square() + self.norm_eps)
        cos_t = ux * inv_norm
        sin_t = uy * inv_norm

        # Rotate residual into principal ellipse axes
        e_para = cos_t * ex + sin_t * ey
        e_perp = -sin_t * ex + cos_t * ey

        d = torch.sqrt((e_para * torch.exp(-s1)).square() + (e_perp * torch.exp(-s2)).square() + self.eps**2)
        s_det = s1 + s2  # 0.5 * log(det(Sigma)) since det(R) == 1

        # Negative Log-Likelihood (including log|det(B)| = log(scale_x * scale_y) Jacobian)
        log_norm = math.log(2.0 * math.pi * scale_x * scale_y)
        nll = s_det + d + log_norm  # [B, K]

        return nll

    @property
    def scalar_uncertainty(self) -> torch.Tensor:
        """
        Scalar log-volume uncertainty (0.5 * log(det(Sigma)) in base-scale-whitened frame).
        """
        return self.s1 + self.s2

    @property
    def variance(self) -> torch.Tensor:
        """
        Returns the geometric mean of the principal variances (sqrt(det(Cov))) under the Radial Laplace model.
        """
        scale_x, scale_y = self.xy_scale_total
        return 3.0 * (scale_x * scale_y) * torch.exp(self.s1 + self.s2)

    def scale(self, scale_x: float, scale_y: float | None = None) -> "Cov":
        """
        Scale the covariance parameters by the given factors.

        Args:
            scale_x: Scaling factor for the x-axis (or both axes if scale_y is None).
            scale_y: Scaling factor for the y-axis. Defaults to scale_x if None.
        Returns:
            Scaled Cov object.
        """
        if scale_y is None:
            scale_y = scale_x
        sx, sy = self.xy_scale
        return Cov(
            params=self.params,
            xy_scale=(sx * float(scale_x), sy * float(scale_y)),
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def clamp(self, s_max: float) -> "Cov":
        s = self.params[..., :2].clamp(min=0.0, max=s_max)
        others = self.params[..., 2:]
        new_params = torch.cat([s, others], dim=-1)
        return Cov(
            params=new_params,
            xy_scale=self.xy_scale,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def __getitem__(self, key):
        return Cov(
            params=self.params[key],
            xy_scale=self.xy_scale,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def clone(self) -> "Cov":
        return Cov(
            params=self.params.clone(),
            xy_scale=self.xy_scale,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def detach(self) -> "Cov":
        return Cov(
            params=self.params.detach(),
            xy_scale=self.xy_scale,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def to(self, device: torch.device | str) -> "Cov":
        return Cov(
            params=self.params.to(device),
            xy_scale=self.xy_scale,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def as_bounded_params(self) -> torch.Tensor:
        """
        Returns the covariance parameters in a bounded form,
        approximately in [-1, 1].

        Returns:
            Bounded covariance parameters of shape [..., 4].
        """

        s = self.params[..., :2] * 0.25
        ux = self.ux
        uy = self.uy
        inv_norm = torch.rsqrt(ux.square() + uy.square() + self.norm_eps)
        cos_t = ux * inv_norm
        sin_t = uy * inv_norm

        return torch.cat(
            [
                s,
                cos_t[..., None],
                sin_t[..., None],
            ],
            dim=-1,
        )  # [..., 4]


    def as_cov2d_params(
        self,
        mode: Literal["cov", "eff"] = "cov",
    ) -> torch.Tensor:
        """
        Extracts 3 params (diagonal elements and off-diagonal element) of 2D covariance matrix in the current coordinate frame.

        Args:
            mode: "cov" returns the statistical covariance matrix of the distribution,
                  "eff" returns the effective scale matrix
        Returns:
            2x2 covariance matrix of shape [..., 3] where last dim is (sigma_xx, sigma_yy, sigma_xy).
        """
        s1 = self.s1
        s2 = self.s2

        ux = self.ux
        uy = self.uy
        inv_norm = torch.rsqrt(ux.square() + uy.square() + self.norm_eps)
        cos_t = ux * inv_norm
        sin_t = uy * inv_norm

        # Construct Sigma_tilde = R @ diag(b1^2, b2^2) @ R^T
        b1_sq = torch.exp(2.0 * s1)
        b2_sq = torch.exp(2.0 * s2)
        cos_t_sq = cos_t.square()
        sin_t_sq = sin_t.square()
        cos_sin_t = cos_t * sin_t

        sigma1_xx = b1_sq * cos_t_sq + b2_sq * sin_t_sq
        sigma1_xy = (b1_sq - b2_sq) * cos_sin_t
        sigma1_yy = b1_sq * sin_t_sq + b2_sq * cos_t_sq

        # Scale matrix: Sigma = B @ Sigma_tilde @ B^T
        scale_x, scale_y = self.xy_scale_total
        sigma_eff_xx = sigma1_xx * (scale_x * scale_x)
        sigma_eff_xy = sigma1_xy * (scale_x * scale_y)
        sigma_eff_yy = sigma1_yy * (scale_y * scale_y)

        sigma_eff = torch.stack(
            [
                sigma_eff_xx,
                sigma_eff_yy,
                sigma_eff_xy,
            ],
            dim=-1,
        )  # [..., 3]

        if mode == "cov":
            return 3.0 * sigma_eff
        elif mode == "eff":
            return sigma_eff
        assert False, f"Unknown mode: {mode}"

    @torch.no_grad()
    def as_cov2d_matrix(
        self,
        mode: Literal["cov", "eff"] = "cov",
    ) -> torch.Tensor:
        """
        Extracts 2x2 covariance matrix in the current coordinate frame.

        Args:
            mode: "cov" returns the statistical covariance matrix of the distribution,
                  "eff" returns the effective scale matrix
        Returns:
            2x2 covariance matrix of shape [..., 2, 2].
        """

        cov2d_params = self.as_cov2d_params(mode=mode)
        sigma_xx, sigma_yy, sigma_xy = cov2d_params.unbind(dim=-1)
        cov2d = torch.stack(
            [
                torch.stack([sigma_xx, sigma_xy], dim=-1),
                torch.stack([sigma_xy, sigma_yy], dim=-1),
            ],
            dim=-2,
        )  # [..., 2, 2]
        return cov2d
