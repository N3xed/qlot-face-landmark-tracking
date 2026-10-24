import torch
import math
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F
from typing import ClassVar, Literal

NUM_PREDS_COV_PARAMS = 5


@dataclass
class Cov:
    """
    A covariance represented as a 2-component 2D Anisotropic Radial Laplace Mixture (MoL) distribution.

    Similar to SEA-RAFT and WAFT, the coordinate error e = p* - p is modeled as a mixture of
    two 2D Radial Laplace components defined relative to a 2D base (inlier) scale matrix
    B = diag(scale_x, scale_y):

       p(e) =   alpha_1 * 1/(2 * pi * sqrt(det(Sigma_1))) * exp(-sqrt(e^T Sigma_1^-1 e))
              + alpha_2 * 1/(2 * pi * sqrt(det(Sigma_2))) * exp(-sqrt(e^T Sigma_2^-1 e))

    where:
       Sigma_2 = B @ I_2 @ B^T = diag(scale_x^2, scale_y^2)
       Sigma_1 = B @ R(theta) @ diag(b_1^2, b_2^2) @ R(theta)^T @ B^T

    In the base-scale-whitened frame e_tilde = B^-1 e = (e_x / scale_x, e_y / scale_y),
    the inlier branch has unit scale I_2 and the outlier branch has scale matrix
    R(theta) @ diag(b_1^2, b_2^2) @ R(theta)^T >= I_2.

    alpha_1 and alpha_2 are the mixture weights, with alpha_1 weighing the outlier branch
    and alpha_2 weighing the inlier branch, parameterized by a single logit w:

       alpha_1 = sigmoid(w), alpha_2 = 1 - alpha_1

    So that the outlier eigenvalues b_i^2 are at least 1 relative to the inlier scale, we
    parameterize:

       b_i = exp(s_i), where s_i are clamped to [0, s_max].

    The rotation angle theta is parameterized as an unconstrained vector (u_x, u_y):

       u_x' = 1 + u_x
       u_y' = u_y
       cos theta = u_x' / sqrt(u_x'^2 + u_y'^2 + norm_eps)
       sin theta = u_y' / sqrt(u_x'^2 + u_y'^2 + norm_eps)

    Attributes:
        params: Tensor of shape (..., 5) containing:
            - s_1, s_2: Log semi-axes of the outlier ellipse relative to `xy_scale_total`.
            - u_x, u_y: Unnormalized direction vector offset from (1, 0) to compute theta.
            - w: Logit of the mixture weight alpha_1 for the outlier branch.
        xy_scale: Coordinate-frame scale `(scale_x, scale_y)` along the x- and y-axes
            (e.g., `(1.0, 1.0)` in pixel space or `(2 / W, 2 / H)` in `[-1, 1]` normalized space).
            Combined with the class constant `base_scale` to form `xy_scale_total`.
    """

    base_scale: ClassVar[float] = 0.5

    params: torch.Tensor  # Shape (..., 5) containing s_1, s_2, u_x, u_y, w
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

    @property
    def w(self) -> torch.Tensor:
        return self.params[..., 4]

    @property
    def alpha_1(self) -> torch.Tensor:
        return torch.sigmoid(self.w)

    def nll_loss(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood of the predicted landmarks under the mixture model.

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

        # Component 2 (Inlier): Fixed Sigma_2 = I_2 -> pure L2
        d2 = torch.sqrt(ex.square() + ey.square() + self.eps**2)  # [B, K]

        # Component 1 (Outlier / Contour): Anisotropic 2x2 Sigma_1 >= I_2
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

        d1 = torch.sqrt((e_para * torch.exp(-s1)).square() + (e_perp * torch.exp(-s2)).square() + self.eps**2)
        s_det_1 = s1 + s2  # 0.5 * log(det(Sigma_1)) since det(R) == 1

        # Mixture Negative Log-Likelihood via logaddexp (including log|det(B)| = log(scale_x * scale_y) Jacobian)
        log_norm = math.log(2.0 * math.pi * scale_x * scale_y)

        log_alpha1 = -F.softplus(-self.w)
        log_alpha2 = -F.softplus(self.w)

        term1 = log_alpha1 - s_det_1 - d1
        term2 = log_alpha2 - d2  # s_det_2 is zero, so omitted
        nll = -torch.logaddexp(term1, term2) + log_norm  # [B, K]

        return nll

    @property
    def scalar_uncertainty(self) -> torch.Tensor:
        """
        Scalar log-volume uncertainty (analogous to SEA-RAFT / WAFT get_heatmap)
        """
        return self.alpha_1 * (self.s1 + self.s2)

    @property
    def variance(self) -> torch.Tensor:
        """
        Returns the geometric mean of the principal variances (sqrt(det(Cov))) under the mixture model.
        """
        scale_x, scale_y = self.xy_scale_total
        alpha_1 = self.alpha_1
        alpha_2 = 1.0 - alpha_1
        var_1 = alpha_1 * torch.exp(2.0 * self.s1) + alpha_2
        var_2 = alpha_1 * torch.exp(2.0 * self.s2) + alpha_2
        return 3.0 * (scale_x * scale_y) * torch.sqrt(var_1 * var_2)

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
            Bounded covariance parameters of shape [..., 5].
        """

        s = self.params[..., :2] * 0.25
        ux = self.ux
        uy = self.uy
        inv_norm = torch.rsqrt(ux.square() + uy.square() + self.norm_eps)
        cos_t = ux * inv_norm
        sin_t = uy * inv_norm
        alpha = self.alpha_1

        return torch.cat(
            [
                s,
                cos_t[..., None],
                sin_t[..., None],
                alpha[..., None],
            ],
            dim=-1,
        )  # [..., 5]


    def as_cov2d_params(
        self,
        mode: Literal["cov", "eff"] = "cov",
    ) -> torch.Tensor:
        """
        Extracts 3 params (diagonal elements and off-diagonal element) of 2D covariance matrix in the current coordinate frame.

        Args:
            mode: "cov" returns the statistical covariance matrix of the mixture,
                  "eff" returns the effective matrix
        Returns:
            2x2 covariance matrix of shape [..., 3] where last dim is (sigma_xx, sigma_yy, sigma_xy).
        """
        alpha_1 = self.alpha_1  # Outlier / anisotropic weight
        alpha_2 = 1.0 - alpha_1  # Inlier / isotropic weight

        s1 = self.s1
        s2 = self.s2

        ux = self.ux
        uy = self.uy
        inv_norm = torch.rsqrt(ux.square() + uy.square() + self.norm_eps)
        cos_t = ux * inv_norm
        sin_t = uy * inv_norm

        # Construct Sigma_1 = R @ diag(b1^2, b2^2) @ R^T
        b1_sq = torch.exp(2.0 * s1)
        b2_sq = torch.exp(2.0 * s2)
        cos_t_sq = cos_t.square()
        sin_t_sq = sin_t.square()
        cos_sin_t = cos_t * sin_t

        sigma1_xx = b1_sq * cos_t_sq + b2_sq * sin_t_sq
        sigma1_xy = (b1_sq - b2_sq) * cos_sin_t
        sigma1_yy = b1_sq * sin_t_sq + b2_sq * cos_t_sq

        # Effective scale matrix: Sigma_eff = B @ (alpha_1 * Sigma_1 + alpha_2 * I_2) @ B^T
        scale_x, scale_y = self.xy_scale_total
        sigma_eff_xx = (alpha_1 * sigma1_xx + alpha_2) * (scale_x * scale_x)
        sigma_eff_xy = (alpha_1 * sigma1_xy) * (scale_x * scale_y)
        sigma_eff_yy = (alpha_1 * sigma1_yy + alpha_2) * (scale_y * scale_y)

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
            mode: "cov" returns the statistical covariance matrix of the mixture,
                  "eff" returns the effective matrix
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
