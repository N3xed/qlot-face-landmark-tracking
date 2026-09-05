import torch
import math
from dataclasses import dataclass
from torch import nn


CovarianceScale = torch.Tensor | float | int | tuple[float, float]


def _scalar_scale_values(scale: CovarianceScale) -> tuple[float, float] | None:
    if isinstance(scale, torch.Tensor):
        return None
    if isinstance(scale, tuple):
        return float(scale[0]), float(scale[1])
    value = float(scale)
    return value, value


@dataclass
class Cov2D:
    """
    Class representing a 2D covariance matrix.

    Attributes:
        params: Tensor of shape (..., 3) representing (var_x, var_y, cov_xy).
    """

    params: torch.Tensor  # Shape: (..., 3) representing (var_x, var_y, cov_xy)

    @property
    def variance_x(self) -> torch.Tensor:
        """Variance along the x-axis."""
        return self.params[..., 0]

    @property
    def stddev_x(self) -> torch.Tensor:
        """Standard deviation along the x-axis."""
        return torch.sqrt(self.variance_x)

    @property
    def variance_y(self) -> torch.Tensor:
        """Variance along the y-axis."""
        return self.params[..., 1]

    @property
    def stddev_y(self) -> torch.Tensor:
        """Standard deviation along the y-axis."""
        return torch.sqrt(self.variance_y)

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance between x and y."""
        return self.params[..., 2]

    @property
    def correlation(self) -> torch.Tensor:
        """Correlation coefficient between x and y."""
        std_x = torch.sqrt(self.variance_x)
        std_y = torch.sqrt(self.variance_y)
        return self.covariance / (std_x * std_y + 1e-6)

    @property
    def max_variance(self) -> torch.Tensor:
        """Maximum of the variances along x and y axes."""
        return self.params[..., 0:2].amax(dim=-1)

    @property
    def min_variance(self) -> torch.Tensor:
        """Minimum of the variances along x and y axes."""
        return self.params[..., 0:2].amin(dim=-1)

    @property
    def generalized_variance(self) -> torch.Tensor:
        """
        Square root of the generalized variance (proportional to the area of the
        uncertainty ellipse, which is sqrt(det(Sigma))).
        """
        return (self.variance_x * self.variance_y - self.covariance.square()).sqrt()

    def as_cov2d_params(self) -> torch.Tensor:
        """
        Return the covariance parameters as a tensor of shape (..., 3).
        The last dimension represents (var_x, var_y, cov_xy).
        """
        return self.params

    @staticmethod
    def from_params(var_x: torch.Tensor, var_y: torch.Tensor, cov_xy: torch.Tensor) -> "Cov2D":
        """
        Create a Cov2D instance from the given parameters.
        """
        params = torch.stack([var_x, var_y, cov_xy], dim=-1)
        return Cov2D(params=params)

    def clone(self) -> "Cov2D":
        return Cov2D(params=self.params.clone())

    def detach(self) -> "Cov2D":
        return Cov2D(params=self.params.detach())

    def __getitem__(self, key) -> "Cov2D":
        return Cov2D(params=self.params[key])

    def to(self, *args, **kwargs) -> "Cov2D":
        return Cov2D(params=self.params.to(*args, **kwargs))

    def to_matrix(self) -> torch.Tensor:
        """Convert the covariance parameters to a full covariance matrix."""
        return torch.stack(
            [torch.stack([self.variance_x, self.covariance], dim=-1), torch.stack([self.covariance, self.variance_y], dim=-1)],
            dim=-2,
        )

    @staticmethod
    def from_matrix(matrix: torch.Tensor) -> "Cov2D":
        """Create a Cov2D instance from a full covariance matrix."""
        var_x = matrix[..., 0, 0]
        var_y = matrix[..., 1, 1]
        cov_xy = matrix[..., 0, 1]
        return Cov2D.from_params(var_x=var_x, var_y=var_y, cov_xy=cov_xy)

    def scale_clamp(
        self,
        scale: CovarianceScale | None = None,
        log_min_stddev: float | None = None,
        log_max_stddev: float | None = None,
        inplace=False,
    ) -> "Cov2D":
        """
        Scale and or clamp the covariance.

        Args:
            scale: Scaling factors of shape (..., 2), a scalar, or an (x, y) pair
                (should be positive).
            log_min_stddev: The log of minimum stddev to clamp to.
            log_max_stddev: The log of maximum stddev to clamp to.
        Returns:
            A new instance with scaled covariance.
        """

        def scale_cov(diag: torch.Tensor, cross: torch.Tensor, scale: CovarianceScale) -> tuple[torch.Tensor, torch.Tensor]:
            # S = diag(scale), then C' = S C S^T
            scalar_scales = _scalar_scale_values(scale)
            if isinstance(scale, torch.Tensor):
                diag = diag * torch.square(scale)  # scale var_x and var_y
                cross = cross * scale[..., 0:1] * scale[..., 1:2]  # scale cov_xy
            else:
                assert scalar_scales is not None
                scale_x, scale_y = scalar_scales
                diag = torch.stack(
                    [diag[..., 0] * scale_x**2, diag[..., 1] * scale_y**2],
                    dim=-1,
                )
                cross = cross * (scale_x * scale_y)  # scale cov_xy
            return diag, cross

        min_var = math.exp(2 * log_min_stddev) if log_min_stddev is not None else None
        max_var = math.exp(2 * log_max_stddev) if log_max_stddev is not None else None

        if inplace:
            if min_var is not None or max_var is not None:
                self.params[..., :2].clamp_(min_var, max_var)
            if scale is not None:
                diag, cross = scale_cov(self.params[..., :2], self.params[..., 2:], scale)
                self.params[..., :2] = diag
                self.params[..., 2] = cross
            return self

        diag = self.params[..., :2]
        cross = self.params[..., 2:]

        if min_var is not None or max_var is not None:
            diag = diag.clamp(min_var, max_var)

        if scale is not None:
            diag, cross = scale_cov(diag, cross, scale)

        params = torch.cat(
            [diag, cross],
            dim=-1,
        )
        return Cov2D(params)

    def gaussian_negative_log_likelihood(self, diffs: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood of given differences under the Gaussian defined by this covariance.

        Args:
            diffs: Tensor of shape (..., 2) representing the differences (mu_x - x, mu_y -
            y). mu_x and mu_y are the predicted means. x and y are the targets (labels).

        Returns:
            Negative log-likelihood values.
        """
        x_diff = diffs[..., 0]
        y_diff = diffs[..., 1]

        shape = x_diff.shape
        assert y_diff.shape == shape, f"{y_diff.shape} != {shape}"
        assert self.variance_x.shape == shape, f"{self.variance_x.shape} != {shape}"
        assert self.variance_y.shape == shape, f"{self.variance_y.shape} != {shape}"
        assert self.covariance.shape == shape, f"{self.covariance.shape} != {shape}"

        # Gaussian Likelihood
        # N(mu, Sigma) ∝ det(Sigma)^(-1/2) * exp(-0.5 * (mu - x)^T Sigma^(-1) (mu - x))
        # max N <=> max log N <=> min -log N = min 1/2 * log(det(Sigma)) + 1/2 * (mu-x)^T Sigma^(-1) (mu-x)
        # => GNLL = 0.5 * log(det(Sigma)) + 0.5 * (mu-x)^T Sigma^(-1) (mu-x)

        det = self.variance_x * self.variance_y - self.covariance**2
        inv_var_x = self.variance_y / det
        inv_var_y = self.variance_x / det
        inv_cov_xy = -self.covariance / det

        mahalanobis = x_diff**2 * inv_var_x + y_diff**2 * inv_var_y + 2 * x_diff * y_diff * inv_cov_xy

        return 0.5 * torch.log(det + 1e-6) + 0.5 * mahalanobis


@dataclass
class LowRankCov2D:
    """
    Class representing a 2D covariance matrix using its symmetric Cholesky-like decomposition.

    The covariance matrix Sigma is represented as:
        - log_sigma_x = log(sigma_x)
        - log_sigma_y = log(sigma_y)
        - rho_raw = atanh(rho), where rho is the correlation coefficient between x and y.

    Attributes:
        params: Tensor of shape (..., 3) representing (log_sigma_x, log_sigma_y, rho_raw).
    """

    params: torch.Tensor  # Shape: (..., 3) representing (log_sigma_x, log_sigma_y, rho_raw)

    @staticmethod
    def from_params(log_sigma_x: torch.Tensor, log_sigma_y: torch.Tensor, rho_raw: torch.Tensor) -> "LowRankCov2D":
        """
        Create an instance from the given parameters.
        """
        params = torch.stack([log_sigma_x, log_sigma_y, rho_raw], dim=-1)
        return LowRankCov2D(params=params)

    def clone(self) -> "LowRankCov2D":
        return LowRankCov2D(params=self.params.clone())

    def detach(self) -> "LowRankCov2D":
        return LowRankCov2D(params=self.params.detach())

    def __getitem__(self, key) -> "LowRankCov2D":
        return LowRankCov2D(params=self.params[key])

    def to(self, *args, **kwargs) -> "LowRankCov2D":
        return LowRankCov2D(params=self.params.to(*args, **kwargs))

    @property
    def log_sigma_x(self) -> torch.Tensor:
        """Logarithm of the standard deviation along the x-axis."""
        return self.params[..., 0]

    @property
    def log_sigma_y(self) -> torch.Tensor:
        """Logarithm of the standard deviation along the y-axis."""
        return self.params[..., 1]

    @property
    def log_sigmas(self) -> torch.Tensor:
        """Logarithm of the standard deviations along both axes (log stddev_x, log stddev_y)."""
        return self.params[..., :2]

    @property
    def stddev_x(self) -> torch.Tensor:
        """Standard deviation along the x-axis."""
        return self.params[..., 0].exp()

    @property
    def variance_x(self) -> torch.Tensor:
        """Variance along the x-axis."""
        return self.stddev_x.square()

    @property
    def stddev_y(self) -> torch.Tensor:
        """Standard deviation along the y-axis."""
        return self.params[..., 1].exp()

    @property
    def variance_y(self) -> torch.Tensor:
        """Variance along the y-axis."""
        return self.stddev_y.square()

    @property
    def correlation(self) -> torch.Tensor:
        """Correlation coefficient between x and y."""
        return self.params[..., 2].tanh()

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance between x and y."""
        return self.correlation * self.stddev_x * self.stddev_y

    @property
    def max_variance(self) -> torch.Tensor:
        """Maximum of the variances along x and y axes."""
        return torch.stack([self.variance_x, self.variance_y], dim=-1).amax(dim=-1)

    @property
    def min_variance(self) -> torch.Tensor:
        """Minimum of the variances along x and y axes."""
        return torch.stack([self.variance_x, self.variance_y], dim=-1).amin(dim=-1)

    @property
    def generalized_variance(self) -> torch.Tensor:
        """
        Square root of the generalized variance (proportional to the area of the
        uncertainty ellipse, which is sqrt(det(Sigma))).
        """
        return (self.variance_x * self.variance_y - self.covariance.square()).sqrt()

    def as_cov2d_params(self) -> torch.Tensor:
        """
        Return the covariance parameters as a tensor of shape (..., 3).
        The last dimension represents (var_x, var_y, cov_xy).
        """
        sigma_x = self.stddev_x
        sigma_y = self.stddev_y
        rho = self.correlation
        var_x = sigma_x.square()
        var_y = sigma_y.square()
        cov = rho * sigma_x * sigma_y

        return torch.stack([var_x, var_y, cov], dim=-1)

    def to_matrix(self) -> torch.Tensor:
        """Convert to a full covariance matrix."""
        cov_params = self.as_cov2d_params()  # Shape: (..., 3)

        C = torch.empty(*cov_params.shape[:-1], 2, 2, device=cov_params.device, dtype=cov_params.dtype)
        C[..., 0, 0] = cov_params[..., 0]  # var_x
        C[..., 1, 1] = cov_params[..., 1]  # var_y
        C[..., 0, 1] = cov_params[..., 2]  # cov_xy
        C[..., 1, 0] = cov_params[..., 2]  # cov_xy
        return C

    def to_cholesky_matrix(self) -> torch.Tensor:
        """Convert to the (lower) Cholesky matrix L."""

        l11 = self.stddev_x
        rho = self.correlation

        sigma_y = self.stddev_y
        l21 = rho * sigma_y
        l22 = torch.sqrt(sigma_y**2 - l21**2)

        L = torch.zeros(*self.params.shape[:-1], 2, 2, device=self.params.device, dtype=self.params.dtype)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 1, 1] = l22
        return L

    @staticmethod
    def from_matrix(matrix: torch.Tensor, eps=1e-6) -> "LowRankCov2D":
        """Create an instance from a full covariance matrix."""
        var_x = matrix[..., 0, 0]
        var_y = matrix[..., 1, 1]
        cov_xy = matrix[..., 0, 1]

        sigma_x = torch.sqrt(var_x + eps)
        sigma_y = torch.sqrt(var_y + eps)
        rho = cov_xy / (sigma_x * sigma_y + eps)
        log_sigma_x = sigma_x.log()
        log_sigma_y = sigma_y.log()

        rho_raw = rho.clamp(-1, 1).atanh()
        return LowRankCov2D.from_params(log_sigma_x=log_sigma_x, log_sigma_y=log_sigma_y, rho_raw=rho_raw)

    def to_cov2d(self) -> Cov2D:
        """Convert to Cov2D representation."""
        sigma_x = self.stddev_x
        sigma_y = self.stddev_y
        rho = self.correlation
        var_x = sigma_x.square()
        var_y = sigma_y.square()
        cov = rho * sigma_x * sigma_y
        return Cov2D.from_params(var_x=var_x, var_y=var_y, cov_xy=cov)

    def scale_clamp(
        self,
        scale: CovarianceScale | None = None,
        log_min_stddev: float | None = None,
        log_max_stddev: float | None = None,
        inplace=False,
    ) -> "LowRankCov2D":
        """
        Scale and or clamp the covariance.

        Args:
            scale: Scaling factors of shape (..., 2), a scalar, or an (x, y) pair
                (should be positive).
            log_min_stddev: The log of minimum stddev to clamp to.
            log_max_stddev: The log of maximum stddev to clamp to.
        Returns:
            A new instance with scaled covariance.
        """
        scalar_scales = _scalar_scale_values(scale) if scale is not None else None

        if inplace:
            if log_min_stddev is not None or log_max_stddev is not None:
                self.params[..., :2].clamp_(min=log_min_stddev, max=log_max_stddev)
            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    self.params[..., :2] += torch.log(scale)  # diagonals are the logs, so add log(scale)
                else:
                    assert scalar_scales is not None
                    scale_x, scale_y = scalar_scales
                    self.params[..., 0] += math.log(scale_x)
                    self.params[..., 1] += math.log(scale_y)
            return self

        diag = self.params[..., :2]
        if log_min_stddev is not None or log_max_stddev is not None:
            diag = diag.clamp(min=log_min_stddev, max=log_max_stddev)
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                diag = diag + torch.log(scale)  # diagonals are the logs, so add log(scale)
            else:
                assert scalar_scales is not None
                scale_x, scale_y = scalar_scales
                diag = torch.stack(
                    [diag[..., 0] + math.log(scale_x), diag[..., 1] + math.log(scale_y)],
                    dim=-1,
                )

        params = torch.cat([diag, self.params[..., 2:]], dim=-1)
        return LowRankCov2D(params)

    def gaussian_negative_log_likelihood(self, diffs: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood of given differences under the Gaussian defined by this covariance.

        Args:
            diffs: Tensor of shape (..., 2) representing the differences (mu_x - x, mu_y -
                y). mu_x and mu_y are the predicted means, x and y are the targets (labels).
        Returns:
            Negative log-likelihood values.
        """

        x_diff = diffs[..., 0]
        y_diff = diffs[..., 1]

        shape = x_diff.shape
        assert y_diff.shape == shape, f"{y_diff.shape} != {shape}"
        assert self.log_sigma_x.shape == shape, f"{self.log_sigma_x.shape} != {shape}"
        assert self.log_sigma_y.shape == shape, f"{self.log_sigma_y.shape} != {shape}"
        assert self.correlation.shape == shape, f"{self.correlation.shape} != {shape}"

        # Gaussian Likelihood
        # N(mu, Sigma) ∝ det(Sigma)^(-1/2) * exp(-0.5 * (mu - x)^T Sigma^(-1) (mu - x))
        # max N <=> max log N <=> min -log N = min 1/2 * log(det(Sigma)) + 1/2 * (mu-x)^T Sigma^(-1) (mu-x)
        # => GNLL = 0.5 * log(det(Sigma)) + 0.5 * (mu-x)^T Sigma^(-1) (mu-x)

        # 1. Calculate z-scores
        z_x = x_diff / self.stddev_x
        z_y = y_diff / self.stddev_y

        rho = self.correlation
        # Clamp rho slightly to prevent division by zero or log of zero during training
        rho_sq = rho.square().clamp(max=1 - 1e-6)
        inv_one_minus_rho_sq = 1.0 / (1.0 - rho_sq)

        # 2. Log-determinant term: log(sigma_x) + log(sigma_y) + 0.5 * log(1 - rho^2)
        # Using log1p(-rho_sq) for better numerical stability when rho is near 0
        log_det = self.log_sigma_x + self.log_sigma_y + 0.5 * torch.log1p(-rho_sq)

        # 3. Mahalanobis distance term: (z_x^2 - 2*rho*z_x*z_y + z_y^2) / (2 * (1 - rho^2))
        mahalanobis = (z_x.square() - 2.0 * rho * z_x * z_y + z_y.square()) * inv_one_minus_rho_sq

        return log_det + 0.5 * mahalanobis

    def __add__(self, other: "GenericCov2D") -> "Cov2D":
        params = self.as_cov2d_params() + other.as_cov2d_params()
        return Cov2D(params=params)


GenericCov2D = Cov2D | LowRankCov2D


class CovGatedUpdate(nn.Module):
    def __init__(self, method: str = "scalar") -> None:
        super().__init__()
        self.method = method  # 'scalar' or 'matrix'

    @staticmethod
    def _cutoff(k: torch.Tensor, bias: float | torch.Tensor) -> torch.Tensor:
        """Cutoff gating factor k below bias."""
        return nn.functional.relu(k - bias) / (1.0 - bias)

    def forward(
        self, dx: torch.Tensor, cov: LowRankCov2D, radius: torch.Tensor | float | int, cutoff: torch.Tensor | float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_old: Tensor of shape (..., 2) representing the old landmark positions.
            dx: Tensor of shape (..., 2) representing the proposed position updates.
            cov: LowRankCov2D representing the covariance of the updates.
            radius: Tensor of shape (...) representing the certainty radius below which
                updates are ignored to avoid jitter.
            cutoff: Tensor or float representing the bias for gating factor cutoff.
        Returns:
            Tuple of (dx_gated, k). k is the gating factor (scalar or None).
        """

        # Reconstruct L matrix
        k: torch.Tensor | None = None

        if self.method == "scalar_tanh":
            # Directional Scalar Gating

            # Normalize delta_p to get direction u
            norm = torch.linalg.vector_norm(dx, dim=-1).clamp(min=1e-6)
            u = dx / norm.unsqueeze(-1)  # Shape: (..., 2)

            # Calculate projected sigma: ||L^T u|| = ||(u^T L)^T||
            # L is lower triangular, so L^T is upper.
            # Lt_u = L.transpose(-2, -1) @ u.unsqueeze(-1)  # Shape: (..., 2, 1)
            # sigma_proj: torch.Tensor = torch.linalg.vector_norm(Lt_u.squeeze(-1), dim=-1) # Shape: (...,)
            #
            # Simplified formula for LowRankCov2D:
            # sigma_proj^2 = (sigma_x * u_x)^2 + (sigma_y * u_y)^2 + 2 * rho * sigma_x * sigma_y * u_x * u_y
            sigma_u_square = (cov.log_sigmas.exp() * u).square()
            cross_term = 2 * cov.correlation * cov.log_sigmas.sum(dim=-1).exp() * u.prod(dim=-1)
            sigma_proj = (sigma_u_square.sum(dim=-1) + cross_term).sqrt()  # Shape: (...,)

            # Gate
            k = torch.tanh(sigma_proj / radius)  # Shape: (...,)
            k = self._cutoff(k, bias=cutoff)  # type: ignore
            k = k.unsqueeze(-1)

            # Apply
            if self.training:
                # Straight-Through Estimator for gate
                # d/d(dx) = 1, d/d(k) = dx
                dx_gated = (k * dx.detach()) + (dx - dx.detach())
            else:
                dx_gated = k * dx
        elif self.method == "determimant":
            # Determinant Scalar Gating

            # Calculate unprojected sigma: sigma = sqrt(sigma_x * sigma_y)
            sigma_unproj: torch.Tensor = torch.exp((cov.log_sigma_x + cov.log_sigma_y) / 2.0)  # Shape: (...,)

            # Gate
            k = torch.tanh(sigma_unproj / radius)  # Shape: (...,)
            k = self._cutoff(k, bias=cutoff)  # type: ignore
            k = k.unsqueeze(-1)

            # Apply
            if self.training:
                # Straight-Through Estimator for gate
                # d/d(dx) = 1, d/d(k) = dx
                dx_gated = (k * dx.detach()) + (dx - dx.detach())
            else:
                dx_gated = k * dx

        elif self.method == "scalar_rational":
            # Directional Scalar Gating

            L = cov.to_cholesky_matrix()  # Shape: (..., 2, 2)
            # Normalize delta_p to get direction u
            u = nn.functional.normalize(dx, dim=-1)  # Shape: (..., 2)

            # Calculate projected sigma: ||L^T u|| = ||(u^T L)^T||
            # L is lower triangular, so L^T is upper.
            Lt_u = L.transpose(-2, -1) @ u.unsqueeze(-1)  # Shape: (..., 2, 1)
            sigma_proj: torch.Tensor = torch.linalg.vector_norm(Lt_u.squeeze(-1), dim=-1) ** 2  # Shape: (...,)
            tau_sq = radius**2

            # Gate
            k = sigma_proj / (sigma_proj + tau_sq)  # Shape: (...,)
            k = self._cutoff(k, bias=cutoff)  # type: ignore
            k = k.unsqueeze(-1)

            # Apply
            if self.training:
                # Straight-Through Estimator for gate
                # d/d(dx) = 1, d/d(k) = dx
                dx_gated = (k * dx.detach()) + (dx - dx.detach())
            else:
                dx_gated = k * dx

        elif self.method == "matrix":
            # Kalman Gain Matrix Gating

            # Reconstruct C = L @ L^T
            C = cov.to_matrix()  # Shape: (..., 2, 2)

            # K = C @ (C + tau*I)^-1
            # Create Identity matrix
            I = torch.eye(2, device=C.device, dtype=C.dtype)  # Shape: (2, 2)

            if isinstance(radius, float) or isinstance(radius, int):
                tau_sq = radius**2
            else:
                tau_sq = radius.to(device=C.device, dtype=C.dtype).square()
                if tau_sq.dim() < C.dim():
                    tau_sq = tau_sq.view(*tau_sq.shape, *([1] * (C.dim() - tau_sq.dim())))  # Shape: (..., 1, 1)

            # Term to invert
            denom = C + (tau_sq * I)

            # Invert 2x2 matrix manually (faster/stabler than torch.inverse for small dims)
            # inv([[a,b],[c,d]]) = 1/det * [[d,-b],[-c,a]]
            det = denom[..., 0, 0] * denom[..., 1, 1] - denom[..., 0, 1] * denom[..., 1, 0]
            inv_denom = torch.zeros_like(denom)
            inv_denom[..., 0, 0] = denom[..., 1, 1]
            inv_denom[..., 1, 1] = denom[..., 0, 0]
            inv_denom[..., 0, 1] = -denom[..., 0, 1]
            inv_denom[..., 1, 0] = -denom[..., 1, 0]
            inv_denom = inv_denom / (det.unsqueeze(-1).unsqueeze(-1) + 1e-8)

            # Calculate Gain K
            K = C @ inv_denom

            # Apply K to dx
            dx_gated = (K @ dx.unsqueeze(-1)).squeeze(-1)  # Shape: (..., 2)

            # For matrix gating, we don't have a single scalar k.
            # We could return the trace or determinant, but for now let's return None
            # or maybe the average diagonal element of K?
            # Let's return None as it's not strictly defined.
            k = None
        else:
            assert False, f"Unknown gating method: {self.method}"

        return dx_gated, k
