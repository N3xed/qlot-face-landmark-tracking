from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .cov import GenericCov2D, LowRankCov2D
from torch import nn
from threading import Lock
import torch.nn.functional as F
import utils.mesh
from utils.torch.datasets import CanonicalLandmarks


def preds_coords(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 0:2]


def preds_cov(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 2:5]


def preds_delta(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 5:7]


NUM_PREDS_COORDS = 2
NUM_PREDS_COV_PARAMS = 3
NUM_PREDS_DELTA = 2

NUM_PREDS_PARAMS = NUM_PREDS_COORDS + NUM_PREDS_COV_PARAMS + NUM_PREDS_DELTA


@dataclass
class LandmarkPrediction:
    mean: torch.Tensor  # Landmarks, shape: (..., num_queries, 2)
    cov: GenericCov2D  # Covariance matrix per landmark, shape: (..., num_queries, params_dim).
    delta: torch.Tensor  # Landmark deltas, shape: (..., num_queries, 2)

    def normalize_coords(self, image_width: int, image_height: int) -> "LandmarkPrediction":
        """
        Normalize the landmark coordinates to [-1, 1] range based on image dimensions.
        Args:
            image_width: Width of the image.
            image_height: Height of the image.
        Returns:
            LandmarkPrediction with normalized coordinates.
        """
        scale_x = 2.0 / image_width
        scale_y = 2.0 / image_height
        # Add 0.5 to center coordinates to align with align_corners=False convention
        # where -1 is the left edge (-0.5 spatial) and coordinate 0 is the pixel center.
        # normalized_mean = (self.mean + 0.5) * scale - 1.0
        normalized_mean = torch.stack(
            [
                self.mean[..., 0] * scale_x + (0.5 * scale_x - 1.0),
                self.mean[..., 1] * scale_y + (0.5 * scale_y - 1.0),
            ],
            dim=-1,
        )
        normalized_delta = torch.stack(
            [self.delta[..., 0] * scale_x, self.delta[..., 1] * scale_y],
            dim=-1,
        )
        cov = self.cov.scale_clamp(scale=(scale_x, scale_y))
        return LandmarkPrediction(mean=normalized_mean, cov=cov, delta=normalized_delta)

    def unnormalize_coords(self, image_width: int, image_height: int) -> "LandmarkPrediction":
        """
        Unnormalize the landmark coordinates from [-1, 1] range to pixel coordinates based on image dimensions.
        Args:
            image_width: Width of the image.
            image_height: Height of the image.
        Returns:
            LandmarkPrediction with unnormalized coordinates.
        """
        scale_x = image_width / 2.0
        scale_y = image_height / 2.0
        # Subtract 0.5 to return to pixel center coordinates
        # unnormalized_mean = (self.mean + 1.0) * scale - 0.5
        unnormalized_mean = torch.stack(
            [
                self.mean[..., 0] * scale_x + (scale_x - 0.5),
                self.mean[..., 1] * scale_y + (scale_y - 0.5),
            ],
            dim=-1,
        )

        unnormalized_delta = torch.stack(
            [self.delta[..., 0] * scale_x, self.delta[..., 1] * scale_y],
            dim=-1,
        )
        cov = self.cov.scale_clamp(scale=(scale_x, scale_y))
        return LandmarkPrediction(mean=unnormalized_mean, cov=cov, delta=unnormalized_delta)

    def unnormalize_coords_clamp(
        self,
        image_width: int,
        image_height: int,
        min: float | None,
        max: float | None,
        log_min_std_dev: float | None,
        log_max_std_dev: float | None,
    ) -> "LandmarkPrediction":
        """
        Clamp and unnormalize the landmark coordinates from [-1, 1] range to pixel coordinates based on image dimensions.
        Args:
            image_width: Width of the image.
            image_height: Height of the image.
            min: Minimum value to clamp the mean coordinates.
            max: Maximum value to clamp the mean coordinates.
            log_min_std_dev: Minimum value to clamp the log standard deviation of the covariance.
            log_max_std_dev: Maximum value to clamp the log standard deviation of the covariance.
        Returns:
            LandmarkPrediction with unnormalized coordinates.
        """
        scale_x = image_width / 2.0
        scale_y = image_height / 2.0

        # A clamp with no bounds is a no-op (e.g. the ONNX export path);
        # an unconditional clamp(min, max) raises under eager torch and
        # fails to export under dynamo.
        if min is not None or max is not None:
            mean = self.mean.clamp(min, max)
        else:
            mean = self.mean
        cov = self.cov.scale_clamp(
            scale=(scale_x, scale_y),
            log_min_stddev=log_min_std_dev,
            log_max_stddev=log_max_std_dev,
        )

        # Subtract 0.5 to return to pixel center coordinates
        # unnormalized_mean = (self.mean + 1.0) * scale - 0.5
        unnormalized_mean = torch.stack(
            [
                mean[..., 0] * scale_x + (scale_x - 0.5),
                mean[..., 1] * scale_y + (scale_y - 0.5),
            ],
            dim=-1,
        )
        unnormalized_delta = torch.stack(
            [self.delta[..., 0] * scale_x, self.delta[..., 1] * scale_y],
            dim=-1,
        )

        return LandmarkPrediction(mean=unnormalized_mean, cov=cov, delta=unnormalized_delta)

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([self.mean, self.cov.params, self.delta], dim=-1)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, cov_type: type = LowRankCov2D) -> "LandmarkPrediction":
        mean = preds_coords(tensor)
        cov = cov_type(preds_cov(tensor))
        delta = preds_delta(tensor)
        return LandmarkPrediction(mean=mean, cov=cov, delta=delta)

    def __getitem__(self, key):
        return LandmarkPrediction(mean=self.mean[key], cov=self.cov[key], delta=self.delta[key])

    def detach(self) -> "LandmarkPrediction":
        return LandmarkPrediction(mean=self.mean.detach(), cov=self.cov.detach(), delta=self.delta.detach())

    def clone(self) -> "LandmarkPrediction":
        return LandmarkPrediction(mean=self.mean.clone(), cov=self.cov.clone(), delta=self.delta.clone())

    def to_mean_max_variance(self) -> torch.Tensor:
        """
        Convert the LandmarkPrediction to a tensor containing mean and maximum variance.
        Returns:
            Tensor of shape (..., 3) containing (mean_x, mean_y, max_variance).
        """
        max_variance = self.cov.max_variance.unsqueeze(-1)
        return torch.cat([self.mean, max_variance], dim=-1)

    def record_stream(self, stream: torch.Stream) -> None:
        """
        Record the current CUDA stream for all tensors in the LandmarkPrediction.
        This ensures that operations on these tensors are synchronized with the given stream.
        Args:
            stream: The CUDA stream to record.
        """
        self.mean.record_stream(stream)
        self.cov.params.record_stream(stream)
        self.delta.record_stream(stream)


class QueryPoints(nn.Module):
    def __init__(self, init: dict[str, torch.Tensor], canonical_landmarks: dict[str, CanonicalLandmarks] | None = None):
        super().__init__()
        self.queries = nn.ParameterDict({k: nn.Parameter(v) for k, v in init.items()})
        self.canonical_landmarks = canonical_landmarks if canonical_landmarks is not None else {}
        self._cache = {}

    def clone(self) -> QueryPoints:
        return QueryPoints(init={k: v.clone() for k, v in self.queries.items()}, canonical_landmarks=self.canonical_landmarks)

    def get(self, key: str) -> torch.Tensor:
        """
        Get the query points for a specific dataset.
        """
        return self.queries[key]

    def _cache_value(self, key: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cache and retrieve flip indices, pupil indices, and pupil points for symmetry enforcement.
        """
        cache = self._cache.get(key, None)
        if cache is None or cache[0].device != device:
            cl = self.canonical_landmarks.get(key, None)
            assert cl is not None, f"canonical landmarks not provided for key '{key}'"
            flip_idx = torch.from_numpy(cl.flip_horizontal_indices()).long().to(device)
            pupils_idx = torch.tensor(cl.indices.pupils, device=device, dtype=torch.long)
            pupil_points = torch.from_numpy(cl.points[cl.indices.pupils]).float().to(device)
            cache = (flip_idx, pupils_idx, pupil_points)
            self._cache[key] = cache
        return cache

    def symmetrize(self, merge: list[list[str]] = []):
        """
        Symmetrize the query points by averaging with their mirrored counterparts.

        Args:
            merge: List of lists of dataset names whose query points should be merged by
              copying the points from the first dataset in the group.
        """

        for key in self.queries.keys():
            points = self.get(key)  # Shape: (num_queries, 3)

            # Enforce symmetry on the projected points
            flip_idx, pupil_indices, pupil_points = self._cache_value(key, points.device)

            # Get flipped points
            points_flipped = points[flip_idx].clone()

            # Mirror X
            points_flipped[:, 0] *= -1

            # Average
            points = (points + points_flipped) / 2.0

            # Force center X to 0
            center_mask = flip_idx == torch.arange(len(flip_idx), device=points.device)
            if center_mask.any():
                points[center_mask, 0] = 0.0

            # Keep pupils the same
            if len(pupil_indices) > 0:
                points[pupil_indices] = pupil_points
            self.queries[key].data.copy_(points)

        for group in merge:
            # Group must have at least two existing parameters to merge
            if len(group) < 2 and any(name not in self.queries for name in group):
                continue

            pts = self.queries[group[0]].data
            for name in group:
                self.queries[name].data.copy_(pts)

    def project_onto_mesh(self, mesh_projector: utils.mesh.MeshProjector) -> None:
        """
        Project the query points onto the mesh surface and enforce exact symmetry by averaging.

        Args:
            mesh_projector: MeshProjector to project points onto the mesh surface.
        """

        for key in self.queries.keys():
            points = self.get(key)  # Shape: (num_queries, 3)
            if points.requires_grad == False:
                continue

            projected_points_np = mesh_projector.project_points(points.detach().cpu().numpy())  # Shape: (num_queries, 3)
            projected_points = torch.from_numpy(projected_points_np).to(self.queries[key].device).type_as(self.queries[key])

            # Enforce symmetry on the projected points
            flip_idx, pupil_indices, pupil_points = self._cache_value(key, points.device)

            # Get flipped points
            points_flipped = projected_points[flip_idx].clone()

            # Mirror X
            points_flipped[:, 0] *= -1

            # Average
            projected_points = (projected_points + points_flipped) / 2.0

            # Force center X to 0
            center_mask = flip_idx == torch.arange(len(flip_idx), device=points.device)
            if center_mask.any():
                projected_points[center_mask, 0] = 0.0

            # Keep pupils the same
            if len(pupil_indices) > 0:
                projected_points[pupil_indices] = pupil_points

            self.queries[key].data = projected_points

    def project_gradients_onto_mesh(self, mesh_projector: utils.mesh.MeshProjector) -> None:
        """
        Project the gradients of the query points to lie tangent to the mesh surface,
        keeping vector lenghts unchanged.

        Args:
            mesh_projector: MeshProjector to compute normals at query points.
        """

        for key in self.queries.keys():
            points = self.get(key)
            if points.grad is None:
                continue

            # Get normals at the current points
            points_np = points.detach().cpu().numpy()
            normals_np = mesh_projector.get_normals(points_np)
            normals = torch.from_numpy(normals_np).to(points.device).type_as(points)

            # Ensure normals are unit vectors
            normals = torch.nn.functional.normalize(normals, dim=-1)

            grad = points.grad
            grad_mag = torch.norm(grad, dim=-1, keepdim=True)

            # Project gradient onto tangent plane: G_tangent = G - (G . N) * N
            dot = (grad * normals).sum(dim=-1, keepdim=True)
            grad_tangent = grad - dot * normals

            # Renormalize tangent gradient to have the original magnitude
            grad_tangent_mag = torch.norm(grad_tangent, dim=-1, keepdim=True)
            grad_projected = grad_tangent / (grad_tangent_mag + 1e-8) * grad_mag

            points.grad.copy_(grad_projected)

    def enforce_gradient_symmetry(self, merge: list[list[str]] = []) -> None:
        """
        Manually symmetrizes gradients for query points before optimizer step.
        Assumes X-axis is the lateral axis (left-right).

        Args:
            merge: List of lists of dataset names whose query points should share gradients.
        """

        for name, param in self.queries.items():
            if param.grad is None:
                continue

            flip_idx, pupil_indices, _ = self._cache_value(name, param.device)
            grad = param.grad

            # Get gradients of the flipped points
            grad_flipped = grad[flip_idx].clone()

            # Mirror the X-axis of the flipped gradients
            grad_flipped[:, 0] *= -1

            # Average the original gradients and the mirrored flipped gradients
            avg_grad = (grad + grad_flipped) / 2.0

            # Assign the averaged gradient back
            grad.copy_(avg_grad)

            # Zero gradients for pupils to keep them fixed
            if len(pupil_indices) > 0:
                grad[pupil_indices] = 0.0

            # Explicitly zero out x-gradient for center points
            center_mask = flip_idx == torch.arange(len(flip_idx), device=param.device)
            if center_mask.any():
                grad[center_mask, 0] = 0.0

        for group in merge:
            # Group must have at least two existing parameters to merge
            if len(group) < 2 and any(name not in self.queries for name in group):
                continue

            # Compute the average gradient across all parameters in the group
            grads = [self.queries[name].grad for name in group if self.queries[name].grad is not None]
            if not grads:
                continue
            avg_grad = torch.stack(grads, dim=0).mean(dim=0)

            # Assign the average gradient back to each parameter in the group
            for name in group:
                if self.queries[name].grad is not None:
                    self.queries[name].grad.copy_(avg_grad)

    def enforce_optimizer_state_symmetry(self, optimizer: torch.optim.Optimizer, merge: list[list[str]] = []) -> None:
        """
        Symmetrizes the optimizer state (momentum/variance) for query points.
        This prevents adaptive optimizers like AdamW from developing asymmetric
        per-parameter state even when gradients are symmetrized.

        Args:
            optimizer: The optimizer containing state for query point parameters.
            merge: List of lists of dataset names whose query points should share optimizer state.
        """
        for name, param in self.queries.items():
            if not param.requires_grad:
                continue

            state = optimizer.state.get(param, None)
            if state is None:
                continue

            flip_idx, pupil_indices, _ = self._cache_value(name, param.device)
            center_mask = flip_idx == torch.arange(len(flip_idx), device=param.device)

            # Helper to symmetrize a buffer (momentum or variance)
            def symmetrize_buffer(buffer_name, flip_sign=False):
                if buffer_name in state:
                    buf = state[buffer_name]  # type: ignore
                    buf_flipped = buf[flip_idx].clone()

                    if flip_sign:
                        # Mirror X-axis (momentum direction should flip for x)
                        buf_flipped[:, 0] *= -1

                    # Average
                    buf_sym = (buf + buf_flipped) / 2.0

                    if flip_sign:
                        # Zero x-momentum for center points
                        if center_mask.any():
                            buf_sym[center_mask, 0] = 0.0
                        # Zero momentum for pupils
                        if len(pupil_indices) > 0:
                            buf_sym[pupil_indices] = 0.0

                    buf.copy_(buf_sym)

            # Symmetrize AdamW states
            symmetrize_buffer("exp_avg", flip_sign=True)  # Momentum
            symmetrize_buffer("exp_avg_sq", flip_sign=False)  # Variance

            # Symmetrize SGD state
            symmetrize_buffer("momentum_buffer", flip_sign=True)

        # Merge optimizer state across groups
        for group in merge:
            if len(group) < 2:
                continue

            params_in_group = [self.queries[n] for n in group if n in self.queries and self.queries[n].requires_grad]
            if len(params_in_group) < 2:
                continue

            states = [optimizer.state.get(p) for p in params_in_group]
            states = [s for s in states if s is not None]
            if not states:
                continue

            for key in ["exp_avg", "exp_avg_sq", "momentum_buffer"]:
                if all(key in s for s in states):
                    avg_val = torch.stack([s[key] for s in states], dim=0).mean(dim=0)
                    for s in states:
                        s[key].copy_(avg_val)


def init(m: Any, nonlinearity: nn.init._NonlinearityType):
    """Init linear/conv weights with Kaiming normal and biases with zeros."""
    assert isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
    nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class FlushStream(ABC):
    """Interface for modules with stream-local state that must be merged."""

    @abstractmethod
    def flush_stream(self) -> None:
        """Merge stream-local state into the module's canonical state."""


@dataclass
class _StreamBatchNormState:
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor
    num_updates: int = 0


class StreamSafeBatchNorm(nn.BatchNorm2d, FlushStream):
    """BatchNorm with independent running statistics for each CUDA stream."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self._stream_states: dict[tuple[int, int], _StreamBatchNormState] = {}
        self._stream_states_lock = Lock()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda" or not self.training or not self.track_running_stats:
            return super().forward(input)
        return self._forward_stream_safe(input)

    @torch._dynamo.disable()
    def _forward_stream_safe(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        device_index = input.device.index
        stream = torch.cuda.current_stream(device=input.device)
        key = (device_index, stream.cuda_stream)

        state = self._stream_states.get(key)
        if state is None:
            # Lazily add a stream-local state.
            with self._stream_states_lock:
                assert self.running_mean is not None
                assert self.running_var is not None
                assert self.num_batches_tracked is not None
                state = _StreamBatchNormState(
                    running_mean=self.running_mean.clone(),
                    running_var=self.running_var.clone(),
                    num_batches_tracked=self.num_batches_tracked.clone(),
                )
                self._stream_states[key] = state
        state.num_updates += 1

        state.num_batches_tracked.add_(1)
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(state.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum

        return F.batch_norm(
            input,
            state.running_mean,
            state.running_var,
            self.weight,
            self.bias,
            training=True,
            momentum=exponential_average_factor,
            eps=self.eps,
        )

    @torch.no_grad()
    def flush_stream(self) -> None:
        """Merge completed stream-local updates into the canonical running statistics."""
        with self._stream_states_lock:
            states = [state for _, state in sorted(self._stream_states.items())]
            active_states = [state for state in states if state.num_updates > 0]
            if not active_states:
                return

            assert self.running_mean is not None
            assert self.running_var is not None
            assert self.num_batches_tracked is not None

            base_mean = self.running_mean.clone()
            base_var = self.running_var.clone()
            total_updates = sum(state.num_updates for state in active_states)

            if self.momentum is None:
                base_count = self.num_batches_tracked.clone()
                total_count = base_count + total_updates
                merged_mean = base_mean * base_count
                merged_var = base_var * base_count
                for state in active_states:
                    local_count = base_count + state.num_updates
                    merged_mean.add_(state.running_mean * local_count - base_mean * base_count)
                    merged_var.add_(state.running_var * local_count - base_var * base_count)
                self.running_mean.copy_(merged_mean / total_count)
                self.running_var.copy_(merged_var / total_count)
            else:
                for state in active_states:
                    decay = (1.0 - self.momentum) ** state.num_updates
                    self.running_mean.mul_(decay).add_(state.running_mean - base_mean * decay)
                    self.running_var.mul_(decay).add_(state.running_var - base_var * decay)

            self.num_batches_tracked.add_(total_updates)
            for state in states:
                state.running_mean.copy_(self.running_mean)
                state.running_var.copy_(self.running_var)
                state.num_batches_tracked.copy_(self.num_batches_tracked)
                state.num_updates = 0
