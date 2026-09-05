from dataclasses import dataclass, field
from pathlib import Path

from ..utils import get_cached_path, format_dict
import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from typing import Literal

@dataclass
class Config:
    name: str
    learning_rate: float
    run: str

    batch_sizes: list[int] = field(default_factory=lambda: [64])
    others: dict[str, Any] = field(default_factory=dict, repr=True)

    def __init__(self, name: str, learning_rate: float, run: str = "r0", batch_sizes: list[int] | None = None, **kwargs):
        if batch_sizes is not None:
            self.batch_sizes = batch_sizes
        self.learning_rate = learning_rate
        self.name = name
        self.run = run
        self.others = kwargs

    def __getattr__(self, key):
        try:
            others = super().__getattribute__("others")
            if key in others:
                return others[key]
        except AttributeError:
            pass
        raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        # Check if it's a defined dataclass field
        if key in self.__dataclass_fields__ or key == "others":
            super().__setattr__(key, value)
        else:
            self.others[key] = value

    def __repr__(self):
        return format_dict(self.__class__.__name__, self.__dict__)

    def dir(self) -> Path:
        d = Path(get_cached_path("models")) / self.name
        d.absolute().mkdir(parents=False, exist_ok=True)
        return d

    def ckpts_dir(self) -> Path:
        m = self.dir() / "checkpoints" / self.run
        m.absolute().mkdir(parents=True, exist_ok=True)
        return m

    def runs_dir(self) -> Path:
        p = self.dir() / "runs" / self.run
        p.absolute().mkdir(parents=True, exist_ok=True)
        return p


torch.serialization.add_safe_globals([Config, np._core.multiarray.scalar, np.dtype, np.dtypes.Float32DType])  # type: ignore


def save(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    global_step: int,
    config: Config,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    **kwargs,
) -> None:
    for k, v in kwargs.items():
        if callable(v):
            kwargs[k] = v()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                [optimizer.state_dict() for optimizer in optimizer] if isinstance(optimizer, list) else optimizer.state_dict()
            ),
            "global_step": global_step,
            "config": config,
            "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
            **kwargs,
        },
        path,
    )


def load(
    path: str | Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    model_func=None,
    strict: bool = True,
    **kwargs,
) -> tuple[int, Config]:
    checkpoint = torch.load(path, map_location="cpu")
    if model is not None:
        model_state_dict = checkpoint["model_state_dict"]
        if model_func is not None:
            model_state_dict = model_func(checkpoint["model_state_dict"])
        model.load_state_dict(model_state_dict, strict=strict)
    try:
        if optimizer is not None:
            if isinstance(optimizer, list):
                for opt, state in zip(optimizer, checkpoint["optimizer_state_dict"]):
                    opt.load_state_dict(state)
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except ValueError as e:
        print(f"Warning: optimizer state could not be loaded due to mismatch: {e}")
    global_step = checkpoint["global_step"]
    config = checkpoint["config"]
    if not isinstance(config, Config):
        config = Config(**config)
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    for k, v in kwargs.items():
        if k not in checkpoint:
            print(f"Warning: key '{k}' not found in checkpoint during load(). Skipping.")
            continue
        if hasattr(v, "load_state_dict"):
            v.load_state_dict(checkpoint[k])
        elif callable(v):
            v(checkpoint[k])
        else:
            print(f"Warning: key '{k}' could not be loaded. Unsupported type {type(v)}.")
    return global_step, config


def optimizer_to(optim: list[torch.optim.Optimizer] | torch.optim.Optimizer, device: torch.device):
    """
    Move all optimizer state tensors to the specified device.
    """

    def impl(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    if isinstance(optim, list):
        for o in optim:
            impl(o, device)
    else:
        impl(optim, device)


def scheduler_to(sched: torch.optim.lr_scheduler._LRScheduler, device: torch.device):
    """
    Move all scheduler state tensors to the specified device.
    """
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


@torch.no_grad()
def calc_nmf(landmarks: torch.Tensor, landmarks_gt: torch.Tensor) -> float:
    """
    Calculate Normalized Mean Flicker (NMF) between predicted and ground truth landmarks.

    Args:
        landmarks: Predicted landmarks of shape (num_clips, clip_len, num_landmarks, 2).
        landmarks_gt: Ground truth landmarks of shape (num_clips, clip_len, num_landmarks, 2).
    Returns:
        float: The computed NMF value.

    The NMF was proposed in [1]: "Recurrence without Recurrence: Stable Video Landmark Detection with Deep Equilibrium Models".

        NMF = sqrt( mean_n ( mean_l ( || r(n,l) - r(n-1,l) ||^2 ) / d_S(n)^2 ) )
    where r(n, l) = p(n, l) - p_gt(n, l) is the error of landmark l in frame n,
    and d_S(n) = sqrt(w h) is the geometric mean of the bounding box width and height in frame n.
    """

    err = landmarks - landmarks_gt  # (num_clips, clip_len, num_landmarks, 2)
    bbox_size_gt = landmarks_gt.max(dim=2).values - landmarks_gt.min(dim=2).values  # (num_clips, clip_len, 2)

    STANDARD_AREA = 256 * 256  # [1] uses 256x256 as standard area in implementation.

    d_S_n = (bbox_size_gt.prod(dim=-1) / STANDARD_AREA).sqrt()  # (num_clips, clip_len)
    d_S_n = d_S_n[:, 1:]  # (num_clips, clip_len - 1)

    err_norm = err.diff(dim=1).square().sum(dim=-1).sqrt()  # (num_clips, clip_len - 1, num_landmarks)
    nmf_n_l = 100.0 * err_norm / d_S_n.unsqueeze(-1)  # (num_clips, clip_len - 1, num_landmarks)

    nmf_n = nmf_n_l.square().mean(dim=-1).sqrt()  # (num_clips, clip_len - 1)
    nmf = nmf_n.square().mean(dim=-1).sqrt()  # (num_clips,)
    return nmf.mean().item()


def navar_pos(m: int, predictions: list[torch.Tensor], gt: list[torch.Tensor]) -> torch.Tensor:
    """
    Calculate the position-based normalized Allan variance (NAVAR).

    Args:
        m: The block size.
        predictions: list of Tensor of shape (num_frames, num_landmarks, 2) containing the
            predicted landmark positions per video.
        gt: list of Tensor of shape (num_frames, num_landmarks, 2) containing the ground
            truth landmark positions per video.
    Returns:
        Tensor of shape (num_videos, num_landmarks) containing the NAVAR for each landmark in each video.
    """
    results: list[torch.Tensor] = []
    for preds, labels in zip(predictions, gt):
        num_frames = preds.shape[0]
        assert num_frames == labels.shape[0], "Number of frames in predictions and gt must match"
        if num_frames < 2 * m:
            continue

        bbox_min = torch.min(labels, dim=-2).values  # (num_frames, 2)
        bbox_max = torch.max(labels, dim=-2).values  # (num_frames, 2)
        bbox_size = torch.sqrt(torch.prod(bbox_max - bbox_min, dim=-1))  # (num_frames,)

        # Normalized error vectors per frame and landmark
        errors = (preds - labels) / bbox_size[:, None, None]  # (num_frames, num_landmarks, 2)

        blocks = errors.unfold(0, m, 1).mean(dim=-1)  # (num_frames - m + 1, num_landmarks, 2)
        vals = blocks[m:, :, :] - blocks[:-m, :, :]  # (num_frames - 2*m, num_landmarks, 2)

        vals = vals.square().sum(dim=-1)  # (num_frames - 2*m, num_landmarks)

        navar = vals.mean(dim=0) / 2  # (num_landmarks,)
        results.append(navar)
    return torch.stack(results, dim=0)  # (num_videos, num_landmarks)


def navar_pos_all(predictions: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Calculate position-based NAVAR for every valid window size at once.

    Args:
        predictions: Predicted positions of shape
            (num_videos, num_frames, num_landmarks, 2).
        gt: Ground-truth positions with the same shape as `predictions`.

    Returns:
        Tensor of shape (num_frames // 2, num_videos, num_landmarks), where
        entry `m - 1` contains the position-based NAVAR for window size `m`.
    """
    assert predictions.shape == gt.shape, "Predictions and ground truth must have the same shape"
    assert predictions.ndim == 4 and predictions.shape[-1] == 2, "Expected shape (videos, frames, landmarks, 2)"

    num_frames = predictions.shape[1]
    assert num_frames >= 2, "At least two frames are required to calculate NAVAR"

    bbox_min = gt.min(dim=2).values  # (num_videos, num_frames, 2)
    bbox_max = gt.max(dim=2).values  # (num_videos, num_frames, 2)
    bbox_size = (bbox_max - bbox_min).prod(dim=-1).sqrt()  # (num_videos, num_frames)
    errors = (predictions - gt) / bbox_size[..., None, None]  # (num_videos, num_frames, num_landmarks, 2)

    navar_values = []
    for m in range(1, num_frames // 2 + 1):
        blocks = errors.unfold(1, m, 1).mean(dim=-1)
        differences = blocks[:, m:] - blocks[:, :-m]
        navar_values.append(differences.square().sum(dim=-1).mean(dim=1) / 2)
    return torch.stack(navar_values, dim=0)


def clips_batch_indices(num_clips: int = 256, clip_len: int = 8, batch_size: int = 64):
    """
    Get batch indices for processing video clips in batches.

    Args:
        num_clips: Total number of clips.
        clip_len: Length of each clip.
        batch_size: Number of clips to process in a batch.

    Returns:
        torch.Tensor: A tensor of shape (num_clips, clip_len) containing the batch indices.
    """

    batch_indices = torch.arange(0, num_clips, dtype=torch.long).repeat_interleave(clip_len).reshape(
        -1, clip_len
    ) * clip_len + torch.arange(0, clip_len, dtype=torch.long).unsqueeze(0)
    batch_indices = batch_indices.T.reshape(clip_len, -1, batch_size).transpose(-3, -2).reshape(-1, batch_size)
    return batch_indices

NMENormType = Literal["iod", "size"]

@torch.no_grad()
def calc_nme(xy: torch.Tensor, labels: torch.Tensor, iod_indices: tuple[int, int] | None = None, norm_type: NMENormType = "iod") -> torch.Tensor:
    """
    Calculate the Normalized Mean Error (NME) between predicted and ground truth
    landmarks.
    
    Args:
        xy: Predicted landmarks of shape (..., nlandmarks, 2).
        labels: Ground truth landmarks of shape (..., nlandmarks, 2).
        iod_indices: Tuple of indices for the left and right eye corners in the
            landmark markup. Required if norm_type is "iod".
        norm_type: Type of normalization to use. Either "iod" for inter-ocular
            distance or "size" for bounding box size.
    Returns:
        Tensor of shape (...) containing the NME for each sample.
    """

    errors = (xy - labels).square().sum(dim=-1).sqrt()  # (..., num_queries)
    
    if norm_type == "iod":
        assert iod_indices is not None, "iod_indices must be provided when norm_type is 'iod'"

        # Normalize by inter-ocular distance (distance between eye corners), i.e. the
        # Euclidean distance between the two provided landmark indices.
        labels_iod = labels[..., iod_indices, :]  # (batch_size, 2, 2)
        norm_factor_iod = (labels_iod[..., 0, :] - labels_iod[..., 1, :]).square().sum(dim=-1).sqrt()  # (...,)
        norm_factor = norm_factor_iod.unsqueeze(-1) # (..., 1)
    elif norm_type == "size":
        norm_factor_s = (
            torch.max(labels, dim=-2).values - torch.min(labels, dim=-2).values
        )  # (batch_size, 2)
        norm_factor = norm_factor_s.prod(dim=-1).sqrt().unsqueeze(-1)  # (..., 1)
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Must be 'iod' or 'size'")

    nme = errors / norm_factor # (..., num_queries)
    return nme
