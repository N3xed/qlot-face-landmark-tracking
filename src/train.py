from typing import Iterator, Callable
import torch
import data
import utils.torch
from dataclasses import dataclass, field
from utils.torch.datasets import DataLoader
from utils.torch.misc import Config, optimizer_to, save, load
from utils.torch.optim import opt_param_cfg, ParamCfg
from model import QLOT, LowRankCov2D, Cov2D
from torch.utils.tensorboard import SummaryWriter
from model.utils import LandmarkPrediction, QueryPoints
from tqdm import tqdm
import numpy as np
import torchvision
import matplotlib
import torch.profiler
import warnings
import copy
import utils
from utils.torch.viz import draw_keypoints, map_to_color
from utils.torch.datasets import QueriedFaceDataset, Batch, StridedClipSampler
import torchvision.transforms.v2.functional as TF
import cv2
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, Future
from functools import partial

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from abc import ABC, abstractmethod


class DataLoadersBase(ABC):
    @abstractmethod
    def train_loaders(self) -> list[DataLoader]:
        pass

    @abstractmethod
    def valid_image_loaders(self) -> list[DataLoader]:
        pass

    @abstractmethod
    def valid_video_loaders(self) -> list[DataLoader]:
        pass


@dataclass
class DataLoaders(DataLoadersBase):
    face_synth: DataLoader
    wflw: DataLoader
    wflw_v: DataLoader
    ibug: DataLoader

    valid: DataLoader

    valid_jitter: torch.utils.data.DataLoader | None = None
    valid_jitter_dataset: utils.torch.datasets.QueriedFaceDataset | None = None

    def __init__(self, datasets: data.Datasets, cfg: Config, rng: np.random.Generator | None = None):
        try:
            face_synth_bs, wflw_bs, wflw_v_bs, ibug_bs = cfg.batch_sizes
        except:
            face_synth_bs, wflw_bs, wflw_v_bs, ibug_bs = 16, 8, 8, 4
            print(f"Using default batch sizes for dataloaders: ({face_synth_bs}, {wflw_bs}, {wflw_v_bs}, {ibug_bs})")

        # Derive distinct generators per training loader so their batch-order streams are independent
        # yet reproducible from the single TrainParams.rng.
        def _seed(rng: np.random.Generator | None) -> torch.Generator | None:
            if rng is None:
                return None
            seed = int(rng.integers(0, 2**31))
            return torch.Generator().manual_seed(seed)

        self.wflw_v = DataLoader(
            datasets.wflw_v,
            batch_size=wflw_v_bs,
            num_workers=2,
            persistent_workers=True,
            generator=_seed(rng),
            stream_idx=0,
        )
        self.face_synth = DataLoader(
            datasets.face_synth,
            batch_size=face_synth_bs,
            num_workers=1,
            persistent_workers=True,
            generator=_seed(rng),
            stream_idx=1,
        )
        self.wflw = DataLoader(
            datasets.wflw, batch_size=wflw_bs, num_workers=1, persistent_workers=True, generator=_seed(rng), stream_idx=2
        )
        self.ibug = DataLoader(
            datasets.ibug, batch_size=ibug_bs, num_workers=1, persistent_workers=True, generator=_seed(rng), stream_idx=3
        )

        self.wflw_valid = DataLoader(
            datasets.wflw_val,
            batch_size=128,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            num_workers=1,
            stream_idx=0,
        )
        self.ibug_valid = DataLoader(
            datasets.ibug_val,
            batch_size=128,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            num_workers=1,
            stream_idx=1,
        )
        self.face_synth_valid = DataLoader(
            datasets.face_synth_val,
            batch_size=256,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            num_workers=2,
            stream_idx=2,
        )

        # Sharded video validation loaders.
        self.wflw_v_valid0 = DataLoader(
            datasets.wflw_v_val,
            batch_size=16,
            num_workers=1,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            sampler=StridedClipSampler(num_clips=len(datasets.wflw_v_val), shard_id=0, num_shards=4),
            stream_idx=0,
        )
        self.wflw_v_valid1 = DataLoader(
            datasets.wflw_v_val,
            batch_size=16,
            num_workers=1,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            sampler=StridedClipSampler(num_clips=len(datasets.wflw_v_val), shard_id=1, num_shards=4),
            stream_idx=1,
        )
        self.wflw_v_valid2 = DataLoader(
            datasets.wflw_v_val,
            batch_size=16,
            num_workers=1,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            sampler=StridedClipSampler(num_clips=len(datasets.wflw_v_val), shard_id=2, num_shards=4),
            stream_idx=2,
        )
        self.wflw_v_valid3 = DataLoader(
            datasets.wflw_v_val,
            batch_size=16,
            num_workers=1,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=1,
            sampler=StridedClipSampler(num_clips=len(datasets.wflw_v_val), shard_id=3, num_shards=4),
            stream_idx=3,
        )

    def train_loaders(self) -> list[DataLoader]:
        return [self.wflw_v, self.face_synth, self.wflw, self.ibug]

    def valid_image_loaders(self) -> list[DataLoader]:
        return [self.face_synth_valid, self.wflw_valid, self.ibug_valid]

    def valid_video_loaders(self) -> list[DataLoader]:
        return [self.wflw_v_valid0, self.wflw_v_valid1, self.wflw_v_valid2, self.wflw_v_valid3]


@dataclass
class TrainParams:
    model: QLOT
    optimizer: torch.optim.Optimizer
    query_points: QueryPoints
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    global_step: int = 0
    device: torch.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    epochs: int = 180
    steps_per_epoch: int = 1000
    extra_save_args: dict | None = None
    model_iterations: int = 3
    init_backbone: bool = False

    small_validate_every: int = 50
    big_validate_every: int = 100
    log_images_every: int = 20
    log_every: int = 10
    save_latest_every: int = 200
    sparsity_loss_enabled: bool = False  # TODO: validate this
    model_steps: int = 50
    query_steps: int = 32
    freeze_backbone_steps: int = 400

    face_mesh: Path | None = None
    freeze_queries: list[str] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    nme_coeff: float = 1.01
    _parallel_streams: dict[int, torch.Stream] = field(default_factory=dict, repr=False, init=False)
    _default_stream: torch.Stream | None = field(default=None, repr=False, init=False)
    _thread_pool: ThreadPoolExecutor = field(repr=False, init=False)
    stream_parallellism: bool = True

    # Pre-calculate iteration weights for deep supervision
    MAX_ITERS = 12
    iter_weights: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        self.iter_weights = torch.tensor(
            [0.85 ** (self.MAX_ITERS - 1 - i) for i in range(self.MAX_ITERS)], device=self.device
        ).unsqueeze(
            -1
        )  # (iterations, 1)

        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    def get_stream(self, idx: int) -> torch.Stream | None:
        if not self.stream_parallellism:
            return None
        if self.device.type == "cpu":
            if idx != 0:
                warnings.warn("cuda stream parallelism is not available: device is not CUDA")
            return None

        s = self._parallel_streams.get(idx)
        if s is None:
            s = torch.Stream(device=self.device, priority=0)
            self._parallel_streams[idx] = s
        return s

    def get_default_stream(self) -> torch.Stream | None:
        if self.device.type != "cuda":
            return None
        if self._default_stream is None:
            self._default_stream = torch.cuda.default_stream(device=self.device)
        return self._default_stream

    def get_rng_state(self) -> dict:
        """Get the owned RNG state: TrainParams.rng and PyTorch CPU/CUDA global RNG.

        Global Python/NumPy RNG state is intentionally excluded.
        """
        state: dict = {
            "version": 1,
            "train_rng": copy.deepcopy(self.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state(),
            "albumentations": data.get_aug_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def set_rng_state(self, state: dict):
        """Set the owned RNG state.

        Tolerates older checkpoints without an ``rng_state`` field or with partial state.
        Does not touch global Python/NumPy RNG.
        """
        if not isinstance(state, dict):
            return
        train_rng = state.get("train_rng")
        if train_rng is not None:
            self.rng.bit_generator.state = train_rng
        torch_cpu = state.get("torch_cpu")
        if torch_cpu is not None:
            torch.set_rng_state(torch_cpu)
        if torch.cuda.is_available():
            torch_cuda = state.get("torch_cuda")
            if torch_cuda is not None:
                torch.cuda.set_rng_state_all(torch_cuda)
        aug_state = state.get("albumentations")
        if aug_state is not None:
            data.set_aug_state(aug_state)


# Assumed AR(1) temporal correlation of the per-landmark prediction error between consecutive
# frames. Consecutive predictions share a recurrent hidden state and are warm-started from the
# previous frame, so their errors are strongly correlated rather than independent.
#
# This is used to propagate the spatial covariance Sigma_t through temporal differences of the
# predictions. Treating the frames as independent overestimates the variance of those differences,
# which (for a live-covariance GNLL such as the acceleration term) pushes Sigma_t toward collapse
# so that the propagated sum still matches the small observed residual.
#
# Set to 0.0 to recover the independent-error assumption.
TEMPORAL_ERROR_CORR = 0.74


def schedule_num_iterations(params: TrainParams, rng: np.random.Generator) -> int:
    # 6 to 3 over 10k steps
    if rng.random() < 0.001:
        return 12
    if params.global_step >= 10000:
        return 1 if rng.random() < 0.05 else int(rng.integers(low=2, high=params.model_iterations, endpoint=True))
    elif params.global_step >= 5000:
        return int(rng.integers(low=2, high=4, endpoint=True))
    elif params.global_step >= 2000:
        return int(rng.integers(low=2, high=5, endpoint=True))
    else:
        return int(rng.integers(low=2, high=6, endpoint=True))


@dataclass(init=False, repr=False)
class ProcessBatch:
    acc_l1_loss: torch.Tensor
    acc_huber_loss: torch.Tensor
    acc_gnll_loss: torch.Tensor
    acc_delta_gnll_loss: torch.Tensor
    acc_acceleration_gnll_loss: torch.Tensor
    acc_hconsistency_loss: torch.Tensor

    batch_loss: torch.Tensor
    params: TrainParams

    initial_preds: LandmarkPrediction | None
    initial_idx: tuple[torch.Tensor, ...] | None
    final_preds: LandmarkPrediction | None
    final_idx: tuple[torch.Tensor, ...] | None

    is_viz: bool
    rng: np.random.Generator

    def __init__(self, params: TrainParams, rng: np.random.Generator | None = None, is_viz: bool = False):
        self.params = params
        self.is_viz = is_viz

        self.acc_l1_loss = torch.zeros(1, device=params.device)
        self.acc_huber_loss = torch.zeros(1, device=params.device)
        self.acc_gnll_loss = torch.zeros(1, device=params.device)
        self.acc_delta_gnll_loss = torch.zeros(1, device=params.device)
        self.acc_acceleration_gnll_loss = torch.zeros(1, device=params.device)
        self.acc_hconsistency_loss = torch.zeros(1, device=params.device)

        self.batch_loss = torch.zeros(1, device=params.device)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.initial_preds = None
        self.initial_idx = None
        self.final_preds = None
        self.final_idx = None

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        yield self.acc_l1_loss
        yield self.acc_huber_loss
        yield self.acc_gnll_loss
        yield self.acc_delta_gnll_loss
        yield self.acc_acceleration_gnll_loss
        yield self.acc_hconsistency_loss
        yield self.batch_loss

    def detach(self) -> None:
        self.batch_loss.detach_()
        # Note: All other tensors are already detached.

    def acc_loss(
        self,
        l1_loss: torch.Tensor,
        huber_loss: torch.Tensor,
        gnll_loss: torch.Tensor,
        delta_gnll_loss: torch.Tensor,
        acceleration_gnll_loss: torch.Tensor,
        hconsistency_loss: torch.Tensor,
        batch_loss: torch.Tensor | None = None,
    ) -> None:
        if batch_loss is not None:
            self.batch_loss += batch_loss.detach()
        else:
            self.batch_loss += (
                1.0 * l1_loss
                + 1.0 * gnll_loss
                + 0.5 * delta_gnll_loss
                + 0.25 * acceleration_gnll_loss
                + (4.0 * max(0.0, min(1.0, (self.params.global_step - 2000.0) / 10000.0))) * hconsistency_loss
            )

        self.acc_l1_loss += l1_loss.detach()
        self.acc_gnll_loss += gnll_loss.detach()
        self.acc_delta_gnll_loss += delta_gnll_loss.detach()
        self.acc_acceleration_gnll_loss += acceleration_gnll_loss.detach()
        self.acc_hconsistency_loss += hconsistency_loss.detach()
        self.acc_huber_loss += huber_loss.detach()

    def process(self, batch: Batch, model: QLOT) -> None:
        nqueries = batch.queries.shape[-2]
        params = self.params

        predictions: LandmarkPrediction | None = None
        hidden_state = None

        batch_idx = torch.arange(batch.orig_batch_size, device=params.device, dtype=torch.long)
        frame_idx = torch.zeros(batch.orig_batch_size, device=batch.images.device, dtype=torch.long)
        if self.is_viz:
            self.initial_idx = (batch_idx, frame_idx.clone())

        prev_labels = None
        prev_pred_mean_detached = None
        prev_pred_cov_detached = None

        prev_2_labels = None
        prev_2_pred_mean_detached = None
        prev_2_pred_cov_detached = None

        # Sample number of iterations for this clip
        curr_iters = schedule_num_iterations(params, self.rng)

        curr_batch: Batch | None = None
        preds: LandmarkPrediction | None = None

        for frame_i in range(batch.orig_clip_len):
            curr_batch = batch[batch_idx, frame_idx]

            starting_predictions = predictions
            if self.rng.random() < 0.005:
                # Randomly reset the hidden state and starting predictions to None to encourage the model to recover from scratch
                starting_predictions = None
                hidden_state = None

            if starting_predictions is None:
                prev_labels = None
                prev_pred_mean_detached = None
                prev_pred_cov_detached = None
                prev_2_labels = None
                prev_2_pred_mean_detached = None
                prev_2_pred_cov_detached = None

            preds, hidden_state = model(
                curr_batch.images,
                curr_batch.queries,
                iterations=curr_iters,
                return_sequence=True,
                store_similarity_maps=False,
                return_hidden_state=True,
                prefill_hidden_state=hidden_state,
                prefill_starting_landmarks=starting_predictions,
                detach_updates=True,
            )

            assert preds is not None

            # --- TTA Bias Cancellation for Paired Flip Batches ---
            hconsistency_loss = torch.zeros(1, 1, 1, device=preds.mean.device)  # Default to zero if not computed
            if batch.dataset.include_flipped:
                # The batch is interleaved [Original, Flipped, Original, Flipped, ...]
                W_img = batch.images.shape[-1]

                # 1. Spatial Consistency
                new_mean = preds.mean.clone()  # Shape: (iters, batch_size, queries, 2)
                pred_orig = new_mean[:, 0::2].clone()
                pred_flip = new_mean[:, 1::2].clone()

                # Unflip the flipped coordinates geometrically to match original
                flip_indices = batch.dataset.flip_indices
                unflipped_mean = pred_flip[:, :, flip_indices, :].clone()
                unflipped_mean[..., 0] = (W_img - 1.0) - unflipped_mean[..., 0]

                spatial_hconsistency = (pred_orig - unflipped_mean).square().sum(dim=-1)  # (its, batch/2, queries)

                # 2. Covariance Consistency
                cov_params = preds.cov.params.clone()  # (iters, batch_size, queries, 3)
                cov_orig = cov_params[:, 0::2].clone()
                cov_flip = cov_params[:, 1::2].clone()

                # Unflip the parameters and negate the correlation (index 2)
                unflipped_cov = cov_flip[:, :, flip_indices, :].clone()
                unflipped_cov[..., 2] = -unflipped_cov[..., 2]

                cov_hconsistency = (cov_orig - unflipped_cov).abs().sum(dim=-1)  # L1 on log_sigma_x, log_sigma_y, rho

                hconsistency_loss = spatial_hconsistency + 0.1 * cov_hconsistency

            if self.is_viz and frame_i == 0:
                # For logging
                self.initial_preds = preds.detach()[-1]

            # --- Composite Iterative Loss ---

            target_expanded = curr_batch.labels.expand(curr_iters, -1, -1, -1)  # (iter, batch, queries, 2)
            weights_expanded = curr_batch.weights.expand(curr_iters, -1, -1)  # (iter, batch, queries)

            # 1. Split Loss: L2 for Accuracy, GNLL (detached) for Uncertainty
            diff = preds.mean - target_expanded

            # L2 Loss
            l1_loss = (diff.square().sum(dim=-1) + 1e-8).sqrt()  # (iterations, batch, queries)
            huber_loss = torch.nn.functional.huber_loss(preds.mean.detach(), target_expanded, reduction="none", delta=10.0).sum(
                dim=-1
            )  # (iterations, batch, queries)

            # GNLL Loss (Accuracy Detached)
            # We detach the difference so the mean is not updated by the covariance loss.
            # The two spatial terms are complementary with non-overlapping jobs. Exactly one side is live in each,
            # so neither degenerates: with only GNLL where mean and the covariance are coupled, the optimizer can
            # reduce the loss by inflating Sigma instead of fixing the prediction -- the standard heteroscedastic-NLL
            # failure mode.
            gnll_loss = preds.cov.gaussian_negative_log_likelihood(diff.detach())

            l1_loss_weighted = (l1_loss * weights_expanded).sum(dim=-1)  # (iterations, batch)
            huber_loss_weighted = (huber_loss * weights_expanded).sum(dim=-1)  # (iterations, batch)
            gnll_loss_weighted = (gnll_loss * weights_expanded).sum(dim=-1)  # (iterations, batch)

            # 2. Temporal GNLL terms (Dynamic Uncertainty)
            # Train coordinate trajectory to be smooth when the covariance is low.
            # We assume uncorrelated errors between frames since the covariance is detached.
            # Note: For reproduibility reasons, delta GNLL still reweights the covariance
            # by AR(1) estimate. A previous implementation supervised covariance with the
            # delta GNLL (detaching the residual).
            #
            # We compute two temporal GNLLs:
            # - First-order (delta): Smoothes the velocity of the trajectory.
            # - Second-order (acceleration): Smoothes the curvature of the trajectory.
            # Both live in *pixel space*: the residual is a displacement in pixels and the
            # covariance is a spatial covariance in pixels^2. We deliberately do not rescale by FPS:
            # the residual and the covariance must share units for the GNLL to be a valid
            # likelihood, and a unit-consistent rescaling only adds a per-clip constant to the
            # log-det term (zero gradient). The clip-to-clip variation in frame spacing is already
            # carried by the magnitude of the pixel displacement itself.
            acceleration_gnll_weighted = torch.zeros_like(l1_loss_weighted)
            delta_gnll_weighted = torch.zeros_like(l1_loss_weighted)

            if frame_i > 0 and prev_labels is not None and prev_pred_mean_detached is not None:
                assert prev_pred_cov_detached is not None
                assert isinstance(preds.cov, LowRankCov2D)

                # --- First-order (delta): velocity consistency of the trajectory ---
                # delta = mu_t - mu_{t-1}, delta_gt = y_t - y_{t-1}, both in pixels.
                #
                # The residual is LIVE and the covariance is DETACHED, so this term trains the
                # *mean*: it penalizes frame-to-frame deviations of the predicted velocity from
                # the ground-truth velocity, making it a trajectory-smoothness (anti-jitter)
                # objective rather than a covariance-calibration one. (A previous implementation
                # did the opposite -- detached residual, live covariance -- supervising Sigma_t
                # with the temporal residual; the static per-frame `gnll_loss` now carries the
                # covariance supervision.)
                #
                # Weighting the residual by the inverse covariance (Mahalanobis metric) turns a
                # plain smoothness penalty into an *adaptive* one, which is what makes it useful
                # for anti-jitter:
                # - Confidence-gated smoothing. The gradient w.r.t. mu_t is Sigma^{-1} r, so
                #   landmarks the model is confident about (low sigma) are forced to be
                #   temporally consistent, while uncertain ones (occlusion, blur, fast motion)
                #   may move freely. Jitter is precisely the oscillation of predictions the model
                #   believes are stable, so the penalty is strongest exactly where jitter lives,
                #   without introducing lag on genuinely moving landmarks.
                # - Per-landmark, per-frame scale. Sigma is learned by the static per-frame GNLL,
                #   so the smoothing strength adapts to each landmark's current difficulty (and
                #   is dimensionless), instead of relying on a single hand-tuned weight.
                #
                # With Sigma detached, the AR(1) shrink factor only rescales the inverse weight
                # by the constant 1/(1 - rho) -- equivalent to retuning this term's loss weight
                # (0.5) -- and cannot corrupt the covariance. It is retained for reproducibility
                # with earlier runs.
                delta = preds.mean - prev_pred_mean_detached.unsqueeze(0)
                delta_gt = target_expanded - prev_labels.unsqueeze(0)

                cov_delta_sum = preds.cov + LowRankCov2D(params=prev_pred_cov_detached.params.unsqueeze(0))
                delta_shrink = max(1.0 - TEMPORAL_ERROR_CORR, 1e-3)
                cov_delta = cov_delta_sum.scale_clamp(scale=delta_shrink**0.5).detach()

                delta_gnll_weighted = (cov_delta.gaussian_negative_log_likelihood(delta - delta_gt) * weights_expanded).sum(
                    dim=-1
                )

                # --- Second-order (acceleration) consistency ---
                if (
                    frame_i > 1
                    and prev_2_labels is not None
                    and prev_2_pred_mean_detached is not None
                    and prev_2_pred_cov_detached is not None
                ):
                    # Second difference of the ground truth: (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}).
                    a_gt = (curr_batch.labels - prev_labels) - (prev_labels - prev_2_labels)
                    a_gt_expanded = a_gt.expand(curr_iters, -1, -1, -1)

                    # Second difference of the predictions. The two previous means are detached,
                    # which keeps backprop through time bounded (only mu_t carries gradient).
                    a_pred = delta - (prev_pred_mean_detached - prev_2_pred_mean_detached).unsqueeze(0)

                    # The residual is LIVE and the covariance is DETACHED, so this term trains the
                    # *mean*: it is a trajectory-curvature (smoothness) objective. A second
                    # difference is a high-pass filter, which makes it far more sensitive to jitter
                    # than the first-order delta term.
                    #
                    # Since the covariance is only an inverse weight here, an over- or
                    # under-estimated propagated variance merely rescales this term's gradient
                    # (equivalent to retuning its loss weight) and cannot corrupt Sigma. The
                    # correlation constant is therefore uncritical for this term.
                    a_diff = a_pred - a_gt_expanded

                    # Propagate Sigma through the second difference e_t - 2*e_{t-1} + e_{t-2}.
                    # Under independent errors this is Sigma_t + 4*Sigma_{t-1} + Sigma_{t-2}.
                    cov_acc_params = (
                        preds.cov.as_cov2d_params().detach()
                        + 4.0 * prev_pred_cov_detached.as_cov2d_params().unsqueeze(0)
                        + prev_2_pred_cov_detached.as_cov2d_params().unsqueeze(0)
                    )
                    cov_acc = Cov2D(params=cov_acc_params)

                    # GNLL Acceleration Loss
                    acceleration_gnll_weighted = (cov_acc.gaussian_negative_log_likelihood(a_diff) * weights_expanded).sum(dim=-1)

            # Aggregate Components
            curr_iter_weights = params.iter_weights[-curr_iters:]  # (curr_iters, 1)
            iter_norm = curr_iter_weights[-params.model_iterations + 1 :].sum()

            norm = nqueries * curr_iters / batch.global_weight
            spatial_norm = batch.global_weight / (nqueries * iter_norm * batch.orig_clip_len)

            l1_loss_norm = (l1_loss_weighted * curr_iter_weights).sum(dim=0).mean() * spatial_norm
            huber_loss_norm = (huber_loss_weighted * curr_iter_weights).sum(dim=0).mean() * spatial_norm
            gnll_loss_norm = (gnll_loss_weighted).sum(dim=0).mean() / (norm * batch.orig_clip_len)
            acceleration_gnll_norm = (acceleration_gnll_weighted).sum(dim=0).mean() / (norm * max(batch.orig_clip_len - 2, 1))
            delta_gnll_norm = (delta_gnll_weighted).sum(dim=0).mean() / (norm * max(batch.orig_clip_len - 1, 1))
            hconsistency_loss_norm = hconsistency_loss.sum(dim=0).mean() / (nqueries * batch.orig_clip_len)

            self.acc_loss(
                l1_loss_norm,
                huber_loss_norm,
                gnll_loss_norm,
                delta_gnll_norm,
                acceleration_gnll_norm,
                hconsistency_loss_norm,
            )

            # Update previous state
            prev_2_labels = prev_labels
            prev_2_pred_mean_detached = prev_pred_mean_detached
            prev_2_pred_cov_detached = prev_pred_cov_detached
            prev_labels = curr_batch.labels
            prev_pred_mean_detached = preds.mean[-1].detach()
            prev_pred_cov_detached = preds.cov[-1].detach()
            predictions = preds[-1]

            # For logging
            if frame_i == batch.orig_clip_len - 1:
                self.final_preds = preds.detach()[-1]
                self.final_idx = (batch_idx, frame_idx.clone())

            # Increment frame index for next iteration
            frame_idx = frame_idx + 1

    def backward(self, buf: list[torch.Tensor | None], model_params: list[torch.nn.Parameter]) -> None:
        # Custom gradient accumulation for parallellism.
        if len(buf) == 0:
            buf = [None] * len(model_params)

        # Accumulate into private grad buffer
        grads = torch.autograd.grad(self.batch_loss, model_params, allow_unused=True, retain_graph=False)
        for i, (g, b) in enumerate(zip(grads, buf)):
            if g is None:
                continue
            if b is None:
                buf[i] = g
            else:
                b.add_(g)


def train_on_images(dataloaders: list[DataLoader], params: TrainParams, viz: bool, log: bool, threaded: bool) -> tuple[
    ProcessBatch | None,
    tuple[list[LandmarkPrediction], list[torch.Tensor]],  # (pred_list, labels_list)
    LandmarkPrediction | None,  # final_preds
    tuple[torch.Tensor, ...] | None,  # final_idx
    LandmarkPrediction | None,  # initial_preds
    tuple[torch.Tensor, ...] | None,  # initial_idx
    utils.torch.datasets.Batch | None,  # batch
]:
    model = params.model
    viz_idx = params.rng.integers(0, len(dataloaders))

    if log:
        all_batches = ProcessBatch(params, is_viz=False)
    else:
        all_batches = None

    pred_list: list[LandmarkPrediction] = []
    labels_list: list[torch.Tensor] = []

    initial_preds: LandmarkPrediction | None = None
    initial_idx: tuple[torch.Tensor, ...] | None = None
    final_preds: LandmarkPrediction | None = None
    final_idx: tuple[torch.Tensor, ...] | None = None
    viz_batch: Batch | None = None

    model_params = [p for p in model.parameters() if p.requires_grad]
    stream_grads: dict[int, list[torch.Tensor | None]] = {}

    default_stream = params.get_default_stream()
    work_items: dict[int, tuple[torch.Stream | None, list[tuple[ProcessBatch, Callable[[], Batch]]]]] = {}

    for dl_idx, dl in enumerate(dataloaders):
        stream = params.get_stream(dl.stream_idx)
        _, items = work_items.setdefault(dl.stream_idx, (stream, []))

        # Synchronize all streams with the default stream.
        if stream is not None:
            assert default_stream is not None
            stream.wait_stream(default_stream)

        # Prepare batch
        with stream if stream is not None else nullcontext():
            process_batch = ProcessBatch(
                params, is_viz=(viz and (dl_idx == viz_idx)), rng=np.random.default_rng(int(params.rng.integers(2**62)))
            )

            raw_batch = next(dl)
            batch = partial(dl.wrap_batch, raw_batch, device=params.device, query_points=params.query_points.queries)
        items.append((process_batch, batch))

    results: list[ProcessBatch] = []

    def process_one(
        model: QLOT,
        model_params: list[torch.nn.Parameter],
        stream: torch.Stream | None,
        items: list[tuple[ProcessBatch, Callable[[], Batch]]],
        log: bool,
    ):
        pred_list: list[LandmarkPrediction] = []
        labels_list: list[torch.Tensor] = []
        results: list[ProcessBatch] = []
        buf: list[torch.Tensor | None] = [None] * len(model_params)

        with stream if stream is not None else nullcontext():
            for process_batch, f_batch in items:
                batch = f_batch()

                process_batch.process(batch, model)
                process_batch.backward(buf, model_params)

                # Detach to avoid computation graph retention across batches.
                process_batch.detach()

                assert process_batch.final_preds is not None
                assert process_batch.final_idx is not None

                if log:
                    pred_list.append(process_batch.final_preds.detach())
                    labels_list.append(batch[process_batch.final_idx].labels)

                    if stream is not None:
                        assert default_stream is not None
                        # We will use these tensors on the default stream later,
                        # so record them.
                        process_batch.final_preds.record_stream(default_stream)
                        process_batch.final_idx[0].record_stream(default_stream)
                        process_batch.final_idx[1].record_stream(default_stream)

                if process_batch.is_viz:
                    assert process_batch.initial_preds is not None
                    assert process_batch.initial_idx is not None
                    nonlocal initial_preds, initial_idx, final_preds, final_idx, viz_batch
                    initial_preds = process_batch.initial_preds
                    initial_idx = process_batch.initial_idx
                    final_preds = process_batch.final_preds
                    final_idx = process_batch.final_idx
                    viz_batch = batch

                    if stream is not None:
                        assert default_stream is not None
                        initial_preds.record_stream(default_stream)
                        initial_idx[0].record_stream(default_stream)
                        initial_idx[1].record_stream(default_stream)
                        final_preds.record_stream(default_stream)
                        final_idx[0].record_stream(default_stream)
                        final_idx[1].record_stream(default_stream)
                        viz_batch.record_stream(default_stream)

                if log and stream is not None:
                    assert default_stream is not None
                    # We will use these tensors on the default stream later,
                    # so record them.
                    for t in process_batch.iter_tensors():
                        t.record_stream(default_stream)
                if log:
                    results.append(process_batch)
        return results, pred_list, labels_list, buf

    futures: list[tuple[int, torch.Stream | None, Future]] = []
    for stream_idx, (stream, items) in work_items.items():
        if threaded:
            fut = params._thread_pool.submit(process_one, model, model_params, stream, items, log)
            futures.append((stream_idx, stream, fut))
        else:
            fut = Future()
            fut.set_result(process_one(model, model_params, stream, items, log))
            futures.append((stream_idx, stream, fut))

    for stream_idx, stream, fut in futures:
        r, p, l, b = fut.result()  # Wait for completion
        results.extend(r)
        pred_list.extend(p)
        labels_list.extend(l)
        stream_grads[stream_idx] = b

        if stream is not None:
            assert default_stream is not None
            default_stream.wait_stream(stream)

    # Accumulate gradients in model parameters.
    for buf in stream_grads.values():
        for p, g in zip(model_params, buf):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
            if default_stream is not None:
                g.record_stream(default_stream)
    model.flush_stream()

    for process_batch in results:
        assert all_batches is not None
        # Accumulate losses for logging. This needs to happen only after
        # we've synchronized with the default stream (all indivitual streams have finished).
        all_batches.acc_loss(
            l1_loss=process_batch.acc_l1_loss,
            huber_loss=process_batch.acc_huber_loss,
            gnll_loss=process_batch.acc_gnll_loss,
            delta_gnll_loss=process_batch.acc_delta_gnll_loss,
            acceleration_gnll_loss=process_batch.acc_acceleration_gnll_loss,
            hconsistency_loss=process_batch.acc_hconsistency_loss,
            batch_loss=process_batch.batch_loss,
        )

    return (
        all_batches,
        (pred_list, labels_list),
        final_preds,
        final_idx,
        initial_preds,
        initial_idx,
        viz_batch,
    )


LABELS_COLOR = (0, 255, 0)  # Green for ground truth
PREDICTIONS_COLOR = (255, 0, 0)  # Red for predictions
SHOW_MAP_COLOR = (0, 0, 255)  # Blue for selected landmark map
VALID_ITERS = 3  # Number of iterations to use during validation


@torch.no_grad()
def validate(
    dataloaders: DataLoadersBase,
    params: TrainParams,
    config: Config,
    tb_writer: SummaryWriter,
    global_step: int,
):
    model = params.model
    model.eval()

    images_args: list[tuple] = []  # List of (image, all_xy, all_cov, colors) tuples for visualization

    similarity_maps = []  # List of (n_heads, H, W) tensors
    nmes_tensor: dict[str, torch.Tensor] = {}  # Dictionary of NME per dataloader name
    nsamples: dict[str, int] = {}  # Dictionary of number of samples per dataloader name

    valid_image_loaders = dataloaders.valid_image_loaders()
    show_loader_i = params.rng.integers(0, len(valid_image_loaders))
    map_str: str | None = None

    default_stream = params.get_default_stream()
    streams: list[torch.Stream | None] = []

    dl_iters = []
    nme_list_list: list[list[torch.Tensor]] = []
    show_i_list = []
    names: list[str] = []

    for dl in valid_image_loaders:
        stream = params.get_stream(dl.stream_idx)
        streams.append(stream)
        # Synchronize all streams with the default stream.
        if stream is not None:
            assert default_stream is not None
            stream.wait_stream(default_stream)

        assert isinstance(dl.dataset, QueriedFaceDataset), f"Expected QueriedFaceDataset, got {type(dl.dataset)}"
        assert dl.dataset.is_clips == False, f"Validation set must not be a clips dataset"
        name = dl.dataset.dataset.short_name
        names.append(name)

        nsamples[name] = len(dl.dataset)
        show_i_list.append(params.rng.integers(0, len(dl)))
        dl_iters.append(enumerate(dl))
        nme_list_list.append([])

    done = [False] * len(dl_iters)
    while not all(done):
        for loader_i, (dl_iter, stream) in enumerate(zip(dl_iters, streams)):
            if done[loader_i]:
                continue
            name = names[loader_i]

            with stream if stream is not None else nullcontext():
                show_i = show_i_list[loader_i]
                nme_list = nme_list_list[loader_i]

                batch = next(dl_iter, None)
                if batch is None:
                    done[loader_i] = True
                    nmes_tensor[name] = torch.stack(nme_list).mean()
                    nme_list_list[loader_i] = []
                    continue
                i, batch = batch

                test_batch = QueriedFaceDataset.wrap(batch, device=params.device, query_points=params.query_points.queries)
                is_show_maps = loader_i == show_loader_i and i == show_i

                # Forward pass for the batch
                res: tuple[LandmarkPrediction, list[torch.Tensor]] | LandmarkPrediction = model(
                    test_batch.images, test_batch.queries, iterations=VALID_ITERS, return_similarity_maps=is_show_maps
                )  # (batch_size, num_queries, 3)
                preds = res[0] if isinstance(res, tuple) else res
                res_similarity_maps = res[1] if isinstance(res, tuple) else None

                errors = torch.sqrt(
                    torch.sum((preds.mean - test_batch.labels[..., :2]) ** 2, dim=-1)
                )  # (batch_size, num_queries)
                norm_factors = (
                    torch.max(test_batch.labels, dim=1).values - torch.min(test_batch.labels, dim=1).values
                )  # (batch_size, 2)
                norm_factors = norm_factors.prod(dim=-1).sqrt()  # (batch_size,)
                nme_validation = (errors.mean(dim=1) / norm_factors).mean().cpu()
                nme_list.append(nme_validation)

                # Store an annotated image for one randomly selected sample of each dataloader.
                # Store the similarity maps of one randomly selected sample across all dataloaders.
                if i == show_i:
                    sample_i = params.rng.integers(0, test_batch.images.shape[0])
                    labels = test_batch.labels[sample_i]
                    preds_xy = preds[sample_i].mean.detach()
                    preds_cov = preds[sample_i].cov.as_cov2d_params()
                    labels_cov = torch.zeros(
                        *labels.shape[:-1], preds_cov.shape[-1], dtype=preds_cov.dtype, device=preds_cov.device
                    )  # (1, num_queries, 3)

                    if is_show_maps:
                        # Show the landmark of the visualized similarity map in blue.
                        map_lmk_idx = params.rng.integers(0, preds_xy.shape[-2])
                        preds_xy_map = preds_xy[map_lmk_idx : map_lmk_idx + 1]
                        preds_cov_map = preds_cov[map_lmk_idx : map_lmk_idx + 1]
                        preds_xy_rest = torch.cat([preds_xy[:map_lmk_idx], preds_xy[map_lmk_idx + 1 :]], dim=0)
                        preds_cov_rest = torch.cat([preds_cov[:map_lmk_idx], preds_cov[map_lmk_idx + 1 :]], dim=0)

                        all_xy = [labels, preds_xy_rest, preds_xy_map]
                        all_cov = [labels_cov, preds_cov_rest, preds_cov_map]
                        colors = [LABELS_COLOR, PREDICTIONS_COLOR, SHOW_MAP_COLOR]

                        assert res_similarity_maps is not None
                        similarity_maps = [m[sample_i, map_lmk_idx] for m in res_similarity_maps]
                        map_str = f"{name}_{i}-{sample_i}_{map_lmk_idx} "
                    else:
                        all_xy = [labels, preds_xy]
                        all_cov = [labels_cov, preds_cov]
                        colors = [LABELS_COLOR, PREDICTIONS_COLOR]
                    images_args.append((test_batch.images[sample_i], all_xy, all_cov, colors))

    for s in streams:
        if s is not None:
            assert default_stream is not None
            default_stream.wait_stream(s)

    # Weigh individual NMEs by the number of samples.
    nmes: dict[str, float] = {name: float(nme.item()) for name, nme in nmes_tensor.items()}

    weighted_nme = sum(nmes[name] * nsamples[name] for name in nmes) / sum(nsamples.values())
    unweighted_nme = sum(nme for nme in nmes.values()) / len(nmes)

    tb_writer.add_scalar("NME/valid", unweighted_nme, global_step)
    tb_writer.add_scalar("NME/valid_weighted", weighted_nme, global_step)

    if config.best_nme is None or unweighted_nme < config.best_nme:
        config.best_nme = unweighted_nme
    config.nme = unweighted_nme
    config.nme_weighted = weighted_nme

    for name, nme in nmes.items():
        tb_writer.add_scalar(f"NME/valid_{name}", nme, global_step)
        config.others[f"nme_{name}"] = nme

    # Show annotated images.
    images = []  # List of (C, H, W) tensors
    for args in images_args:
        img, all_xy, all_cov, colors = args
        img = img.cpu()
        all_xy = [v.cpu() for v in all_xy]
        all_cov = [v.cpu() for v in all_cov]
        img = draw_keypoints(img, all_xy, variances=all_cov, colors=colors, radius=1, width=1)
        images.append(img)
    tb_writer.add_image("Images/valid", torchvision.utils.make_grid(images, nrow=2), global_step)

    assert len(similarity_maps) != 0
    assert map_str is not None
    res_maps = []

    # Show similarity maps for a single sample.
    MAP_SIZE = 224
    for level, m in enumerate(similarity_maps):
        m, (min_vals, max_vals) = map_to_color(m, norm_per_batch=False, cmap="inferno")
        m = TF.resize(m, [MAP_SIZE, MAP_SIZE], interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
        m = TF.pad(m, [0, 20, 0, 0], fill=0)  # Pad top for text overlay

        # Add text overlay
        for i in range(m.shape[0]):
            img = m[i].cpu().permute(1, 2, 0).contiguous().numpy()  # (H, W, C)
            img = cv2.putText(
                img,
                f"{map_str}L{level}H{i} r:{min_vals[i].item():.2f}/{max_vals[i].item():.2f}",
                (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            map_str = ""  # Only show the map_str on the first image
            res_maps.append(torch.from_numpy(img).permute(2, 0, 1))  # (C, H, W)
    tb_writer.add_image("Images/sim_maps", torchvision.utils.make_grid(res_maps, nrow=2), global_step)


@torch.no_grad()
def validate_jitter(
    dataloaders: list[DataLoader],
    params: TrainParams,
    config: Config,
    tb_writer: SummaryWriter,
    global_step: int,
):
    model = params.model
    model.eval()

    nmf_list: list[float] = []
    nad_aoc_list: list[float] = []
    nad_vals_list: list[torch.Tensor] = []

    landmarks_list: list[torch.Tensor] = []
    landmarks_gt_list: list[torch.Tensor] = []
    streams: list[torch.Stream | None] = []
    clip_len_list: list[int] = []
    num_clips_list: list[int] = []
    nlandmarks_list: list[int] = []

    default_stream = params.get_default_stream()
    streams: list[torch.Stream | None] = []
    for dataloader in dataloaders:
        stream = params.get_stream(dataloader.stream_idx)
        streams.append(stream)
        # Synchronize all streams with the default stream.
        if stream is not None:
            assert default_stream is not None
            stream.wait_stream(default_stream)

        assert isinstance(dataloader.dataset, QueriedFaceDataset), f"Expected QueriedFaceDataset, got {type(dataloader.dataset)}"
        dataset = dataloader.dataset
        assert dataset.is_clips, "Dataset must be a QueriedFaceDataset with is_clips=True for jitter validation."
        assert isinstance(
            dataset.dataset, utils.datasets.AsClips
        ), f"Expected dataset.dataset to be AsClips, got {type(dataset.dataset)}"

        if isinstance(dataloader.sampler, StridedClipSampler):
            num_clips = len(dataloader.sampler)
        else:
            num_clips = len(dataloader.dataset)
        num_clips_list.append(num_clips)
        clip_len = dataset.dataset.clip_len
        clip_len_list.append(clip_len)
        nlandmarks = dataset.dataset.nlandmarks
        nlandmarks_list.append(nlandmarks)

        with stream if stream is not None else nullcontext():
            landmarks = torch.zeros((num_clips, clip_len, nlandmarks, 2), dtype=torch.float32, device=params.device)
            landmarks_gt = torch.zeros((num_clips, clip_len, nlandmarks, 2), dtype=torch.float32, device=params.device)
            landmarks_list.append(landmarks)
            landmarks_gt_list.append(landmarks_gt)

    last_clip_idx_list = [0] * len(dataloaders)

    done = np.zeros(len(dataloaders), dtype=bool)
    dl_iters = [iter(dl) for dl in dataloaders]

    while not done.all():
        for dl_idx, (dataloader, stream) in enumerate(zip(dl_iters, streams)):
            if done[dl_idx]:
                continue
            with stream if stream is not None else nullcontext():
                batch = next(dataloader, None)
                if batch is None:
                    done[dl_idx] = True
                    continue

                batch = QueriedFaceDataset.wrap(batch, device=params.device, query_points=params.query_points.queries)
                assert (
                    batch.orig_clip_len == clip_len_list[dl_idx]
                ), f"Expected batch.orig_clip_len={clip_len_list[dl_idx]}, got {batch.orig_clip_len}"
                curr_clips_idx = slice(last_clip_idx_list[dl_idx], last_clip_idx_list[dl_idx] + batch.orig_batch_size)
                last_clip_idx_list[dl_idx] += batch.orig_batch_size

                landmarks = landmarks_list[dl_idx]
                landmarks_gt = landmarks_gt_list[dl_idx]

                hidden_state = None
                preds: LandmarkPrediction | None = None
                for frame_i in range(clip_len_list[dl_idx]):
                    frame_batch = batch[:, frame_i]

                    preds, hidden_state = model(
                        frame_batch.images,
                        frame_batch.queries,
                        # Make sure the predictions for the first frame are accurate.
                        # This is needed so that we don't measure how accurate the model is starting
                        # from scratch with one iteration, i.e. we want to measure the steady-state
                        # jitter, not the startup error.
                        iterations=VALID_ITERS if frame_i == 0 else 1,
                        return_sequence=False,
                        store_similarity_maps=False,
                        return_hidden_state=True,
                        prefill_hidden_state=hidden_state,
                        prefill_starting_landmarks=preds,
                    )
                    assert preds is not None
                    landmarks[curr_clips_idx, frame_i, :, :] = preds.mean
                    landmarks_gt[curr_clips_idx, frame_i, :, :] = frame_batch.labels

    # Wait for all streams to finish before computing metrics
    for s in streams:
        if s is not None:
            assert default_stream is not None
            default_stream.wait_stream(s)

    for landmarks, landmarks_gt in zip(landmarks_list, landmarks_gt_list):
        if default_stream is not None:
            landmarks.record_stream(default_stream)
            landmarks_gt.record_stream(default_stream)
        nmf = utils.torch.misc.calc_nmf(landmarks, landmarks_gt)
        nad_vals = utils.torch.misc.navar_pos_all(landmarks, landmarks_gt).mean(dim=(1, 2))
        nad_aoc = nad_vals.sum().item()

        nmf_list.append(nmf)
        nad_aoc_list.append(nad_aoc)
        nad_vals_list.append(nad_vals)

    nmf = float(np.mean(nmf_list))
    nad_aoc = float(np.mean(nad_aoc_list))
    nad_vals = torch.stack(nad_vals_list).mean(dim=0).cpu().tolist()

    tb_writer.add_scalar("NMF/valid_jitter", nmf, global_step)
    tb_writer.add_scalar("NMF/nad_aoc", nad_aoc, global_step)
    config.nmf = nmf
    config.nad_aoc = nad_aoc
    config.nad_vals = nad_vals


def compile_model(model: QLOT):
    print("Compiling model...")
    # model.feature_extractor.compile()
    # model.image_feature_correlator.proj_to_kernel.compile(dynamic=True)
    # model.query_encoder.compile(dynamic=True)
    # model.image_feature_correlator.compile(dynamic=True)
    model.encoder.compile(dynamic=True)
    model.update_predictor.compile(dynamic=True)


def train(cfg: Config, params: TrainParams, dataloaders: DataLoadersBase):
    scheduler = params.scheduler
    optimizer = params.optimizer
    optimizer_to(optimizer, device)
    extra_save_args = params.extra_save_args if params.extra_save_args is not None else {}
    extra_save_args["query_points"] = lambda: params.query_points.state_dict()
    extra_save_args["rng_state"] = params.get_rng_state

    face_mesh_projector = utils.mesh.MeshProjector(params.face_mesh) if params.face_mesh is not None else None
    is_first_time = True
    params.query_points.requires_grad_(False)

    print("Starting training...")
    tb_writer = SummaryWriter(cfg.runs_dir())

    model = params.model
    model.train()

    is_backbone_frozen = params.global_step < params.freeze_backbone_steps
    model.freeze_backbone(is_backbone_frozen, all=params.init_backbone)
    compile_model(model)

    # Profiler schedule: wait 1 step, profile 3 steps, then repeat once.
    # profiler = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=2, warmup=10, active=1, repeat=0),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(str(cfg.runs_dir() / "profiler")),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_modules=True,
    #     with_stack=True,
    # )
    # profiler.start()

    is_validate = False
    for epoch in range(params.epochs):
        model.train()
        for i in tqdm(range(params.steps_per_epoch), desc=f"Epoch {epoch+1}/{params.epochs}", unit="step"):
            # Unfreeze backbone if needed
            if is_backbone_frozen and params.global_step >= params.freeze_backbone_steps:
                print(f"Unfreezing backbone at global step {params.global_step}.")
                is_backbone_frozen = False
                model.freeze_backbone(False, all=True)
                compile_model(model)

            is_viz = params.global_step % params.log_images_every == 0
            is_log = params.global_step % params.log_every == 0

            optimizer.zero_grad(set_to_none=True)

            (
                all_batches,
                (pred_list, labels_list),
                final_preds,
                final_idx,
                initial_preds,
                initial_idx,
                batch,
            ) = train_on_images(dataloaders.train_loaders(), params, is_viz, is_log, threaded=not is_first_time)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # profiler.step()

            # --- LOGGING ---

            if is_log:
                assert all_batches is not None
                assert len(pred_list) > 0
                assert len(labels_list) > 0

                # Log losses
                tb_writer.add_scalar("Loss/train", all_batches.batch_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/l1", all_batches.acc_l1_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/gnll", all_batches.acc_gnll_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/delta_gnll", all_batches.acc_delta_gnll_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/acceleration_gnll", all_batches.acc_acceleration_gnll_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/map_consistency", all_batches.acc_hconsistency_loss.item(), params.global_step)
                tb_writer.add_scalar("Loss/huber", all_batches.acc_huber_loss.item(), params.global_step)

                nme_list = []
                cov_max, cov_min, cov_mean, cov_std = [], [], [], []
                with torch.no_grad():
                    # Calculate normalized mean error (NME) over the last batch
                    for pred_pixel, pred_labels in zip(pred_list, labels_list):

                        # Compute per-point Euclidean error
                        errors = torch.sqrt(torch.sum((pred_pixel.mean - pred_labels) ** 2, dim=-1))  # (batch_size, num_queries)

                        # Normalize by sqrt(area of bounding box) to get NME
                        norm_factor = (
                            torch.max(pred_labels, dim=1).values - torch.min(pred_labels, dim=1).values
                        )  # (batch_size, 2)
                        norm_factor = norm_factor.prod(dim=-1).sqrt()  # (batch_size,)
                        nme_v = (errors.mean(dim=-1) / norm_factor).mean()
                        nme_list.append(nme_v)

                        cov_max.append(pred_pixel.cov.max_variance.max())
                        cov_min.append(pred_pixel.cov.min_variance.min())
                        cov_mean.append(pred_pixel.cov.max_variance.mean())
                        cov_std.append(pred_pixel.cov.max_variance.std())

                    tb_writer.add_scalar("NME/train", torch.stack(nme_list).mean(), params.global_step)

                    # Log variance statistics
                    tb_writer.add_scalar("Variance/mean", torch.stack(cov_mean).mean().item(), params.global_step)
                    tb_writer.add_scalar("Variance/std", torch.stack(cov_std).mean().item(), params.global_step)
                    tb_writer.add_scalar("Variance/min", torch.stack(cov_min).mean().item(), params.global_step)
                    tb_writer.add_scalar("Variance/max", torch.stack(cov_max).mean().item(), params.global_step)

                # Log learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar("Learning_Rate", current_lr, params.global_step)

            if is_viz:
                assert batch is not None
                assert final_preds is not None
                assert final_idx is not None
                assert initial_preds is not None
                assert initial_idx is not None

                try:
                    MAX_NUM_IMG = 4
                    # Log images
                    images_pred = [
                        torchvision.utils.draw_keypoints(
                            torchvision.utils.draw_keypoints(
                                batch.images[final_idx][i].cpu().clip(0.0, 1.0),
                                batch.labels[final_idx][i, ..., :2].unsqueeze(0).cpu(),
                                colors=LABELS_COLOR,
                                radius=2,
                            ),
                            final_preds[i].mean.unsqueeze(0).cpu(),
                            colors=PREDICTIONS_COLOR,
                            radius=2,
                        )
                        for i in range(min(MAX_NUM_IMG, len(final_idx[0])))
                    ]
                    images_initial_est = [
                        torchvision.utils.draw_keypoints(
                            torchvision.utils.draw_keypoints(
                                batch.images[initial_idx][i].cpu().clip(0.0, 1.0),
                                batch.labels[initial_idx][i, ..., :2].unsqueeze(0).cpu(),
                                colors=LABELS_COLOR,
                                radius=2,
                            ),
                            initial_preds[i].mean.unsqueeze(0).cpu(),
                            colors=PREDICTIONS_COLOR,
                            radius=2,
                        )
                        for i in range(min(MAX_NUM_IMG, len(initial_idx[0])))
                    ]
                    tb_writer.add_image("Images/train", torchvision.utils.make_grid(images_pred, nrow=2), params.global_step)
                    tb_writer.add_image(
                        "Images/train_initial_preds",
                        torchvision.utils.make_grid(images_initial_est, nrow=2),
                        params.global_step,
                    )
                    del images_pred, images_initial_est
                except Exception as e:
                    print(e)

            # --- VALIDATION ---
            save_model = False
            best_checkpoints: dict[int, tuple[float, float]] = cfg.others.setdefault("_best_checkpoints", {})

            if not is_first_time and (params.global_step % params.small_validate_every) == 0:
                validate(dataloaders, params, cfg, tb_writer, params.global_step)

                nme: float = cfg.nme
                best_nme: float = cfg.best_nme

                if nme <= params.nme_coeff * best_nme:
                    save_model = True
                is_validate = True

            cfg.nmf = None
            cfg.nad_aoc = None
            cfg.nad_vals = None
            if (not is_first_time and (params.global_step % params.big_validate_every) == 0) or save_model:
                # Validate jitter robustness on WFLW validation set
                validate_jitter(
                    dataloaders=dataloaders.valid_video_loaders(),
                    params=params,
                    config=cfg,
                    tb_writer=tb_writer,
                    global_step=params.global_step,
                )
                is_validate = True
                assert cfg.nmf is not None

                nme: float = cfg.nme
                nmf: float = cfg.nmf
                best_checkpoints[params.global_step] = (nme, nmf)

            if save_model:
                NUM_KEEP = 20
                if len(best_checkpoints) > NUM_KEEP:
                    l = sorted(best_checkpoints.items(), key=lambda x: (x[1][0], x[1][1], x[0]))  # Sort by NME, then NMF
                    step_to_delete = l[-1][0]
                    if step_to_delete == params.global_step:
                        save_model = False
                    else:
                        del best_checkpoints[step_to_delete]
                        ckpt_path = cfg.ckpts_dir() / f"{step_to_delete}.pth"
                        if ckpt_path.exists():
                            ckpt_path.unlink()
                        print(f"Deleted checkpoint at step {step_to_delete} to keep only the best {NUM_KEEP} checkpoints.")

                if save_model:
                    save(
                        cfg.ckpts_dir() / f"{params.global_step}.pth",
                        model,
                        optimizer,
                        params.global_step,
                        cfg,
                        scheduler,
                        **extra_save_args,
                    )

            if is_validate:
                is_validate = False
                model.train()

            if params.global_step % params.save_latest_every == 0:
                save(cfg.ckpts_dir() / "latest.pth", model, optimizer, params.global_step, cfg, scheduler, **extra_save_args)

            params.global_step += 1
            is_first_time = False

        # save(
        #     cfg.ckpts_dir() / f"{params.global_step}.pth", model, optimizer, params.global_step, cfg, scheduler, **extra_save_args
        # )
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Continuous Landmark Detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--feature-extractor-from-scratch", action="store_true", help="Whether to train the feature extractor from scratch"
    )
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device index to use (default: auto-select)")
    parser.add_argument("--cont", action="store_true", help="Continue training from the given checkpoint")
    parser.add_argument("--epochs", type=int, default=110, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of steps per epoch")
    parser.add_argument("--model-iterations", type=int, default=3, help="Number of model iterations per forward pass")
    parser.add_argument("--small-validate-every", type=int, default=20, help="Frequency of small validation runs (in steps)")
    parser.add_argument("--log-images-every", type=int, default=20, help="Frequency of logging images to TensorBoard (in steps)")
    parser.add_argument("--save-latest-every", type=int, default=200, help="Frequency of saving latest checkpoint (in steps)")
    parser.add_argument("--name", type=str, default="infinite-lmk-v4", help="Name of model experiment")
    parser.add_argument("--run", type=str, default="r0", help="Run identifier for the experiment")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for optimizer")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to datasets directory")
    parser.add_argument("--queries-file", type=str, default=str(data.queries_file), help="Path to queries file")
    parser.add_argument(
        "--freeze-queries", type=str, nargs="+", default=[], help="Names of datasets for which to freeze query points"
    )
    parser.add_argument("--face-mesh", type=str, default="", help="Path to face mesh file for query point projection")
    parser.add_argument(
        "--cycle-model-steps",
        type=int,
        default=None,
        help="Number of steps to optimize the model before switching to query optimization",
    )
    parser.add_argument(
        "--cycle-query-steps",
        type=int,
        default=None,
        help="Number of steps to optimize the query points before switching to model optimization",
    )
    parser.add_argument("--queries-lr", type=float, default=1e-3, help="Learning rate for query points optimizer")
    parser.add_argument("--detect-anomalies", action="store_true", help="Enable PyTorch anomaly detection")
    parser.add_argument("--init-backbone", action="store_true", help="Initialize backbone with pretrained weights")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory to use for caching datasets and models")
    parser.add_argument(
        "--freeze-backbone-steps", type=int, default=None, help="Number of steps to freeze the backbone during training"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility; or None to use a random seed")
    parser.add_argument("--disable-streams", action="store_true", help="Disable CUDA streams")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 20
    torch._dynamo.config.accumulated_recompile_limit = 64
    # torch.backends.cudnn.benchmark = True
    # torch._inductor.config.aggressive_fusion = True
    # torch._inductor.config.trace.enabled = True

    if args.cache_dir is not None:
        utils.set_cache_dir(str(args.cache_dir))

    if args.detect_anomalies:
        torch.autograd.set_detect_anomaly(True)
        print("Enabled PyTorch anomaly detection.")

    default_lr = 2.0e-4
    lr = args.learning_rate if args.learning_rate is not None else default_lr

    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Seed before any stochastic construction ---
    # When --seed is provided, seed PyTorch (CPU/CUDA) and create the TrainParams NumPy generator
    # before constructing the model, datasets, augmentations, query points, optimizer, and loaders.
    # This makes fresh runs reproducible. Global Python/NumPy RNG is intentionally not seeded.
    train_rng: np.random.Generator
    if args.seed is not None:
        print(f"Setting random seed to {args.seed} for reproducibility.")
        torch.manual_seed(args.seed)
        train_rng = np.random.default_rng(args.seed)
        data.set_rng_seed(train_rng)
    else:
        train_rng = np.random.default_rng()

    model = QLOT(feature_extractor_pretrained=not args.feature_extractor_from_scratch)
    model = model.to(device)
    datasets = data.Datasets(Path(args.dataset_dir), rng=train_rng)

    # Initialize query points for each dataset
    query_points = data.make_query_points().to(device)

    opt_params = utils.torch.optim.opt_param_cfg(
        model.named_parameters(),
        config={
            "query_encoder.*": ParamCfg(lr=lr, weight_decay=0.0),
            "encoder.corr_feat_head_convs.?.0.log_temp": ParamCfg(weight_decay=0.0),
            "encoder.image_feature_proj.*": ParamCfg(weight_decay=0.0),
            "update_predictor.mixer.write_temperature": ParamCfg(weight_decay=0.0),
            "update_predictor.mixer.residual_weights": ParamCfg(lr=2.0 * lr, weight_decay=0.0),
        },
        other_params=[
            # (query_points.parameters(), ParamCfg(lr=args.queries_lr, weight_decay=0.0))
        ],
    )

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=lr,
        weight_decay=0.005,
    )

    # if args.init_backbone:
    #     def piecewise_cosine_anneal(step):
    #         phase1_steps = 10000
    #         phase2_steps = 170000

    #         if step < phase1_steps:
    #             # Phase 1: Anneal from `lr` to 8e-5
    #             start_factor = 1.0
    #             end_factor = 1e-4 / lr
    #             progress = step / phase1_steps
    #         else:
    #             # Phase 2: Anneal from 8e-5 down to 1e-6
    #             start_factor = 1e-4 / lr
    #             end_factor = 5e-6 / lr
    #             progress = min(1.0, (step - phase1_steps) / phase2_steps)

    #         # Standard Cosine Annealing formula
    #         return end_factor + 0.5 * (start_factor - end_factor) * (1 + math.cos(math.pi * progress))

    #     # Initialize the scheduler
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=piecewise_cosine_anneal)
    # else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=110000, eta_min=1e-5)

    params = TrainParams(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        model_iterations=args.model_iterations,
        small_validate_every=args.small_validate_every,
        log_images_every=args.log_images_every,
        save_latest_every=args.save_latest_every,
        query_points=query_points,
        face_mesh=Path(args.face_mesh) if args.face_mesh else None,
        freeze_queries=args.freeze_queries,
        device=device,
        init_backbone=args.init_backbone,
        rng=train_rng,
        stream_parallellism=not args.disable_streams,
    )
    if args.freeze_backbone_steps is not None:
        params.freeze_backbone_steps = args.freeze_backbone_steps

    if args.cycle_model_steps is not None:
        params.model_steps = args.cycle_model_steps
    if args.cycle_query_steps is not None:
        params.query_steps = args.cycle_query_steps

    def weights_filter(state_dict: dict) -> dict:
        state_dict = QLOT.translate_weights(state_dict)

        if args.init_backbone:
            state_dict = QLOT.filter_weights_backbone(state_dict)
        return state_dict

    def load_queries(state_dict: dict) -> None:
        try:
            params.query_points.load_state_dict(state_dict)  # type: ignore
            print("Loaded query points from checkpoint.")
        except Exception as e:
            print(f"Failed to load query points from checkpoint: {e}")

    def load_rng_state(state: dict) -> None:
        params.set_rng_state(state)

    cfg = Config(
        name=args.name,
        learning_rate=args.learning_rate if args.learning_rate is not None else default_lr,
        run=args.run,
    )
    cfg.best_nme = None
    load_optimizer = not args.init_backbone
    load_scheduler = not args.init_backbone

    if args.checkpoint is None and args.cont:
        checkpoint = cfg.ckpts_dir() / "latest.pth"
        args.checkpoint = str(checkpoint) if checkpoint.exists() else None
    if args.checkpoint is not None:
        _global_step, _cfg = load(
            args.checkpoint,
            model,
            optimizer if load_optimizer else None,
            scheduler if load_scheduler else None,
            model_func=weights_filter,
            query_points=load_queries,
            rng_state=load_rng_state,
            strict=not params.init_backbone,  # Allow missing keys when initializing backbone with pretrained weights
        )
        if args.cont:
            params.global_step = _global_step
            cfg = _cfg
        print(f"Resumed training from checkpoint {args.checkpoint} at global step {_global_step}")

    if args.queries_file:
        params.query_points.load_state_dict(torch.load(args.queries_file))
        print(f'Loaded query points from {args.queries_file}. Run with `--queries-file=""` to load from checkpoint.')

    dataloaders = DataLoaders(datasets, cfg, rng=params.rng)
    train(cfg, params, dataloaders)
