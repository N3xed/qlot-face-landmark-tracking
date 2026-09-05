import os
from dataclasses import dataclass
from pathlib import Path
import torch
import json
import io
from utils.torch.transforms import RandomHorizontalFlip, Videoify
import utils
from utils.datasets import CanonicalLandmarks
from utils.datasets.image import DatasetName
from utils.torch.datasets import QueriedFaceDataset, first_element
from model.utils import QueryPoints
import pyvista
import albumentations as A
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

data_dir = Path(os.path.dirname(__file__) + "/../data")
assert data_dir.exists(), f"Data directory {data_dir} does not exist."

########## Load queries and dense points ##########

face_mesh_file = data_dir / "face_mesh.obj"
queries_file = data_dir / "query_points.pth"

queries_70_synth = CanonicalLandmarks.load(data_dir / "canonical_landmarks_70_synth.json")
queries_68_ibug = CanonicalLandmarks.load(data_dir / "canonical_landmarks_68_ibug.json")
queries_98_wflw = CanonicalLandmarks.load(data_dir / "canonical_landmarks_98_wflw.json")


def make_query_points() -> QueryPoints:
    return QueryPoints(
        init={
            DatasetName.WFLW: torch.from_numpy(queries_98_wflw.points).float(),
            DatasetName.WFLW_V: torch.from_numpy(queries_98_wflw.points).float(),
            DatasetName.FaceSynth: torch.from_numpy(queries_70_synth.points).float(),
            DatasetName.Ibug: torch.from_numpy(queries_68_ibug.points).float(),
        },
        canonical_landmarks={
            DatasetName.WFLW: queries_98_wflw,
            DatasetName.WFLW_V: queries_98_wflw,
            DatasetName.FaceSynth: queries_70_synth,
            DatasetName.Ibug: queries_68_ibug,
        },
    )


queries_opt = make_query_points()
queries_opt.load_state_dict(torch.load(data_dir / "query_points.pth"))

face_mesh: pyvista.PolyData = pyvista.read(data_dir / "face_mesh.obj")  # type: ignore

with open(data_dir / "dense_points.json", "r") as f:
    # TODO: update from `face_mesh`, these are currently outdated
    dense_points = json.load(f)
    dense_points_faces = torch.tensor(dense_points["faces"], dtype=torch.long)
    dense_points = torch.tensor(dense_points["points"], dtype=torch.float32)

########### Define augmentation pipeline ###########

train_im_size = 224
test_im_size = 224
test_padding_ratio = 0.10
synth_clip_len = 4
video_clip_len = 16


def _make_videoify() -> Videoify:
    """Create a fresh Videoify instance with the project's standard parameters."""
    return Videoify(
        clip_len=synth_clip_len,
        translate=(-0.06, 0.06),
        scale=(0.95, 1.05),
        degrees=(-10.0, 10.0),
        shear=(-5.0, 5.0),
        corr_angle=2,
        corr_translate=2,
        corr_scale=2,
        corr_shear=2,
        # Computed from the average motion of the synthetic clips vs. real clips in WFLW_V,
        # ~9px synthetic motion vs. ~3px real motion (variable step size)
        # (see Datasets.estimate_synth_fps()).
        effective_fps=380.0,
    )


def _make_video_consistent() -> A.Compose:
    """
    Create a fresh video-consistent (same augmentations for all frames in a clip)
    color/shadow augmentation pipeline.
    """
    return A.Compose(
        [
            # Existing global color/lighting adjustments
            A.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.2, hue=0.2, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.8),
            A.RandomGamma(gamma_limit=(75, 125), p=0.8),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),  # Handle uneven lighting
            # Shadows
            A.OneOf(
                [
                    A.PlasmaShadow(roughness=1.2, shadow_intensity_range=(0.2, 0.8), p=1.0),
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_dimension=5, p=1.0),
                ],
                p=0.4,
            ),
            # Extreme Highlights / Flares (simulating specular glare)
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=100, p=0.25),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _make_noise() -> A.Compose:
    """Create a fresh frame-specific noise/blur augmentation pipeline."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussNoise(p=1.0, std_range=(0.01, 0.15)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.5), p=1.0),
                    A.ShotNoise(p=1.0, scale_range=(0.01, 0.08)),
                ],
                p=0.98,
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=0.5),
                    A.GaussianBlur(5, (0.1, 2.0), p=1.0),
                ],
                p=1.0,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


augment_wflw = A.ReplayCompose(
    [
        A.Affine(
            scale=(1.1, 1.8),
            translate_percent=(-0.1, 0.1),
            rotate=(-15.0, 15.0),
            border_mode=cv2.BORDER_REPLICATE,
            keep_ratio=True,
            p=1.0,
        ),
        A.Resize(train_im_size, train_im_size),
        _make_video_consistent(),
        _make_videoify(),
        # include_flipped=True, so must come after videoify since
        # we want the flipped version to also have the videoify transformations flipped
        # (e.g. face moving up-right in the original should move up-left in the hflipped version).
        RandomHorizontalFlip(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    is_check_shapes=False,
)


augment_test = A.ReplayCompose(
    [
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.15, 0.15),
            rotate=(-25.0, 25.0),
            border_mode=cv2.BORDER_REPLICATE,
            p=1.0,
        ),
        A.Resize(train_im_size, train_im_size),
        _make_video_consistent(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)


augment_wflw_v = A.ReplayCompose(
    [
        A.Affine(
            scale=(1.15, 1.5),
            translate_percent=(-0.15, 0.15),
            rotate=(-15.0, 15.0),
            border_mode=cv2.BORDER_REPLICATE,
            p=1.0,
        ),
        A.Resize(train_im_size, train_im_size),
        RandomHorizontalFlip(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)


augment_face_synth = A.ReplayCompose(
    [
        A.Affine(
            scale=(1.2, 2.5),
            translate_percent=(-0.1, 0.1),
            rotate=(-25.0, 25.0),
            border_mode=cv2.BORDER_REPLICATE,
            keep_ratio=True,
            p=1.0,
        ),
        A.Resize(train_im_size, train_im_size),
        # include_flipped=False, so put before videoify for efficiency.
        RandomHorizontalFlip(),
        _make_video_consistent(),
        _make_videoify(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    is_check_shapes=False,
)


augment_ibug = A.ReplayCompose(
    [
        A.Affine(
            scale=(1.2, 2.0),
            translate_percent=(-0.1, 0.1),
            rotate=(-15.0, 15.0),
            border_mode=cv2.BORDER_REPLICATE,
            keep_ratio=True,
            p=1.0,
        ),
        A.Resize(train_im_size, train_im_size),
        _make_video_consistent(),
        _make_videoify(),
        # include_flipped=True, so must come after videoify since
        # we want the flipped version to also have the videoify transformations flipped
        # (e.g. face moving up-right in the original should move up-left in the hflipped version).
        RandomHorizontalFlip(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    is_check_shapes=False,
)

noise_wflw = _make_noise()
noise_face_synth = _make_noise()
noise_ibug = _make_noise()
noise_wflw_v = _make_noise()

_aug: dict[str, A.BaseCompose] = {
    "wflw": augment_wflw,
    "test": augment_test,
    "wflw_v": augment_wflw_v,
    "face_synth": augment_face_synth,
    "ibug": augment_ibug,
    "noise_wflw": noise_wflw,
    "noise_face_synth": noise_face_synth,
    "noise_ibug": noise_ibug,
    "noise_wflw_v": noise_wflw_v,
}


def get_aug_state() -> dict:
    """
    Get the current Albumentations state for all pipelines.
    """
    state: dict[str, dict] = {}
    for key, root in _aug.items():
        buffer = io.StringIO()
        A.save(root, buffer)
        state[key] = json.loads(buffer.getvalue())
    return state


def set_aug_state(state: dict) -> None:
    """
    Restore the Albumentations state for all pipelines.
    """
    if not isinstance(state, dict):
        logger.warning(f"Albumentations state is not a dict")
        return
    for key, state in state.items():
        try:
            state_str = json.dumps(state)
        except Exception as e:
            logger.warning(f"Albumentations state for `{key}` is not JSON serializable: {e}")
            continue
        if key not in _aug:
            logger.warning(f"Albumentations state for `{key}` is not a known pipeline")
            continue
        buffer = io.StringIO(state_str)

        val = A.load(buffer)
        if not isinstance(val, _aug[key].__class__):
            logger.warning(f"Albumentations state for `{key}` is not the correct type")
            continue
        _aug[key] = val


def set_rng_seed(rng: np.random.Generator) -> None:
    """
    Seed all Albumentations pipelines with a deterministic, `rng` derived seed.
    """
    for root in _aug.values():
        seed = int(rng.integers(0, 2**31))
        root.set_random_seed(seed)


@dataclass
class Datasets:
    # Training sets
    wflw_v: QueriedFaceDataset
    face_synth: QueriedFaceDataset
    ibug: QueriedFaceDataset
    wflw: QueriedFaceDataset

    # Validation sets
    wflw_v_val: QueriedFaceDataset
    face_synth_val: QueriedFaceDataset
    ibug_val: QueriedFaceDataset
    wflw_val: QueriedFaceDataset

    # Test sets
    wflw_v_test: QueriedFaceDataset

    face_synth_test: QueriedFaceDataset

    wflw_test_full: QueriedFaceDataset
    wflw_test_blur: QueriedFaceDataset
    wflw_test_expression: QueriedFaceDataset
    wflw_test_illumination: QueriedFaceDataset
    wflw_test_largepose: QueriedFaceDataset
    wflw_test_makeup: QueriedFaceDataset
    wflw_test_occlusion: QueriedFaceDataset

    ibug_test_common: QueriedFaceDataset
    ibug_test_challenging: QueriedFaceDataset
    ibug_test_indoor: QueriedFaceDataset
    ibug_test_outdoor: QueriedFaceDataset

    wflw_v_videos: utils.datasets.WFLW_V

    def __init__(self, datasets_dir: Path, rng: np.random.Generator | None = None):
        self.wflw_v_videos = utils.datasets.WFLW_V(datasets_dir / "WFLW_V")
        wflw_v_frames_path = datasets_dir / "WFLW_V_frames"
        if not wflw_v_frames_path.exists():
            wflw_v_videos = utils.datasets.WFLW_V(datasets_dir / "WFLW_V")
            wflw_v_videos.dump_as_images(
                wflw_v_frames_path, padding_ratio=0.25, max_resolution=512, min_lossy_resolution=300, quality=95
            )
        wflw_v = utils.datasets.WFLW_V_Frames(
            wflw_v_frames_path, self.wflw_v_videos, video_clip_len, rng=rng
        )

        split_rng = np.random.Generator(np.random.PCG64(42))

        def split_seed() -> int:
            return int(split_rng.integers(0, 2**31))

        wflw_v_train, wflw_v_val = wflw_v.split(
            800,
            seed=split_seed(),
            train_step=(1, 10),
            train_clip_len=video_clip_len,
            valid_step=(1, 1),
            valid_clip_len=0,
            valid_rng=np.random.default_rng(split_seed()),
        )
        wflw_v_val.output_resolution = test_im_size
        wflw_v_val.padding_ratio = test_padding_ratio
        wflw_v_val, wflw_v_test = wflw_v_val.split(
            50, seed=split_seed(), train_step=(1, 1), train_clip_len=0, valid_step=(1, 1), valid_clip_len=0
        )
        self.wflw_v = QueriedFaceDataset(
            wflw_v_train,
            queries_98_wflw,
            pre_transform=augment_wflw_v,
            post_transform=noise_wflw_v,
            global_weight=1.2,
            include_flipped=False,
            is_clips=True
        )

        self.wflw_v_val = QueriedFaceDataset(wflw_v_val, queries_98_wflw, is_clips=True)
        self.wflw_v_test = QueriedFaceDataset(wflw_v_test, queries_98_wflw, is_clips=True)

        face_synth = utils.datasets.FaceSynthetics(datasets_dir / "FaceSyntheticsSmall", image_ext=".webp")
        face_synth_train, face_synth_val = face_synth.split(0.95, seed=split_seed())
        face_synth_val.output_resolution = test_im_size
        face_synth_val.padding_ratio = test_padding_ratio
        face_synth_val, face_synth_test = face_synth_val.split(0.5, seed=split_seed())

        self.face_synth = QueriedFaceDataset(
            face_synth_train,
            queries_70_synth,
            pre_transform=augment_face_synth,
            post_transform=noise_face_synth,
            global_weight=1.1,
            include_flipped=False,
            is_clips=True
        )
        self.face_synth_val = QueriedFaceDataset(face_synth_val, queries_70_synth, is_clips=False)
        self.face_synth_test = QueriedFaceDataset(face_synth_test, queries_70_synth, is_clips=False)

        wflw = utils.datasets.WFLW(datasets_dir / "WFLW", split="train")
        wflw_train, wflw_val = wflw.split(7000, seed=split_seed())

        self.wflw = QueriedFaceDataset(
            wflw_train,
            queries_98_wflw,
            pre_transform=augment_wflw,
            post_transform=noise_wflw,
            global_weight=0.9,
            include_flipped=True,
            is_clips=True
        )
        wflw_val.output_resolution = test_im_size
        wflw_val.padding_ratio = test_padding_ratio
        self.wflw_val = QueriedFaceDataset(wflw_val, queries_98_wflw, is_clips=False)

        def wflw_split(split: str) -> QueriedFaceDataset:
            d = utils.datasets.WFLW(datasets_dir / "WFLW", split=split)
            d.output_resolution = test_im_size
            d.padding_ratio = test_padding_ratio
            return QueriedFaceDataset(d, queries_98_wflw, is_clips=False)

        self.wflw_test_full = wflw_split("test")
        self.wflw_test_blur = wflw_split("test_blur")
        self.wflw_test_expression = wflw_split("test_expression")
        self.wflw_test_illumination = wflw_split("test_illumination")
        self.wflw_test_largepose = wflw_split("test_largepose")
        self.wflw_test_makeup = wflw_split("test_makeup")
        self.wflw_test_occlusion = wflw_split("test_occlusion")

        ibug = utils.datasets.Ibug(datasets_dir / "300W")
        ibug_train, ibug_val = ibug.split(ibug.num_images - 300, seed=split_seed())
        self.ibug = QueriedFaceDataset(
            ibug_train,
            queries_68_ibug,
            pre_transform=augment_ibug,
            post_transform=noise_ibug,
            global_weight=0.8,
            include_flipped=True,
            is_clips=True
        )

        ibug_val.output_resolution = test_im_size
        ibug_val.padding_ratio = test_padding_ratio
        self.ibug_val = QueriedFaceDataset(ibug_val, queries_68_ibug, is_clips=False)

        self.ibug_test_common = QueriedFaceDataset(
            utils.datasets.IbugTest(datasets_dir / "300W", subset="common"),
            queries_68_ibug,
            is_clips=False
        )
        self.ibug_test_common.dataset.output_resolution = test_im_size
        self.ibug_test_common.dataset.padding_ratio = test_padding_ratio

        self.ibug_test_challenging = QueriedFaceDataset(
            utils.datasets.IbugTest(datasets_dir / "300W", subset="challenging"),
            queries_68_ibug,
            is_clips=False
        )
        self.ibug_test_challenging.dataset.output_resolution = test_im_size
        self.ibug_test_challenging.dataset.padding_ratio = test_padding_ratio

        self.ibug_test_indoor = QueriedFaceDataset(
            utils.datasets.IbugTest(datasets_dir / "300W", subset="indoor"),
            queries_68_ibug,
            is_clips=False
        )
        self.ibug_test_indoor.dataset.output_resolution = test_im_size
        self.ibug_test_indoor.dataset.padding_ratio = test_padding_ratio

        self.ibug_test_outdoor = QueriedFaceDataset(
            utils.datasets.IbugTest(datasets_dir / "300W", subset="outdoor"),
            queries_68_ibug,
            is_clips=False
        )
        self.ibug_test_outdoor.dataset.output_resolution = test_im_size
        self.ibug_test_outdoor.dataset.padding_ratio = test_padding_ratio

    def __str__(self):
        return "\n".join(
            [f"{name} = {dataset.dataset}" for name, dataset in self.__dict__.items() if isinstance(dataset, QueriedFaceDataset)]
        )

    @torch.no_grad()
    def estimate_synth_fps(self, dataset: QueriedFaceDataset, n_samples: int = 1000) -> float:
        """
        Estimate the effective FPS of the synthetic video clips generated by Videoify.
        This is done by sampling a number of clips and measuring the average motion between frames.
        """
        from tqdm import tqdm
        total_synth_motion = 0.0
        n_synth_samples = 0
        for i in tqdm(range(n_samples), desc="Estimating synthetic motion", total=n_samples):
            sample = dataset[i % len(dataset)]
            batch = QueriedFaceDataset.wrap(sample)
            assert batch.clip_len > 0
            
            if batch.dataset.include_flipped:
                labels = batch.labels[0]
            else:
                labels = batch.labels
            h, w = batch.image_size

            delta = labels.diff(dim=0)
            dx, dy = delta.unbind(dim=-1)
            dx /= w
            dy /= h

            val = (dx.square() + dy.square()).sqrt().mean().item()
            if np.isnan(val):
                print(f"Warning: NaN motion in sample {i} (labels={labels.shape}, image_size={batch.image_size}, dx={dx}, dy={dy})")
                continue
            total_synth_motion += val
            n_synth_samples += 1
        avg_synth_motion = total_synth_motion / n_synth_samples * test_im_size
        print(f"avg_synth_motion={avg_synth_motion:.4f} px")

        total_motion = 0.0
        total_fps = 0.0
        for i in tqdm(range(n_samples), desc="Estimating real motion", total=n_samples):
            sample = self.wflw_v[i % len(self.wflw_v)]
            batch = QueriedFaceDataset.wrap(sample)
            assert batch.clip_len > 0

            if batch.dataset.include_flipped:
                labels = batch.labels[0]
            else:
                labels = batch.labels
            assert batch.fps is not None
            h, w = batch.image_size
            
            fps: float = first_element(batch.fps).item()
            total_fps += fps

            delta = labels.diff(dim=0)
            dx, dy = delta.unbind(dim=-1)
            dx /= w
            dy /= h
            total_motion += (dx.square() + dy.square()).sqrt().mean().item()
        avg_motion = total_motion / n_samples * test_im_size
        avg_fps = total_fps / n_samples

        estimated_synth_fps = avg_fps * (avg_synth_motion / avg_motion)
        print(f"estimated_fps={estimated_synth_fps:.2f} (avg_motion={avg_motion:.4f} px, avg_fps={avg_fps:.2f})")
        return estimated_synth_fps

    @classmethod
    @torch.no_grad()
    def measure_frame_correlation(
        cls,
        model: torch.nn.Module,
        dataset: QueriedFaceDataset,
        query_points: QueryPoints,
        n_clips: int = 200,
        max_lag: int = 3,
        iterations: int = 3,
        device: torch.device | None = None,
    ) -> list[float]:
        """
        Measure the temporal autocorrelation of the model's per-landmark prediction error.

        The prediction error e_t = mu_t - y_t of consecutive frames is strongly correlated,
        because consecutive predictions share a recurrent hidden state and are warm-started
        from the previous frame. That correlation determines how the spatial covariance
        Sigma_t propagates through temporal differences of the predictions, and hence the
        ``TEMPORAL_ERROR_CORR`` constant used by the temporal GNLL losses in ``train.py``.

        Lag-k correlation is estimated by pooling cross-products and variances over all
        clips, landmarks and axes, rather than correlating each short per-landmark series
        individually. Per-track normalization is strongly biased toward zero here: removing
        the sample mean of a series only ``clip_len`` frames long also removes most of the
        correlated component (the usual -1/(n-1) small-sample bias). Verified on synthetic
        AR(1) data, the pooled estimator recovers rho to ~1e-3 while the per-track variant
        underestimates rho=0.85 as ~0.33.

        For a true AR(1) process lag-k = rho^k, so comparing the returned lag-2/lag-3 values
        against ``lag1 ** k`` indicates how well the AR(1) assumption of the losses holds.

        Args:
            model: The model to evaluate. Must accept the same arguments as ``QLOT.forward``.
            dataset: A clips dataset (``is_clips=True``) to evaluate on.
            query_points: Query points providing the per-dataset query tensors.
            n_clips: Number of clips to evaluate.
            max_lag: Largest lag to report.
            iterations: Refinement iterations for the first frame. Subsequent frames use one
                iteration, matching the steady-state tracking regime.
            device: Device to run on. Defaults to the model's device.
        Returns:
            A list of ``max_lag`` correlations, where entry ``k - 1`` is the lag-``k``
            correlation of the prediction error.
        """
        from tqdm import tqdm

        assert dataset.is_clips, "measure_frame_correlation requires a clips dataset (is_clips=True)."
        if device is None:
            device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        # Pooled second-moment accumulators (see the docstring on why we do not normalize
        # each short per-landmark series individually).
        sum_e = 0.0
        sum_sq = 0.0
        count = 0
        lag_sums = [0.0] * max_lag
        lag_counts = [0] * max_lag
        clip_errors: list[torch.Tensor] = []

        try:
            for i in tqdm(range(n_clips), desc="Measuring frame correlation", total=n_clips):
                sample = dataset[i % len(dataset)]
                batch = QueriedFaceDataset.wrap(sample, device=device, query_points=query_points.queries)
                clip_len = batch.orig_clip_len
                if clip_len <= max_lag + 1:
                    continue

                errors: list[torch.Tensor] = []
                hidden_state = None
                preds = None
                for frame_i in range(clip_len):
                    frame_batch = batch[frame_i:frame_i+1]
                    preds, hidden_state = model(
                        frame_batch.images,
                        frame_batch.queries,
                        # Converge on the first frame so we measure steady-state tracking error
                        # rather than the startup transient.
                        iterations=iterations if frame_i == 0 else 1,
                        return_sequence=False,
                        store_similarity_maps=False,
                        return_hidden_state=True,
                        prefill_hidden_state=hidden_state,
                        prefill_starting_landmarks=preds,
                    )
                    assert preds is not None
                    errors.append((preds.mean - frame_batch.labels).flatten())

                # (clip_len, batch * queries * 2) -> per-track series along dim 0
                e = torch.stack(errors, dim=0).float().cpu()
                clip_errors.append(e)
                sum_e += e.sum().item()
                sum_sq += e.square().sum().item()
                count += e.numel()
        finally:
            model.train(was_training)

        if count == 0:
            print("measure_frame_correlation: no usable clips (are they longer than max_lag + 1?)")
            return [float("nan")] * max_lag

        # Second pass with the pooled mean removed.
        mean = sum_e / count
        var = max(sum_sq / count - mean**2, 1e-12)
        for e in clip_errors:
            x = e - mean
            for lag in range(1, max_lag + 1):
                if x.shape[0] <= lag:
                    continue
                prod = x[lag:] * x[:-lag]
                lag_sums[lag - 1] += prod.sum().item()
                lag_counts[lag - 1] += prod.numel()

        corrs = [
            (min(max((s / c) / var, -1.0), 1.0) if c > 0 else float("nan")) for s, c in zip(lag_sums, lag_counts)
        ]
        print(f"pooled error std = {var**0.5:.4f} px over {count} (frame, landmark, axis) samples")
        for lag, corr in enumerate(corrs, start=1):
            # For a true AR(1) process, lag-k correlation would be rho^k.
            implied = corrs[0] ** lag if not np.isnan(corrs[0]) else float("nan")
            print(f"lag-{lag} correlation = {corr:+.4f} (AR(1) prediction from lag-1: {implied:+.4f})")
        print(f"\nSuggested TEMPORAL_ERROR_CORR (train.py) = {corrs[0]:.2f}")
        return corrs


__all__ = [
    "data_dir",
    "queries_70_synth",
    "queries_68_ibug",
    "queries_98_wflw",
    "dense_points",
    "dense_points_faces",
    "train_im_size",
    "synth_clip_len",
    "Datasets",
    "make_query_points",
    "queries_opt",
    "face_mesh",
    "set_rng_seed",
    "get_aug_state",
    "set_aug_state",
]
