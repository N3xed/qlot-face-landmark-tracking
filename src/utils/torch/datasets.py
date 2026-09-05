from utils.datasets.base import VideoDataset
from ..datasets import ImageDataset, CanonicalLandmarks
from ..datasets.base import AsClips

import numpy as np
from functools import partial
from typing import Optional, Any, Iterator
from numpy.typing import NDArray
import dataclasses
from dataclasses import dataclass
from torch.utils.data import Dataset, Sampler
import torch
import albumentations as A
import kornia
import cv2


def weights_for_queries(queries: CanonicalLandmarks, weights: dict[str, float] = {}) -> torch.Tensor:
    """
    Generate weights for each landmark so that all facial features are equally important,
    regardless of the number of landmarks representing them.

    Args:
        queries: An instance of CanonicalLandmarks containing landmark indices.
        weights: A dictionary specifying custom weights for each facial feature group.
                 If a group is not specified, it defaults to 1.0.
    Returns:
        A tensor of weights corresponding to each landmark in queries.indices.all.
    """
    idxs = queries.indices
    groups = {k: v for k, v in dataclasses.asdict(idxs).items() if k != "all"}

    len_groups = {k: len(v) for k, v in groups.items()}

    len_all = sum(len_groups.values())
    assert len_all == len(idxs.all), "Mismatch in total number of landmarks."
    num_groups = len(len_groups)

    custom_weights = {k: weights.get(k, 1.0) for k in groups.keys()}
    res = []
    for idx in idxs.all:
        for group_name, group_idxs in groups.items():
            if len(group_idxs) == 0:
                continue
            if idx in group_idxs:
                group_size = len_groups[group_name]
                weight = custom_weights[group_name]
                res.append(weight)
                break
        else:
            raise ValueError(f"Landmark index {idx} not found in any group.")
    return torch.tensor(res, dtype=torch.float32)


@dataclass
class Batch:
    images: torch.Tensor  # Shape: ([batch_size,] [clip_len,] channels, height, width)
    queries: torch.Tensor  # Shape: ([batch_size,] [clip_len,] num_queries, 3)
    labels: torch.Tensor  # Shape: ([batch_size,] [clip_len,] num_queries, 2)
    weights: torch.Tensor  # Shape: ([batch_size,] [clip_len,] num_queries)
    dataset: "QueriedFaceDataset"
    orig_clip_len: int = 0  # Original clip length before any indexing or slicing
    orig_batch_size: int = 0  # Original batch size before any indexing or slicing
    global_weight: float = 1.0

    # Frames per second for video clips, None for image datasets
    # Shape: ([batch_size,])
    fps: torch.Tensor | None = None

    def get_scales(self) -> Optional[torch.Tensor]:
        """
        Retrieve the scaling factors applied during augmentation for this batch.

        Returns:
            A tensor of scaling factors if RandomAffine augmentation was applied; otherwise, None.
        """
        return None  # TODO: implement

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the height and width of the images in the batch.

        Returns:
            A tuple (height, width) representing the image dimensions.
        """
        h, w = self.images.shape[-2:]
        return h, w

    def normalize_landmarks(self, landmarks=None) -> torch.Tensor:
        """
        Normalize the labels to the range [-1, 1], where (-1, -1) corresponds to the top-left.

        Returns:
            A tensor of normalized labels.
        """
        h, w = self.images.shape[-2:]
        if landmarks is None:
            normalized_labels = self.labels.clone()
        else:
            normalized_labels = landmarks.clone()
        normalized_labels[..., 0] = (2.0 * normalized_labels[..., 0] + 1.0) / w - 1.0
        normalized_labels[..., 1] = (2.0 * normalized_labels[..., 1] + 1.0) / h - 1.0
        return normalized_labels

    @property
    def clip_len(self) -> int:
        if self.orig_clip_len > 0:
            if self.images.ndim == 5:
                return self.images.shape[1]
            elif self.images.ndim == 4:
                return self.images.shape[0]
        return 0

    @property
    def batch_size(self) -> int:
        if self.orig_clip_len > 0:
            if self.images.ndim == 5:
                return self.images.shape[0]
        elif self.images.ndim == 4:
            return self.images.shape[0]
        return 0

    def __getitem__(self, key) -> "Batch":
        if self.fps is not None and self.fps.ndim > 0:
            if isinstance(key, tuple):
                # Only index the first dimension for multi-dimensional slices.
                first_key = key[0]
            else:
                first_key = key
            # Index the batch dimension if available.
            fps = self.fps[first_key]
        else:
            fps = self.fps
        return Batch(
            images=self.images[key],
            queries=self.queries[key],
            labels=self.labels[key],
            weights=self.weights[key],
            dataset=self.dataset,
            orig_clip_len=self.orig_clip_len,
            orig_batch_size=self.orig_batch_size,
            fps=fps,
        )

    def record_stream(self, stream: torch.Stream):
        """
        Record the current stream for all tensors in the batch.

        Args:
            stream: The CUDA stream to record. If None, no action is taken.
        """
        self.images.record_stream(stream)
        self.queries.record_stream(stream)
        self.labels.record_stream(stream)
        self.weights.record_stream(stream)
        if self.fps is not None:
            self.fps.record_stream(stream)


TransformFunc = A.BaseCompose | A.BasicTransform


@dataclass
class TransformResult:
    img: NDArray[np.uint8]
    landmarks: NDArray[np.float32]
    clip_len: int | None = None
    is_hflipped: bool = False
    replay: dict[str, Any] | None = None
    fps: float | None = None

    def __init__(
        self, img: NDArray[np.uint8], landmarks: NDArray[np.float32], transform, is_hflipped=False, replay=None, fps=None
    ):
        lmks_dim = landmarks.shape[-1]
        if transform is not None:
            if replay is not None and isinstance(transform, A.ReplayCompose):
                t = partial(transform.replay, saved_augmentations=replay)
            else:
                t = transform

            if landmarks.ndim == 3:
                assert img.ndim == 4, "Expected (clip_len, C, H, W) images"
                num_landmarks = landmarks.shape[1]
                landmarks = landmarks.reshape(-1, lmks_dim)
                data = t(
                    images=img,
                    keypoints=landmarks,
                    clip_len=landmarks.shape[0] // num_landmarks,
                    is_hflipped=is_hflipped,
                    fps=fps,
                )
                img = data["images"]
            else:
                data = t(image=img, keypoints=landmarks, is_hflipped=is_hflipped, fps=fps)
                img = data.get("images", None)  # type: ignore
                if img is None:
                    img = data["image"]
            landmarks = data["keypoints"]
            self.is_hflipped = data.get("is_hflipped", False)
            self.replay = data.get("replay", None)
            self.fps = data.get("fps", None)

            # landmarks was expanded by Videoify to (clip_len * nlandmarks, 2).
            self.clip_len = data.get("clip_len", None)
            if self.clip_len is not None:
                landmarks = landmarks.reshape(self.clip_len, -1, lmks_dim)
        else:
            self.clip_len = None
            self.is_hflipped = is_hflipped
            self.fps = fps
        self.img = img
        self.landmarks = landmarks

    def apply(self, transform, replay=None) -> "TransformResult":
        return TransformResult(self.img, self.landmarks, transform, is_hflipped=self.is_hflipped, replay=replay, fps=self.fps)


class QueriedFaceDataset(Dataset):
    _constructed_datasets: list["QueriedFaceDataset"] = []

    def __init__(
        self,
        dataset: ImageDataset,
        queries: CanonicalLandmarks,
        transform: Optional[TransformFunc] = None,
        pre_transform: Optional[TransformFunc] = None,
        post_transform: Optional[TransformFunc] = None,
        group_weights: Optional[dict[str, float]] = None,
        is_clips: bool = True,
        global_weight: float = 1.0,
        include_flipped: bool = False,
    ):
        """
        Initialize a QueriedFaceDataset.

        Args:
            dataset: An instance of ImageDataset containing images and labels.
            queries: An instance of CanonicalLandmarks containing query points.
            pre_transform: Optional transform to be applied before the main transform (see include_flipped).
            transform: Optional transform to be applied on a sample.
            post_transform: Optional transform to be applied after the main transform.
            group_weights: Optional dict specifying custom weights for each facial feature group.
            is_clips: Whether the dataset should be treated as clips (video frames) or individual images.
            global_weight: A global loss weight to be applied for this dataset.
            include_flipped: When `include_flipped` is True, the returned sample will consist of an original
                and flipped version. The flipped version is obtained by applying the same
                `*transform` in replay mode (the replay state is obtained from the `*transform`ed original version).
                Thus, for this to work, any of the `*transform` must be a `ReplayCompose`,
                and include a `.transforms.RandomHorizontalFlip` with `invert_replay=True`, this produces
                the horizontally flipped version of the original. Otherwise if `*transform`
                does not include `RandomHorizontalFlip` or if `invert_replay=False`,
                the flipped version will be identical to the original after `*transform`.
                Or, if `pre_transform` is not a `ReplayCompose`, the flipped version will
                have different augmentations applied.
        """
        self.dataset = dataset

        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.transform = transform
        self.global_weight = global_weight

        if isinstance(dataset, AsClips) and is_clips:
            self.len = dataset.num_clips
            self.is_clip_set = True
        else:
            self.len = dataset.num_images
            self.is_clip_set = False
        self.is_clips = is_clips
        self.include_flipped = include_flipped

        self.queries = torch.from_numpy(queries.points)
        self.weights = weights_for_queries(
            queries,
            group_weights if group_weights is not None else queries.group_weights,
        )
        self.canonical_landmarks = queries
        self.flip_indices = torch.tensor(queries.flip_horizontal_indices(), dtype=torch.long)

        idx = len(self.__class__._constructed_datasets)
        self.idx = idx
        self.__class__._constructed_datasets.append(self)

    def unregister(self):
        if self in self.__class__._constructed_datasets:
            self.__class__._constructed_datasets.remove(self)

    @classmethod
    def unregister_all(cls):
        cls._constructed_datasets = []

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        res_flipped = None
        if self.is_clip_set:
            # Dataset implements AsClips interface, so we can get a clip of images and landmarks.
            images, landmarks, fps = self.dataset.get_clip(idx)  # type: ignore

            # Load all images in RGB order
            images: list[NDArray[np.uint8]] = [cv2.imread(img_path, flags=cv2.IMREAD_COLOR_RGB) for img_path in images]  # type: ignore

            landmarks_list = []
            # Apply dataset specific transform to each image/landmark pair.
            for i in range(len(images)):
                images[i], l = self.dataset.transform(images[i], None, landmarks[i])
                landmarks_list.append(l)
            images: NDArray[np.uint8] = np.stack(images, axis=0)  # type: ignore
            landmarks: NDArray[np.float32] = np.stack(landmarks_list, axis=0)  # type: ignore

            # Apply pre_transform.
            res = TransformResult(images, landmarks, self.pre_transform, fps=fps)
            if self.include_flipped:
                # Append a duplicate of the original with replayed augmentations if available.
                res_flipped = TransformResult(images, landmarks, self.pre_transform, replay=res.replay, fps=fps)
        else:
            # Load/transform single image and landmarks.
            images = cv2.imread(self.dataset.get(idx), flags=cv2.IMREAD_COLOR_RGB)  # type: ignore
            bbox, landmarks = self.dataset.load_labels(idx)
            images, landmarks = self.dataset.transform(images, bbox, landmarks)

            # Apply pre_transform.
            res = TransformResult(images, landmarks, self.pre_transform)
            if self.include_flipped:
                # Append a duplicate of the original with replayed augmentations if available.
                res_flipped = TransformResult(images, landmarks, self.pre_transform, replay=res.replay)

        # Apply main transform/post_transform to both original and flipped versions.
        res = res.apply(self.transform)
        if res_flipped is not None:
            res_flipped = res_flipped.apply(self.transform, replay=res.replay)

        res = res.apply(self.post_transform)
        if res_flipped is not None:
            res_flipped = res_flipped.apply(self.post_transform, replay=res.replay)

        # Build sample dict.
        # tensor_images have shape (clip_len, C, H, W) or (C, H, W) if not a clip, and are normalized to [0, 1].
        # labels have shape (clip_len, num_landmarks, 2) or (num_landmarks, 2) if not a clip.
        tensor_images = kornia.utils.image_to_tensor(res.img.astype(np.float32), keepdim=True).div_(255)
        labels = torch.from_numpy(res.landmarks.copy())

        fps: torch.Tensor | None = None
        if res_flipped is not None:
            tensor_images_flipped = kornia.utils.image_to_tensor(res_flipped.img.astype(np.float32), keepdim=True).div_(255)
            labels_flipped = torch.from_numpy(res_flipped.landmarks.copy())

            all_images = torch.stack([tensor_images, tensor_images_flipped], dim=0)
            labels = torch.stack([labels, labels_flipped], dim=0)
            weights = self.weights.expand(2, -1)
            is_hflipped = torch.tensor([res.is_hflipped, res_flipped.is_hflipped], dtype=torch.bool)
            if res_flipped.fps is not None or res.fps is not None:
                assert (
                    res.fps is not None and res_flipped.fps is not None
                ), "Both original and flipped samples must have fps if one has it."
                fps = torch.tensor([res.fps, res_flipped.fps], dtype=torch.float32)
        else:
            all_images = tensor_images
            weights = self.weights
            is_hflipped = torch.tensor(res.is_hflipped, dtype=torch.bool)
            if res.fps is not None:
                fps = torch.tensor(res.fps, dtype=torch.float32)  # Scalar fps for single image/clip

        sample = {
            "image": all_images,  # (clip_len, C, H, W) or (2, clip_len, C, H, W) if include_flipped, or (C, H, W) or (2, C, H, W) if not a clip
            "labels": labels,  # (clip_len, N, 2) or (2, clip_len, N, 2) if include_flipped, or (N, 2) or (2, N, 2) if not a clip
            "weights": weights,  # (N,) or (2, N) if include_flipped
            "idx": torch.tensor(self.idx, dtype=torch.long),  # scalar
            "is_hflipped": is_hflipped,  # (2,) if include_flipped, or scalar otherwise
        }
        if fps is not None:
            sample["fps"] = fps  # scalar or (2,) if include_flipped
        return sample

    @classmethod
    def wrap(
        cls,
        batch: dict,
        device: Optional[torch.device] = None,
        query_points: dict[str, torch.Tensor] | torch.nn.ParameterDict | None = None,
        pin_memory: bool = False,
    ) -> Batch:
        """
        Wrap a batch/sample into a Batch object.

        Apply label reordering based on the is_hflipped flag in the batch,
        to align labels and query points.

        Args:
            batch: A dict from the DataLoader or QueriedFaceDataset containing 'image', 'queries', 'labels', 'weights', and 'idx'.
            copy: Whether to create a copy of the queries before augmentation.
            device: The device to which tensors should be moved.
            query_points: Optional dict or ParameterDict of query points to use instead of those in the dataset.

        Returns:
            A Batch object with augmented images, queries, labels, weights, and dataset reference.
        """

        def unflip_labels(
            dataset: QueriedFaceDataset,
            labels: torch.Tensor,
            is_hflipped: torch.Tensor,
            copy: bool = True,
        ) -> torch.Tensor:
            if is_hflipped.any():
                if copy:
                    labels = labels.clone()
                labels[is_hflipped] = labels[is_hflipped][:, dataset.flip_indices, :]
                labels = labels.contiguous()
            return labels

        images: torch.Tensor = batch["image"]
        labels = batch["labels"]
        weights = batch["weights"]
        is_hflipped: torch.Tensor = batch["is_hflipped"]
        fps: torch.Tensor | None = batch.get("fps", None)

        # We expect that all samples come from the same dataset
        # and so batch["idx"] contains the same dataset index for each sample in the
        # batch, therefore just get the first one.
        dt: QueriedFaceDataset = cls._constructed_datasets[first_element(batch["idx"])]

        if images.ndim == 6:
            # Batch of clips and flipped versions, so shape (batch_size, 2, clip_len, C, H, W)
            # for images, so flatten batch and flip dimensions to shape (batch_size * 2, clip_len, C, H, W).
            images = images.flatten(0, 1)
            labels = labels.flatten(0, 1)
            weights = weights.flatten(0, 1)

        if is_hflipped.ndim == 2:
            # Has shape:
            # - (batch_size, 2) if include_flipped and batched
            # - (2,) if include_flipped and not batched
            # - scalar if not include_flipped and not batched
            # Does not include a clip_len dimension. So flatten
            # batch and flip dimension similarly to images and labels.
            is_hflipped = is_hflipped.flatten(0, 1)

            # Same for fps, if it exists.
            if fps is not None:
                assert fps.ndim == 2
                fps = fps.flatten(0, 1)

        assert 5 >= images.ndim >= 3
        if images.ndim == 5:
            # Easy case: batch of clips (or 2 flipped clips), so shape (batch_size, clip_len, C, H, W)
            # Note: flip dimension is folded into batch dimension.
            batch_size, clip_len = images.shape[:2]
        elif images.ndim == 4:
            # Could be either a batch of images (batch_size, C, H, W) or a single clip
            # (clip_len, C, H, W), we differentiate by checking the dataset.
            if dt.is_clips:
                batch_size = 0
                clip_len = images.shape[0]
            else:
                batch_size = images.shape[0]
                clip_len = 0
        else:
            # Otherwise: a single image (C, H, W) since
            # we only use color images.
            assert images.ndim == 3
            batch_size, clip_len = 0, 0

        if query_points is None:
            queries: torch.Tensor = dt.queries.clone()
        else:
            queries = query_points[dt.dataset.short_name]  # Shape: (num_queries, 3)

        # Expand queries to sampe shape as labels, assumes queries are the same for all
        # samples in a batch/clip.
        if queries.ndim < labels.ndim:
            queries = queries.expand(*labels.shape[:-2], -1, -1)  # Shape: (..., num_queries, 3)

        # Same for weights (except one dimension less).
        if weights.ndim < labels.ndim - 1:
            if weights.ndim == 2:
                weights = weights.unsqueeze(1)
            weights = weights.expand(*labels.shape[:-2], -1)  # Shape: (..., num_queries)

        if clip_len > 0:
            if is_hflipped.ndim == 1:
                is_hflipped = is_hflipped.unsqueeze(1).expand(-1, clip_len)
            else:
                # If not is_hflipped and not batched; this is a scalar so expand to shape (clip_len,)
                is_hflipped = is_hflipped.expand(clip_len)

        # If the sample (image and labels) was flipped during augmentation,
        # we need to unflip (permute) the labels back to the order of the queries.
        labels = unflip_labels(dt, labels, is_hflipped, copy=batch_size == 0)

        def transfer_to_device(tensor: torch.Tensor) -> torch.Tensor:
            if device is None or tensor.device == device:
                return tensor
            if device.type != "cpu":
                if pin_memory and tensor.is_cpu and not tensor.is_pinned():
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    tensor = tensor.pin_memory(device)
                return tensor.to(device, non_blocking=True)
            return tensor.to(device)

        if fps is not None:
            fps = transfer_to_device(fps)

        res = Batch(
            images=transfer_to_device(images),
            queries=transfer_to_device(queries),
            labels=transfer_to_device(labels),
            weights=transfer_to_device(weights),
            dataset=dt,
            orig_clip_len=clip_len,
            orig_batch_size=batch_size,
            global_weight=dt.global_weight,
            fps=fps,
        )
        return res


def first_element(arr: torch.Tensor | NDArray) -> Any:
    """Get the first element of a tensor or array, regardless of its shape."""
    return arr[(0,) * arr.ndim]


def sample_random_clips(
    video_dataset: VideoDataset,
    clip_len: int,
    num_clips: int,
    step: int = 1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample random clip frame indices from a VideoDataset.

    Args:
        video_dataset: The VideoDataset to sample from.
        clip_len: Number of frames per clip.
        num_clips: Total number of clips to sample.
        step: Frame step between frames in a clip.
    Returns:
        A tensor of shape (num_clips * clip_len,) containing the sampled frame indices.
    """

    rng = np.random.default_rng(seed)
    clip_indices = np.arange(clip_len * step, step=step)  # type: ignore

    v_lens = np.array([video.nframes for video in video_dataset.videos])
    v_lens_cumsum = np.concat(([0], np.cumsum(v_lens)))
    video_indices = np.arange(video_dataset.num_videos)

    selected_videos = rng.choice(video_indices, size=num_clips, replace=True)
    batch_clip_indices = video_indices[selected_videos]
    global_clip_indices = v_lens_cumsum[batch_clip_indices]

    batch_clip_sizes = v_lens[batch_clip_indices] - (step * clip_len - 1)
    clip_starts = rng.integers(0, batch_clip_sizes) + global_clip_indices
    clip_starts = np.expand_dims(clip_starts, axis=1) + clip_indices
    clip_starts = clip_starts.flatten()
    return torch.from_numpy(clip_starts)


class DataLoader(torch.utils.data.DataLoader):
    stream_idx: int

    def __init__(
        self,
        dataset: QueriedFaceDataset,
        batch_size: int = 1,
        num_workers=2,
        use_tqdm=False,
        shuffle=True,
        stream_idx: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        self.stream_idx = stream_idx
        self.use_tqdm = use_tqdm
        self._pin_memory = pin_memory

    def _new_iter(self) -> Iterator[dict]:
        it = super().__iter__()
        if self.use_tqdm:
            from tqdm import tqdm

            it = iter(tqdm(it, total=len(self), unit="batch"))
        return it

    def __iter__(self) -> Iterator[dict]:  # type: ignore
        return self._new_iter()

    def __next__(self):
        try:
            return next(self._iter)
        except (AttributeError, StopIteration):
            self._iter = self._new_iter()
        return next(self._iter)

    def wrap_batch(
        self,
        batch: dict,
        device: Optional[torch.device] = None,
        query_points: dict[str, torch.Tensor] | torch.nn.ParameterDict | None = None,
    ) -> Batch:
        """
        Apply dataset-specific augmentations to a batch.

        Args:
            batch: A dict from the DataLoader containing 'image', 'queries', 'labels', 'weights', and 'idx'.
            copy: Whether to create a copy of the queries before augmentation.
            device: The device to which tensors should be moved.
            query_points: Optional dict or ParameterDict of query points to use instead of those in the dataset.
        Returns:
            A Batch object with augmented images, queries, labels, weights, and dataset reference.
        """
        return QueriedFaceDataset.wrap(batch, device=device, query_points=query_points, pin_memory=self._pin_memory)


class StridedClipSampler(Sampler[int]):
    def __init__(self, num_clips: int, shard_id: int, num_shards: int) -> None:
        self.indices = range(shard_id, num_clips, num_shards)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
