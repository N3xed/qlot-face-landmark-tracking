from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Final, Optional, Iterator
import textwrap
import tqdm
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import json
import cv2
from ..utils import format_dict
from ..video import VideoInfo, decode_video_frames, probe_video


@dataclass
class Dataset:
    """
    A dataset.

    Attributes:
        name: Name of the dataset.
        description: A short description of the dataset.
        source_url: URL to the dataset's source or homepage.
        dl_url: Direct download URL for the dataset.
        can_auto_download: Whether the dataset can be automatically downloaded.
        dir: Directory where the dataset is or will be stored.
        nlandmarks: Number of facial landmarks per image in the dataset.
    """

    name: str
    description: str
    source_url: str
    dl_url: str
    can_auto_download: bool
    dir: Path
    nlandmarks: int
    short_name: str = field(repr=False)

    def __str__(self):
        d = asdict(self)
        for k, v in asdict(self).items():
            if isinstance(v, list):
                del d[k]

        if isinstance(self, AsClips):
            d["num_clips"] = self.num_clips
            d["clip_len"] = self.clip_len
            d["num_videos"] = self.num_videos
            d["step_size"] = (self.min_step, self.max_step)
        
        return format_dict(f"Dataset {self.dir.name}", d)


@dataclass
class AnnotatedVideo:
    """
    A video with associated face bounding boxes and facial landmarks.

    Attributes:
        info: Metadata about the video.
        bboxes: Array of face bounding boxes with shape
            (nframes, 4), where each bounding box is represented as (x1, y1, x2, y2).
        landmarks: Array of facial landmarks with shape
            (nframes, nlandmarks, 2), where each landmark is represented as (x, y).
        frames: Array of video frames in BGR format with shape
            (nframes, height, width, 3).
        iter: An iterator that yields video frames
            one by one in BGR format with shape (height, width, 3).
    """

    info: VideoInfo
    bboxes: NDArray[np.int32]
    landmarks: NDArray[np.float32]
    frames: NDArray[np.uint8] | None = None
    iter: Iterator[NDArray[np.uint8]] | None = None

    @property
    def frames_iter(self) -> Iterator[NDArray[np.uint8]]:
        """
        An iterator that yields video frames one by one in BGR format with shape (height, width, 3).
        If the frames are already loaded in memory, it will yield from the array.
        Otherwise, it will yield from the provided iterator.

        Yields:
            A video frame in BGR format with shape (height, width, 3).
        """
        if self.frames is not None:
            return iter(self.frames)
        elif self.iter is not None:
            return self.iter
        else:
            raise ValueError("No frames or iterator available to yield frames.")

    def annotate_frame(
        self, idx: int, color_bbox=(0, 255, 0), color_landmarks=(0, 0, 255), image: Optional[NDArray[np.uint8]] = None,
        show_bbox=True,
    ) -> NDArray[np.uint8]:
        """
        Annotate a single frame with its bounding box and landmarks.

        Args:
            image: The optional input frame to annotate (H, W, 3) in BGR format.
            idx: Index of the frame to annotate.
            color_bbox: Color for the bounding box in BGR format. Default is green (0, 255, 0).
            color_landmarks: Color for the landmarks in BGR format. Default is red (0, 0, 255).

        Returns:
            The annotated frame.
        """
        import cv2

        if image is None:
            assert self.frames is not None, "No frames available to annotate."
            image = self.frames[idx].copy()
        else:
            image = image.copy()
        if show_bbox:
            bbox = self.bboxes[idx].copy().round().astype(int)
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color_bbox, 2)  # type: ignore

        for x, y in self.landmarks[idx].round().astype(int):
            cv2.circle(image, (x, y), 2, color_landmarks, -1)  # type: ignore
        return image  # type: ignore

    def dump_frames(
        self,
        out_dir: Path,
        video_idx: int,
        padding_ratio=0.25,
        max_resolution: Optional[int] = None,
        min_lossy_resolution: Optional[int] = None,
        quality: int = 95,
        out_landmarks: list[NDArray[np.float32]] | None = None,
    ):
        assert self.frames is not None, "Frames must be loaded in memory to dump them."
        landmarks = self.landmarks.copy()

        video_name = Path(self.info.path).stem
        for j, frame in enumerate(self.frames_iter):
            fh, fw = frame.shape[:2]
            bbox = self.bboxes[j]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1

            # Add padding around bounding box
            pad_w = int(padding_ratio * bw)
            pad_h = int(padding_ratio * bh)
            x1_c = round(x1 - pad_w)
            y1_c = round(y1 - pad_h)
            x2_c = round(x2 + pad_w)
            y2_c = round(y2 + pad_h)

            # Expand to square aspect ratio
            crop_w = x2_c - x1_c
            crop_h = y2_c - y1_c
            if crop_w < crop_h:
                diff = crop_h - crop_w
                x1_c -= diff // 2
                x2_c += diff - diff // 2
            elif crop_h < crop_w:
                diff = crop_w - crop_h
                y1_c -= diff // 2
                y2_c += diff - diff // 2

            # Border replication padding for out-of-bounds regions
            pad_left = max(0, -x1_c)
            pad_top = max(0, -y1_c)
            pad_right = max(0, x2_c - fw)
            pad_bottom = max(0, y2_c - fh)
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
                x1_c += pad_left
                y1_c += pad_top
                x2_c += pad_left
                y2_c += pad_top

            # Crop and move landmarks to cropped frame coordinates
            frame = frame[y1_c:y2_c, x1_c:x2_c]
            landmarks[j][:, 0] -= x1_c - pad_left
            landmarks[j][:, 1] -= y1_c - pad_top

            # Downscale if crop exceeds max_resolution
            crop_size = x2_c - x1_c
            if max_resolution is not None and crop_size > max_resolution:
                scale = max_resolution / crop_size
                frame = cv2.resize(frame, (max_resolution, max_resolution), interpolation=cv2.INTER_LANCZOS4)
                landmarks[j] *= scale

            # Lossless if resolution < min_lossy_resolution, otherwise use specified quality.
            # If no min_lossy_resolution is set, use specified quality for all resolutions.
            # OpenCV encodes lossless WebP when IMWRITE_WEBP_QUALITY == 101.
            is_lossless = True
            this_quality = quality
            if min_lossy_resolution is not None:
                if crop_size >= min_lossy_resolution:
                    is_lossless = False
                else:
                    this_quality = 100

            cv2.imwrite(
                str(out_dir / f"{video_idx:04d}_{video_name}_{j:05d}.webp"),
                frame,
                [cv2.IMWRITE_WEBP_QUALITY, 101 if is_lossless else this_quality],
            )
        if out_landmarks is not None:
            out_landmarks.append(landmarks)

@dataclass
class VideoFrame:
    video_idx: int
    name: str
    frame_idx: int

def parse_video_frame_path(path: Path) -> VideoFrame:
    """
    Parse a video frame path in the format "<video_idx>_<video_name>_<frame_idx>.*".
    
    Returns:
        A tuple of (video_idx, video_name, frame_idx).
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid video frame path: {path}")
    video_idx = int(parts[0])
    video_name = "_".join(parts[1:-1])
    frame_idx = int(parts[-1])
    return VideoFrame(video_idx=video_idx, name=video_name, frame_idx=frame_idx)

@dataclass
class CanonicalLandmarkIndices:
    jaw: list[int]
    brows: list[int]
    nose: list[int]
    eyes: list[int]
    mouth: list[int]
    pupils: list[int] = field(default_factory=lambda: [])
    all: list[int] = field(default_factory=lambda: [])

    def __post_init__(self):
        if len(self.all) == 0:
            self.all = self.jaw + self.brows + self.nose + self.eyes + self.mouth + self.pupils

    def group_name(self, idx: int) -> str | None:
        """
        Get the group name for a given landmark index.

        Args:
            idx: The landmark index to look up.
        Returns:
            The group name as a string, or None if the index is not found in any group.
        """
        for group in ["jaw", "brows", "nose", "eyes", "mouth", "pupils"]:
            try:
                if idx in getattr(self, group):
                    return group
            except:
                continue
        return None


@dataclass
class CanonicalLandmarks:
    points: NDArray[np.float32]
    mesh_indices: CanonicalLandmarkIndices
    indices: CanonicalLandmarkIndices
    group_weights: dict[str, float] = field(default_factory=lambda: {})

    norm_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    norm_scale: float = 1.0

    @staticmethod
    def load(file: str | Path) -> "CanonicalLandmarks":
        with open(file, "r") as f:
            data: dict = json.load(f)
        norm: dict = data.get("normalization", {})
        return CanonicalLandmarks(
            points=np.array(data["points"], dtype=np.float32),
            mesh_indices=CanonicalLandmarkIndices(**data["mesh_indices"]),
            indices=CanonicalLandmarkIndices(**data["indices"]),
            group_weights=data.get("group_weights", {}),
            norm_center=tuple(norm.get("center", (0.0, 0.0, 0.0))),
            norm_scale=norm.get("scale", norm.get("range", 1.0))
        )

    def flip_horizontal_indices(self) -> NDArray[np.int_]:
        """
        Get the indices of the landmarks that correspond to the closest horizontally flipped
        version of the canonical landmarks.

        Returns:
            An array of shape (num_landmarks,) containing the indices of the
            horizontally flipped landmarks.
        """

        num_queries = self.points.shape[0]

        flipped = self.points * np.array([-1, 1, 1])
        indices = np.empty(num_queries, dtype=int)

        for i in range(num_queries):
            flipped_sel = flipped[i]
            queries_dists = np.linalg.norm(self.points - flipped_sel, axis=-1)  # Shape (num_queries,)
            indices[i] = np.argmin(queries_dists)
        return indices

    def save(self, file: str | Path) -> None:
        indices = asdict(self.indices)
        del indices["all"]
        data = {
            "mesh_indices": asdict(self.mesh_indices),
            "indices": indices,
            "normalization": {
                "center": self.norm_center,
                "scale": self.norm_scale,
            },
            "points": self.points.tolist(),
            "group_weights": self.group_weights,
        }
        with open(file, "w") as f:
            json.dump(data, f, indent=4)


@dataclass
class ImageDataset(Dataset):
    num_images: int
    padding_ratio: float
    output_resolution: int | None

    def __init__(
        self,
        name: str,
        short_name: str,
        description: str,
        source_url: str,
        dl_url: str,
        can_auto_download: bool,
        dir: Path,
        nlandmarks: int,
        num_images: int,
    ):
        super().__init__(name, description, source_url, dl_url, can_auto_download, dir, nlandmarks, short_name)
        self.num_images = num_images
        self.padding_ratio = 0.25
        self.output_resolution = 300

    def __len__(self) -> int:
        return self.num_images

    @abstractmethod
    def get(self, idx: int) -> str:
        """
        Get the file path of a specific image.

        Args:
            idx: Index of the image to get.
        Returns:
            The file path of the image.
        """
        pass

    @abstractmethod
    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Load labels for a specific image.

        Args:
            idx: Index of the image to load labels for.
        Returns:
            The face bounding box with shape (4,) in (x1, y1, x2, y2) format,
            and the facial landmarks with shape (nlandmarks, 2).
        """
        pass

    def transform(
        self, img: NDArray[np.uint8], bbox: NDArray[np.int32] | None, landmarks: NDArray[np.float32]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """
        Transform image and landmarks.
    
        Args:
            img: The input image as a numpy array of shape (H, W, 3).
            bbox: The face bounding box as a numpy array of shape (4,) in (x1, y1, x2, y2) format.
            landmarks: The facial landmarks as a numpy array of shape (nlandmarks, 2).

        Returns:
            The transformed image and landmarks.
        """
        if self.padding_ratio == 0.0 and self.output_resolution is None:
            return img, landmarks
        return crop_and_resize(img, landmarks, padding_ratio=self.padding_ratio, output_size=self.output_resolution)

    @abstractmethod
    def clone(self) -> "ImageDataset":
        """
        Create a shallow copy.
        """
        pass

    @abstractmethod
    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple["ImageDataset", "ImageDataset"]:
        """
        Split the dataset into training and validation sets.

        Args:
            train_fraction: Fraction of the dataset to use for training. Or if >1, the
                number of training samples. Defaults to 0.9.
            seed: Random seed for reproducibility. Defaults to None.
        Returns:
            A tuple of (train_dataset, val_dataset).
        """
        pass


class AsClips(ABC):
    clip_len: int  # number of frames in each clip
    min_step: int  # minimum step size between frames in a clip
    max_step: int  # maximum step size between frames in a clip

    @property
    @abstractmethod
    def num_clips(self) -> int:
        """
        Get the total number of clips in the dataset.

        Returns:
            The total number of clips in the dataset.
        """
        pass

    @property
    @abstractmethod
    def num_videos(self) -> int:
        """
        Get the total number of videos in the dataset.

        Returns:
            The total number of videos in the dataset.
        """
        pass

    @abstractmethod
    def get_clip(self, idx: int) -> tuple[list[str], NDArray[np.float32], float]:
        """
        Get a video clip as a list of image file paths and corresponding landmarks.

        Args:
            idx: Index of the video to get a clip from.
        Returns:
            A tuple of (image_paths, landmarks), where:
            - image_paths is a list of file paths for the images in the clip.
            - landmarks is an array of shape (clip_len, nlandmarks, 2) containing the
              facial landmarks for each frame in the clip.
            - The sampled FPS of the clip.
        """


@dataclass
class VideoDataset(Dataset):
    num_videos: int
    num_frames: int

    videos: list[VideoInfo] = field(default_factory=list, repr=False)

    INDEX_FILE = ".video_index.npy"

    def __init__(
        self,
        name: str,
        short_name: str,
        description: str,
        source_url: str,
        dl_url: str,
        can_auto_download: bool,
        dir: Path,
        nlandmarks: int,
        videos: Optional[list[str]] = None,
    ):
        super().__init__(name, description, source_url, dl_url, can_auto_download, dir, nlandmarks, short_name)
        self.num_videos = 0
        self.num_frames = 0
        if videos is not None:
            self.index_videos(videos)

    def __len__(self) -> int:
        return self.num_videos

    def index_videos(self, videos: list[str], force=False) -> None:
        """
        Index the videos in the dataset by probing their metadata and saving it to a file.

        Args:
            videos: List of video file paths to index.
            force: Whether to force re-indexing even if an index file already exists.
        """
        index_file = self.dir / self.INDEX_FILE
        if index_file.exists() and not force:
            self.videos = list(np.load(index_file, allow_pickle=True))
            if len(self.videos) != len(videos):
                print(
                    textwrap.dedent(
                        f"""\
                    Warning: Video index file {index_file} contains {len(self.videos)} entries, but found {len(videos)} video files.
                    Rebuilding the index file..."""
                    )
                )
            else:
                # Check that all videos in index file still exist
                missing_videos = [v for v in self.videos if not Path(v.path).exists()]
                if len(missing_videos) > 0:
                    print(
                        textwrap.dedent(
                            f"""\
                        Warning: The following videos from the index file are missing:
                        {', '.join(v.path for v in missing_videos)}
                        Rebuilding the index file..."""
                        )
                    )
                else:
                    if videos != [v.path for v in self.videos]:
                        print(
                            textwrap.dedent(
                                f"""\
                            Warning: The list of videos has changed since the index file was created.
                            Rebuilding the index file..."""
                            )
                        )
                    else:
                        # All good, use the index file
                        self.num_videos = len(self.videos)
                        self.num_frames = sum(v.nframes for v in self.videos)
                        return

        self.videos = list(probe_video(v) for v in tqdm(videos))
        np.save(index_file, self.videos)  # type: ignore
        self.num_videos = len(self.videos)
        self.num_frames = sum(v.nframes for v in self.videos)

    @abstractmethod
    def load_labels(self, idx: int, start=0, nframes: Optional[int] = None) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Load labels for a specific video.

        Args:
            idx: Index of the video to load labels for.
            start: Frame index to start loading from. Defaults to 0.
            nframes: Number of frames to load. If None, loads all frames from `start` to the end. Defaults to None.
        Returns:
            The face bounding boxes with shape (nframes, 4) in (x1, y1, x2, y2) format,
            and the facial landmarks with shape (nframes, nlandmarks, 2).
        """
        pass

    def load(self, idx: int, start=0, nframes: Optional[int] = None, mem=True) -> AnnotatedVideo:
        """
        Load a video along with its bounding boxes and landmarks.
        Args:
            idx: Index of the video to load.
            start: Frame index to start loading from. Defaults to 0.
            nframes: Number of frames to load. If None, loads all frames from `start` to the end. Defaults to None.
            mem: Whether to load all frames into memory as a numpy array.
        Returns:
            An `AnnotatedVideo` object containing the video frames,
        """
        frames, info = self.get_video_frames(idx, start=start, nframes=nframes, mem=mem)
        bboxes, landmarks = self.load_labels(idx, start=start, nframes=nframes)
        assert (
            info.nframes == bboxes.shape[0] == landmarks.shape[0]
        ), f"Number of frames, bboxes and landmarks do not match: {info.nframes}, {bboxes.shape[0]}, {landmarks.shape[0]}"
        return AnnotatedVideo(
            info=info,
            bboxes=bboxes,
            landmarks=landmarks,
            frames=frames if isinstance(frames, np.ndarray) else None,
            iter=frames if not isinstance(frames, np.ndarray) else None,
        )

    def get_video_frames(self, idx: int, mem=True, **args) -> tuple[NDArray[np.uint8] | Iterator[NDArray[np.uint8]], VideoInfo]:
        """
        Get all frames of a video as a numpy array.
        Args:
            idx: Index of the video to load.
            mem: Whether to load all frames into memory as a numpy array. If False, returns an iterator that yields frames one by one. Defaults to True.
            **args: Additional arguments to pass to `decode_video_frames`.
        Returns:
            The images and video info.
            Images is either a numpy array of shape (nframes, height, width, 3) in BGR
            format, or an iterator that yields frames one by one if `load` is False.
        """

        iter, info = decode_video_frames(self.videos[idx].path, **args)
        if not mem:
            return iter, info
        arr = np.empty((info.nframes, info.height, info.width, 3), dtype=np.uint8)
        for i, frame in enumerate(iter):
            arr[i] = frame
        return arr, info

    DUMP_LABELS_FILE: Final = "labels.npy"
    DUMP_FRAMES_DIR: Final = "frames"

    def dump_as_images(
        self,
        out_dir: str | Path,
        padding_ratio: float = 0.25,
        max_resolution: Optional[int] = None,
        min_lossy_resolution: Optional[int] = None,
        quality: int = 95,
    ):
        """
        Dump all videos as images with associated labels to `out_dir`.

        Each frame is cropped to a square region around the face bounding box with
        padding, using border replication for out-of-bounds areas. If the crop size
        exceeds `max_resolution`, it is downscaled using Lanczos interpolation.

        The images will be saved in a subdirectory `frames`, and the labels
        will be saved in a file `labels.npy` in `out_dir`, with the following format:
        - Each image will be named as `<image_index>_<video_name>_<frame_index>.png`.
        - The labels file will be a numpy array of shape (num_images, nlandmarks, 2),
          containing the facial landmarks for each image.

        Args:
            out_dir: Output directory to save the images and labels.
            padding_ratio: Fraction of the bounding box size to use as padding. Defaults to 0.1.
            max_resolution: If set, crops larger than this side length are downscaled to it.
        """
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = Path(out_dir) / self.DUMP_FRAMES_DIR
        frames_dir.mkdir(parents=True, exist_ok=True)

        def _dump_video(i: int) -> NDArray[np.float32]:
            video = self.load(i, mem=True)
            per_video_landmarks: list[NDArray[np.float32]] = []
            video.dump_frames(
                out_dir=frames_dir,
                video_idx=i,
                padding_ratio=padding_ratio,
                max_resolution=max_resolution,
                min_lossy_resolution=min_lossy_resolution,
                quality=quality,
                out_landmarks=per_video_landmarks,
            )
            return per_video_landmarks[0]

        with ThreadPoolExecutor() as executor:
            landmarks = list(
                tqdm(
                    executor.map(_dump_video, range(self.num_videos)),
                    total=self.num_videos,
                    desc="Dumping videos as images",
                )
            )

        labels_file = Path(out_dir) / self.DUMP_LABELS_FILE
        np.save(labels_file, np.concatenate(landmarks))


def crop_and_resize(
    img: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    padding_ratio=0.25,
    output_size: int | None = 300,
    resize_if_bigger: int | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """
    Crop and resize the image to a square region centered around the landmarks, with some
    padding. The landmarks are also transformed accordingly. If the crop region goes
    outside the image boundaries, it is padded with edge values.

    Args:
        img: Numpy array of shape (H, W, 3)
        landmarks: Numpy array of shape (N, 2) with (x, y) coordinates
        padding_ratio: Percentage of the bounding box size to use as padding around the
                        landmarks
        output_size: The size (in pixels) of the output square image (e.g. 400 for 400x400)
        resize_if_bigger: If set, only resize if the cropped region is larger than this size. This is useful to avoid upscaling small faces.
    Returns:
        A tuple of (cropped_and_resized_img, transformed_landmarks), where:
        - cropped_and_resized_img is a numpy array of shape (output_size, output_size, 3)
        - transformed_landmarks is a numpy array of shape (N, 2).

    Note that landmarks outside of the image bounds are preserved (i.e. neither removed
    nor clipped).
    """

    h, w = img.shape[:2]
    bbox_max = np.max(np.array(landmarks), axis=0).round()
    bbox_min = np.min(np.array(landmarks), axis=0).round()
    x0, y0 = bbox_min.astype(np.int32)
    x1, y1 = bbox_max.astype(np.int32)

    padding = int(max(x1 - x0, y1 - y0) * padding_ratio)

    x0 = x0 - padding
    x1 = x1 + padding
    y0 = y0 - padding
    y1 = y1 + padding

    size = int(max(y1 - y0, x1 - x0))
    if size % 2 != 0:
        size += 1
    half_size = size // 2
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    x0 = center_x - half_size
    x1 = center_x + half_size
    y0 = center_y - half_size
    y1 = center_y + half_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)  # type: ignore
        landmarks = landmarks + np.array([pad_left, pad_top])
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top

    # Crop
    img = img[y0:y1, x0:x1]
    landmarks = landmarks - np.array([x0, y0], dtype=np.float32)
    
    if resize_if_bigger is not None and max(img.shape[:2]) > resize_if_bigger:
        output_size = resize_if_bigger
    if output_size is None:
        return img, landmarks

    # Resize
    old_h, old_w = img.shape[:2]
    img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)  # type: ignore

    # Scale landmarks
    scale_x = output_size / old_w
    scale_y = output_size / old_h
    landmarks = landmarks * np.array([scale_x, scale_y], dtype=np.float32)

    return img, landmarks
