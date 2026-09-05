from enum import StrEnum
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from .base import ImageDataset, VideoDataset, crop_and_resize, AsClips, parse_video_frame_path, VideoInfo
from ..utils import list_files_in_dir
from .video import WFLW_V


class DatasetName(StrEnum):
    """
    Dataset short names.
    """

    WFLW = "WFLW"
    WFLW_V = "WFLW_V"
    FaceSynth = "FaceSynth"
    Ibug = "Ibug"

    def get_testset_name(self, split: str = "") -> str:
        """Get the Datasets attribute name for the corresponding test set."""
        
        if self == DatasetName.WFLW:
            assert split in ["full", "blur", "expression", "illumination", "makeup", "occlusion", "largepose"]
            return f"wflw_test_{split}"
        elif self == DatasetName.WFLW_V:
            assert split == ""
            return f"wflw_v_test"
        elif self == DatasetName.FaceSynth:
            assert split == ""
            return f"face_synth_test"
        elif self == DatasetName.Ibug:
            assert split in ["common", "challenging", "indoor", "outdoor"]
            return f"ibug_test_{split}"
        else:
            raise ValueError(f"Unknown dataset name: {self}")


@dataclass
class WFLW(ImageDataset):
    split_name: str
    images_dir: Path
    label_path: Path
    height = 256
    width = 256

    def __init__(self, dir: str | Path, split: str = "train"):
        self.split_name = split
        self.images_dir = Path(dir) / split
        self.label_path = Path(dir) / (split + ".txt")

        super().__init__(
            name="WFLW (Wider Facial Landmarks in-the-Wild)",
            short_name=DatasetName.WFLW,
            description="A dataset of 10,000 faces in the wild, annotated with 98 landmarks, along with attributes for pose, expression, illumination, makeup, occlusion, and blur.",
            source_url="https://wywu.github.io/projects/LAB/WFLW.html",
            dl_url="https://wywu.github.io/projects/LAB/WFLW.html",
            can_auto_download=False,
            nlandmarks=98,
            dir=Path(dir),
            num_images=0,
        )
        self.padding_ratio = 0.35
        assert (
            self.images_dir.exists() and self.label_path.exists()
        ), f"WFLW dataset directory or label file does not exist: {dir}, please download it manually from {self.dl_url}."

        with open(self.label_path, "r") as f:
            # These landmarks are given in [0,1] range
            data_txt = f.readlines()
        data_info = np.array([x.strip().split() for x in data_txt])

        self._img_paths: list[str] = data_info[:, 0].tolist()
        self.num_images = len(self._img_paths)

        if self.split_name == "train":
            # Fix paths for training set since they contain '_with_box_'
            for i in range(len(self._img_paths)):
                p = self._img_paths[i].split("_")
                p = Path(f"{p[0]}_{p[1]}_with_box_{p[2]}").stem + ".jpg"
                self._img_paths[i] = p

        self._landmarks = data_info[:, 1:].astype(np.float32).reshape(data_info.shape[0], -1, 2).copy()
        del data_txt, data_info

        self._landmarks[:, :, 0] *= self.width
        self._landmarks[:, :, 1] *= self.height

    def get(self, idx: int) -> str:
        return str(self.images_dir / self._img_paths[idx])

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        bboxes = np.empty(4, dtype=np.int32)
        landmarks = self._landmarks[idx]
        return bboxes, landmarks

    def clone(self) -> "WFLW":
        inst = object.__new__(WFLW)
        inst.split_name = self.split_name
        inst.images_dir = self.images_dir
        inst.label_path = self.label_path
        super(WFLW, inst).__init__(
            name=self.name,
            short_name=self.short_name,
            description=self.description,
            source_url=self.source_url,
            dl_url=self.dl_url,
            can_auto_download=self.can_auto_download,
            nlandmarks=self.nlandmarks,
            dir=self.dir,
            num_images=self.num_images,
        )
        inst.padding_ratio = self.padding_ratio
        inst.output_resolution = self.output_resolution
        inst._img_paths = self._img_paths
        inst._landmarks = self._landmarks
        return inst

    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple["WFLW", "WFLW"]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.num_images)
        train_len = int(self.num_images * train_fraction) if train_fraction <= 1 else int(train_fraction)
        assert 0 < train_len < self.num_images, "train_fraction must be in (0, 1) or a valid number of training samples."
        train_indices = indices[:train_len]
        valid_len = self.num_images - train_len
        valid_indices = indices[train_len:]

        train_inst = self.clone()
        train_inst._img_paths = [self._img_paths[i] for i in train_indices]
        train_inst._landmarks = self._landmarks[train_indices]
        train_inst.num_images = train_len

        valid_inst = self.clone()
        valid_inst._img_paths = [self._img_paths[i] for i in valid_indices]
        valid_inst._landmarks = self._landmarks[valid_indices]
        valid_inst.num_images = valid_len

        return train_inst, valid_inst


@dataclass
class WFLW_V_Frames(ImageDataset, AsClips):
    def __init__(
        self,
        dir: str | Path,
        videos: WFLW_V,
        clip_len: int,
        min_step: int = 1,
        max_step: int = 3,
        rng: np.random.Generator | None = None,
    ):
        self.clip_len = clip_len
        assert (
            self.clip_len >= 0
        ), "clip_len must be >= 0. If clip_len=0, it will be set to the minimum number of frames across all videos."
        self.min_step = min_step
        self.max_step = max_step
        self.videos = videos
        self.rng = rng if rng is not None else np.random.default_rng()

        self._frames_dir = Path(dir) / VideoDataset.DUMP_FRAMES_DIR
        self._labels_file = Path(dir) / VideoDataset.DUMP_LABELS_FILE

        if not self._frames_dir.exists() or not self._labels_file.exists():
            raise ValueError(
                f"Directory {self._frames_dir} or labels file {self._labels_file} do not exist. Please run WFLW_V.dump_as_images() first to extract images and labels from the video dataset."
            )

        self._images = list_files_in_dir(self._frames_dir, pattern="*.webp")

        # Sort images by video index and frame index so that we get:
        # video0_frame0, video0_frame1, ..., video0_frameN, video1_frame0, video1_frame1, ..., video1_frameN, ...
        def frame_sort_key(p: str) -> tuple[int, int]:
            f = parse_video_frame_path(Path(p))
            return f.video_idx, f.frame_idx

        self._images.sort(key=frame_sort_key)

        # Note: The landmarks are stored in the same order as the images sorted by video
        # and frame index, they have shape (total_frames, nlandmarks, 2).
        self._landmarks: NDArray[np.float32] = np.load(self._labels_file).astype(np.float32)

        super().__init__(
            name="WFLW_V (Wider Facial Landmarks in-the-Wild Video) Frames",
            short_name=DatasetName.WFLW_V,
            description="A dataset of 1000 creative-commons YouTube videos, each 5 seconds long, designed to be challenging. It is split into 'hard' and 'easy' subsets of 500 videos each. The videos are semi-automatically labeled with a 98-landmark scheme.",
            source_url="https://github.com/polo5/LDEQ_RwR?tab=readme-ov-file#wflw-v-download",
            dl_url="https://drive.google.com/file/d/1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt/view",
            can_auto_download=False,
            nlandmarks=98,
            dir=Path(dir),
            num_images=len(self._images),
        )
        self.padding_ratio: float = 0.4

        self.video_infos: dict[str, VideoInfo] = {}
        for v in self.videos.videos:
            name = Path(v.path).stem
            self.video_infos[name] = v

        # Ordered list of video names as they appear in _images and _landmarks.
        self.ordered_video_names: list[str] = []

        # Mapping from video name to its starting index in _images and _landmarks.
        self.video_start_indices: dict[str, int] = {}

        # Validate video frame ordering and nframes, and build ordered_video_names and video_start_indices.
        seen = set()
        last_frame_idx = -1
        for global_frame_idx, img_path in enumerate(self._images):
            f = parse_video_frame_path(Path(img_path))
            last_video_name = self.ordered_video_names[-1] if self.ordered_video_names else None
            if last_video_name != f.name:
                # The last inserted video name is different, this is a new video.

                # If this assertion fails, it means we encountered a sequence like:
                # video1_frame1, video2_frame1, video1_frame2, which should have been
                # prevented by VideoDataset.dump_as_images(), and the sorting of
                # self._images above.
                assert f.name not in seen, f"Duplicate non-sequential video name found: {f.name}"

                # Since this frame is from a new video, the previous frame was the last
                # frame of the previous video. Check that the recorded length and actual number of frames match.
                last_video_len = self.video_infos[last_video_name].nframes if last_video_name is not None else 0
                assert (
                    last_frame_idx + 1
                ) == last_video_len, f"Video {last_video_name} should have {last_video_len} frames, but has {last_frame_idx + 1}"

                self.ordered_video_names.append(f.name)
                self.video_start_indices[f.name] = global_frame_idx
            else:
                # This is the same video as in the last iteration, mark it as seen.
                seen.add(f.name)
            last_frame_idx = f.frame_idx

        self.clip_start_indices: NDArray[np.int32]
        self.clip_fps: NDArray[np.float32]

        self._build_frame_indices()

    def subset(self, video_names: set[str]) -> None:
        """
        Modify this dataset to contain only the specified video names. This is an in-place operation.

        Args:
            video_names: List of video names to include in the subset.
        """
        self.ordered_video_names = [name for name in self.ordered_video_names if name in video_names]

        all_video_start_indices = self.video_start_indices
        # Indices will change, so we need to rebuild video_start_indices for the new subset.
        self.video_start_indices = {}

        # Filter _images and _landmarks to include only frames from the specified videos
        subset_images = []
        subset_landmarks = []
        global_idx = 0
        for name in self.ordered_video_names:
            video_info = self.video_infos[name]

            start_idx = all_video_start_indices[name]
            end_idx = start_idx + video_info.nframes
            subset_images.extend(self._images[start_idx:end_idx])
            subset_landmarks.append(self._landmarks[start_idx:end_idx])

            self.video_start_indices[name] = global_idx
            global_idx += video_info.nframes

        self._images = subset_images
        self._landmarks = np.concatenate(subset_landmarks, axis=0)
        self.num_images = len(self._images)

        # Rebuild clip indices for the new subset
        self._build_frame_indices()

    def _build_frame_indices(self) -> None:
        clip_start_indices = []
        clip_fps = []
        mean_step = max((self.min_step + self.max_step) // 2, 1)

        if self.clip_len <= 0:
            self.clip_len = min(self.video_infos[name].nframes for name in self.ordered_video_names)

        # We build a list of clip start indices for each video. The start indices are
        # built such that with step<=mean_step, we get non-overlapping clips, while all
        # frames of a clip are guaranteed to be from the same video.
        #
        # Like this we can sample clips with variable step sizes, while actually using
        # most of the frames in the dataset.
        for video_name in self.ordered_video_names:
            global_start_idx = self.video_start_indices[video_name]
            video_info = self.video_infos[video_name]
            global_end_idx = global_start_idx + video_info.nframes

            indices = np.arange(global_start_idx, global_end_idx - (self.clip_len - 1) * self.max_step, mean_step * self.clip_len)
            clip_start_indices.append(indices)

            clip_fps.append(np.full_like(indices, video_info.fps, dtype=np.float32))

        self.clip_start_indices = np.concat(clip_start_indices).astype(np.int32)
        self.clip_fps = np.concat(clip_fps).astype(np.float32)

    def get(self, idx: int) -> str:
        return self._images[idx]

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        bboxes = np.empty(4, dtype=np.int32)
        landmarks = self._landmarks[idx]
        return bboxes, landmarks

    @property
    def num_clips(self) -> int:
        return len(self.clip_start_indices)

    @property
    def num_videos(self) -> int:
        return len(self.ordered_video_names)

    def get_clip(self, idx: int) -> tuple[list[str], NDArray[np.float32], float]:
        start_idx = self.clip_start_indices[idx]
        fps = self.clip_fps[idx]

        if self.min_step == self.max_step:
            step = self.min_step
        else:
            step = int(self.rng.integers(self.min_step, self.max_step, endpoint=True))
        end_idx = start_idx + (self.clip_len * step)

        img_paths = [self._images[i] for i in range(start_idx, end_idx, step)]
        landmarks = self._landmarks[start_idx:end_idx:step]
        return img_paths, landmarks, fps * float(step)

    def clone(self) -> "WFLW_V_Frames":
        inst = object.__new__(WFLW_V_Frames)
        inst.clip_len = self.clip_len
        inst.min_step = self.min_step
        inst.max_step = self.max_step
        inst.videos = self.videos
        inst.rng = self.rng

        inst._frames_dir = self._frames_dir
        inst._labels_file = self._labels_file
        inst._images = self._images
        inst._landmarks = self._landmarks

        super(WFLW_V_Frames, inst).__init__(
            name=self.name,
            short_name=self.short_name,
            description=self.description,
            source_url=self.source_url,
            dl_url=self.dl_url,
            can_auto_download=self.can_auto_download,
            nlandmarks=self.nlandmarks,
            dir=self.dir,
            num_images=self.num_images,
        )
        inst.padding_ratio = self.padding_ratio
        inst.output_resolution = self.output_resolution
        inst.video_infos = self.video_infos
        inst.ordered_video_names = self.ordered_video_names
        inst.video_start_indices = self.video_start_indices
        inst.clip_start_indices = self.clip_start_indices
        inst.clip_fps = self.clip_fps
        return inst

    def split(
        self,
        train_fraction: float = 0.9,
        seed: int | None = None,
        train_step: tuple[int, int] | None = None,
        train_clip_len: int | None = None,
        valid_step: tuple[int, int] | None = None,
        valid_clip_len: int | None = None,
        valid_rng: np.random.Generator | None = None,
    ) -> tuple["WFLW_V_Frames", "WFLW_V_Frames"]:
        """
        Split into training and validation sets.

        Both new datasets will inherit self.rng.
        If valid_rng is provided, it will be used for the validation set instead of self.rng.

        Args:
            train_fraction: Fraction of videos if <= 1, or the number of videos to use for training.
            seed: Random seed for reproducibility.
            train_step: Optional (min_step, max_step) for training set.
            train_clip_len: Optional clip length for training set.
            valid_step: Optional (min_step, max_step) for validation set.
            valid_clip_len: Optional clip length for validation set.
            valid_rng: Optional random generator for validation set.

        Returns:
            A tuple of (train_dataset, valid_dataset).
        """
        rng = np.random.default_rng(seed)
        nvideos = len(self.ordered_video_names)

        num_train_videos = int(nvideos * train_fraction) if train_fraction <= 1 else int(train_fraction)
        assert 0 < num_train_videos < nvideos, "train_fraction must be in (0, 1) or a valid number of training videos."
        num_valid_videos = nvideos - num_train_videos

        easy_video_ids = [name for name in self.ordered_video_names if name in self.videos.easy_video_ids]
        hard_video_ids = [name for name in self.ordered_video_names if name in self.videos.hard_video_ids]
        assert nvideos == len(easy_video_ids) + len(hard_video_ids)

        # Split valid videos proportionally across hard and easy subsets
        easy_frac = len(easy_video_ids) / nvideos
        num_easy_valid = int(num_valid_videos * easy_frac)
        num_hard_valid = num_valid_videos - num_easy_valid
        assert num_easy_valid + num_hard_valid == num_valid_videos

        easy_indices = rng.permutation(len(easy_video_ids))
        easy_valid_indices = easy_indices[:num_easy_valid]
        easy_train_indices = easy_indices[num_easy_valid:]

        hard_indices = rng.permutation(len(hard_video_ids))
        hard_valid_indices = hard_indices[:num_hard_valid]
        hard_train_indices = hard_indices[num_hard_valid:]

        valid_video_ids: set[str] = set(
            [easy_video_ids[i] for i in easy_valid_indices] + [hard_video_ids[i] for i in hard_valid_indices]
        )
        train_video_ids: set[str] = set(
            [easy_video_ids[i] for i in easy_train_indices] + [hard_video_ids[i] for i in hard_train_indices]
        )
        assert valid_video_ids.isdisjoint(train_video_ids), "Overlap between training and validation video IDs."

        # Create train and valid datasets
        train_inst = self.clone()
        if train_step is not None:
            train_inst.min_step, train_inst.max_step = train_step
        if train_clip_len is not None:
            train_inst.clip_len = train_clip_len
        train_inst.subset(train_video_ids)

        valid_inst = self.clone()
        if valid_step is not None:
            valid_inst.min_step, valid_inst.max_step = valid_step
        if valid_clip_len is not None:
            valid_inst.clip_len = valid_clip_len
        valid_inst.subset(valid_video_ids)
        if valid_rng is not None:
            valid_inst.rng = valid_rng

        return train_inst, valid_inst


@dataclass
class FaceSynthetics(ImageDataset):
    def __init__(self, dir: str | Path, image_ext=".png", indices: NDArray[np.int32] | None = None):
        """
        Create a FaceSynthetics dataset instance.

        Args:
            dir: Directory containing the FaceSynthetics dataset.
            image_ext: Image file extension (default: ".png").
            indices: Optional array of indices to subset the dataset. If None, use all images.
        """

        dir = Path(dir)
        super().__init__(
            name="FaceSynthetics",
            short_name=DatasetName.FaceSynth,
            description="A dataset of synthetically generated face images with perfect landmark annotations.",
            source_url="https://github.com/microsoft/FaceSynthetics",
            dl_url="https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_100000.zip",
            can_auto_download=False,
            nlandmarks=70,
            dir=Path(dir),
            num_images=0,
        )
        self.padding_ratio = 0.0
        self.output_resolution = None
        assert (
            dir.exists()
        ), f"FaceSynthetics dataset directory does not exist: {dir}, please download it manually from {self.dl_url}."

        self.num_images = sum(1 for f in dir.iterdir() if f.suffix == ".txt")
        if indices is not None:
            assert indices.ndim == 1, "Indices must be a 1D array."
            assert np.all((indices >= 0) & (indices < self.num_images)), "Indices are out of bounds."
            self.num_images = len(indices)
        self.indices = indices
        self.image_ext = image_ext

    def _get_base_path(self, idx: int) -> str:
        return str(self.dir / f"{idx:06d}")

    def get(self, idx: int) -> str:
        if self.indices is not None:
            idx = self.indices[idx]
        return self._get_base_path(idx) + self.image_ext

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        if self.indices is not None:
            idx = self.indices[idx]
        bboxes = np.empty(4, dtype=np.int32)
        landmarks = np.loadtxt(self._get_base_path(idx) + "_ldmks.txt", dtype=np.float32)
        return bboxes, landmarks

    def clone(self) -> "FaceSynthetics":
        inst = object.__new__(FaceSynthetics)
        inst.image_ext = self.image_ext
        super(FaceSynthetics, inst).__init__(
            name=self.name,
            short_name=self.short_name,
            description=self.description,
            source_url=self.source_url,
            dl_url=self.dl_url,
            can_auto_download=self.can_auto_download,
            nlandmarks=self.nlandmarks,
            dir=self.dir,
            num_images=self.num_images,
        )
        inst.padding_ratio = self.padding_ratio
        inst.output_resolution = self.output_resolution
        inst.indices = self.indices
        return inst

    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple["FaceSynthetics", "FaceSynthetics"]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.num_images)
        train_len = int(self.num_images * train_fraction) if train_fraction <= 1 else int(train_fraction)
        assert 0 < train_len < self.num_images, "train_fraction must be in (0, 1) or a valid number of training samples."
        train_indices = indices[:train_len]
        valid_len = self.num_images - train_len
        valid_indices = indices[train_len:]

        train_inst = self.clone()
        train_inst.num_images = train_len
        if train_inst.indices is not None:
            train_inst.indices = train_inst.indices[train_indices].copy()
        else:
            train_inst.indices = train_indices

        valid_inst = self.clone()
        valid_inst.num_images = valid_len
        if valid_inst.indices is not None:
            valid_inst.indices = valid_inst.indices[valid_indices].copy()
        else:
            valid_inst.indices = valid_indices

        return train_inst, valid_inst

    def dump(
        self,
        out_dir: Path,
        padding_ratio: float = 0.75,
        max_resolution: int = 512,
        min_lossy_resolution: int = 256,
        quality: int = 95,
    ):
        """
        Dump the dataset as lightly cropped image with WebP compression
        making the dataset size much smaller (~10x).
        """
        assert self.indices is None, "Cannot dump a subset of the dataset, please use the full dataset."

        import cv2
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor

        def dump_one(idx: int):
            img_path = self.get(idx)
            _, landmarks = self.load_labels(idx)

            img: NDArray[np.uint8] = cv2.imread(img_path)  # type: ignore
            img, landmarks = crop_and_resize(
                img, landmarks, padding_ratio=padding_ratio, resize_if_bigger=max_resolution, output_size=None
            )
            size = max(img.shape[:2])

            # Lossless if resolution < min_lossy_resolution, otherwise use specified quality.
            # If no min_lossy_resolution is set, use specified quality for all resolutions.
            # OpenCV encodes lossless WebP when IMWRITE_WEBP_QUALITY == 101.
            is_lossless = True
            this_quality = quality
            if min_lossy_resolution is not None:
                if size >= min_lossy_resolution:
                    is_lossless = False
                else:
                    this_quality = 100

            out_path = out_dir / f"{idx:06d}.webp"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_WEBP_QUALITY, 101 if is_lossless else this_quality])
            np.savetxt(out_dir / f"{idx:06d}_ldmks.txt", landmarks)

        out_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(dump_one, range(self.num_images)),
                    total=self.num_images,
                )
            )


def load_pts(path: str | Path) -> NDArray[np.float32]:
    with open(path, "r") as f:
        line0 = f.readline()
        line1 = f.readline()
        assert line0.strip() == "version: 1"
        npts = int(line1.strip().split(":")[1])
    return np.loadtxt(path, skiprows=3, max_rows=npts, dtype=np.float32)


@dataclass
class Ibug(ImageDataset):
    _images: list[str] = field(init=False, repr=False)

    def __init__(self, dir: str | Path):
        dir = Path(dir)
        super().__init__(
            name="Ibug + Helen + AFW + LFPW (300-W)",
            short_name=DatasetName.Ibug,
            description="LFPW, Helen, AFW and Ibug datasets combined, as used in the 300-W challenge. Contains 68-landmark annotations for each image.",
            source_url="https://ibug.doc.ic.ac.uk/resources/300-W/",
            dl_url="",
            can_auto_download=False,
            nlandmarks=68,
            dir=Path(dir),
            num_images=0,
        )
        self.padding_ratio = 0.55
        assert dir.exists()

        self._images = []
        self._images.extend(list_files_in_dir(dir / "helen" / "trainset", pattern="*.pts"))
        self._images.extend(list_files_in_dir(dir / "afw", pattern="*.pts"))
        self._images.extend(list_files_in_dir(dir / "lfpw" / "trainset", pattern="*.pts"))

        for i in reversed(range(len(self._images))):
            img_file = Path(self._images[i]).with_suffix(".jpg")
            landmarks = load_pts(self._images[i])
            if landmarks.shape[0] != self.nlandmarks:
                del self._images[i]
                continue
            if not img_file.exists():
                png_img = img_file.with_suffix(".png")
                if png_img.exists():
                    self._images[i] = str(png_img)
                else:
                    del self._images[i]
            else:
                self._images[i] = str(img_file)
        self.num_images = len(self._images)

    def get(self, idx: int) -> str:
        return self._images[idx]

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        landmarks = load_pts(Path(self._images[idx]).with_suffix(".pts"))
        bboxes = np.empty(4, dtype=np.int32)
        return bboxes, landmarks

    def clone(self) -> "Ibug":
        inst = object.__new__(Ibug)
        super(Ibug, inst).__init__(
            name=self.name,
            short_name=self.short_name,
            description=self.description,
            source_url=self.source_url,
            dl_url=self.dl_url,
            can_auto_download=self.can_auto_download,
            nlandmarks=self.nlandmarks,
            dir=self.dir,
            num_images=self.num_images,
        )
        inst.padding_ratio = self.padding_ratio
        inst.output_resolution = self.output_resolution
        inst._images = self._images.copy()
        return inst

    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple["Ibug", "Ibug"]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.num_images)
        train_len = int(self.num_images * train_fraction) if train_fraction <= 1 else int(train_fraction)
        assert 0 < train_len < self.num_images, "train_fraction must be in (0, 1) or a valid number of training samples."
        train_indices = indices[:train_len]
        valid_len = self.num_images - train_len
        valid_indices = indices[train_len:]

        train_inst = self.clone()
        train_inst._images = [self._images[i] for i in train_indices]
        train_inst.num_images = train_len

        valid_inst = self.clone()
        valid_inst._images = [self._images[i] for i in valid_indices]
        valid_inst.num_images = valid_len

        return train_inst, valid_inst


@dataclass
class IbugTest(ImageDataset):
    _images: list[str] = field(init=False, repr=False)

    def __init__(self, dir: str | Path, subset: str = "common"):
        dir = Path(dir)
        super().__init__(
            name="300-W Ibug Test Set",
            short_name=DatasetName.Ibug,
            description="LFPW, Helen, AFW and Ibug datasets combined, as used in the 300-W challenge. Contains 68-landmark annotations for each image.",
            source_url="https://ibug.doc.ic.ac.uk/resources/300-W/",
            dl_url="",
            can_auto_download=False,
            nlandmarks=68,
            dir=Path(dir),
            num_images=0,
        )
        self.padding_ratio = 0.1
        assert dir.exists()

        self._images = []
        if subset == "common":
            self._images.extend(list_files_in_dir(dir / "helen" / "testset", pattern="*.pts"))
            self._images.extend(list_files_in_dir(dir / "lfpw" / "testset", pattern="*.pts"))
        elif subset == "challenging":
            self._images.extend(list_files_in_dir(dir / "ibug", pattern="*.pts"))
        elif subset == "indoor":
            self._images.extend(list_files_in_dir(dir / "300W" / "01_Indoor", pattern="*.pts"))
        elif subset == "outdoor":
            self._images.extend(list_files_in_dir(dir / "300W" / "02_Outdoor", pattern="*.pts"))
        else:
            raise ValueError(f"Unknown subset '{subset}' for IbugTest dataset.")

        for i in reversed(range(len(self._images))):
            img_file = Path(self._images[i]).with_suffix(".jpg")
            landmarks = load_pts(self._images[i])
            if landmarks.shape[0] != self.nlandmarks:
                del self._images[i]
                continue
            if not img_file.exists():
                png_img = img_file.with_suffix(".png")
                if png_img.exists():
                    self._images[i] = str(png_img)
                else:
                    del self._images[i]
            else:
                self._images[i] = str(img_file)
        self.num_images = len(self._images)

    def get(self, idx: int) -> str:
        return self._images[idx]

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        landmarks = load_pts(Path(self._images[idx]).with_suffix(".pts"))
        bbox = np.empty(4, dtype=np.int32)

        return bbox, landmarks

    def clone(self) -> ImageDataset:
        assert False, "not supported"

    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple[ImageDataset, ImageDataset]:
        assert False, "not supported"


@dataclass
class WFLW_V_TestClips(ImageDataset):
    def __init__(self, dir: str | Path):
        dir = Path(dir)
        self._images = sorted(list_files_in_dir(dir, pattern="*.png"))

        super().__init__(
            name="WFLW_V Test Clips",
            short_name=DatasetName.WFLW_V,
            description="WFLW_V video augmented video clips",
            source_url="https://github.com/polo5/LDEQ_RwR?tab=readme-ov-file#wflw-v-download",
            dl_url="https://drive.google.com/file/d/1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt/view",
            can_auto_download=False,
            nlandmarks=98,
            dir=Path(dir),
            num_images=len(self._images),
        )
        self.padding_ratio = 0.0
        self.output_resolution = None

    def get(self, idx: int) -> str:
        return self._images[idx]

    def load_labels(self, idx: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        bboxes = np.empty(4, dtype=np.int32)
        landmarks = np.load(self._images[idx].replace(".png", "_lbl.npy")).astype(np.float32)
        return bboxes, landmarks

    def clone(self) -> ImageDataset:
        assert False, "not supported"

    def split(self, train_fraction: float = 0.9, seed: int | None = None) -> tuple[ImageDataset, ImageDataset]:
        assert False, "not supported"
