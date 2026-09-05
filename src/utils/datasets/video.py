from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional
from .base import VideoDataset
from ..utils import list_files_in_dir
from numpy.typing import NDArray
import numpy as np

@dataclass
class WFLW_V(VideoDataset):
    BBOXES_DIR: Final[str] = "bboxes"
    LANDMARKS_DIR: Final[str] = "landmarks"
    VIDEOS_DIR: Final[str] = "videos"

    def __init__(self, dir: str | Path):
        dir = Path(dir)
        if not dir.exists():
            raise ValueError(
                f"WFLW-V dataset directory does not exist: {dir}, please download it manually.")
        if not ((dir / self.BBOXES_DIR).exists() and (dir / self.LANDMARKS_DIR).exists() and (dir / self.VIDEOS_DIR).exists()):
            raise ValueError(
                f"WFLW-V dataset directory is missing required subdirectories: {self.BBOXES_DIR}, {self.LANDMARKS_DIR}, {self.VIDEOS_DIR}. Please check the dataset structure.")

        videos = sorted(list_files_in_dir(
            dir / self.VIDEOS_DIR, pattern=["*.mp4", "*.avi", "*.mov"]))

        super().__init__(
            name="WFLW_V (Wider Facial Landmarks in-the-Wild Video)",
            short_name="WFLW_V",
            description="A dataset of 1000 creative-commons YouTube videos, each 5 seconds long, designed to be challenging. It is split into 'hard' and 'easy' subsets of 500 videos each. The videos are semi-automatically labeled with a 98-landmark scheme.",
            source_url="https://github.com/polo5/LDEQ_RwR?tab=readme-ov-file#wflw-v-download",
            dl_url="https://drive.google.com/file/d/1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt/view",
            can_auto_download=False,
            nlandmarks=98,
            dir=dir,
            videos=videos
        )

        self.easy_video_ids: set[str] = set(np.loadtxt(dir / "easy_video_IDs.txt", dtype=str).tolist())
        self.hard_video_ids: set[str] = set(np.loadtxt(dir / "hard_video_IDs.txt", dtype=str).tolist())

        self.video_ids = [Path(v.path).stem for v in self.videos]
        assert set(self.video_ids) == self.easy_video_ids.union(self.hard_video_ids), "Mismatch between video files and ID lists."
        assert self.easy_video_ids.isdisjoint(self.hard_video_ids), "Overlap between easy and hard video IDs."

    @property
    def bboxes(self) -> list[str]:
        return [
            str(self.dir / self.BBOXES_DIR / (Path(v.path).stem + ".npy"))
            for v in self.videos
        ]

    @property
    def landmarks(self) -> list[str]:
        return [
            str(self.dir / self.LANDMARKS_DIR / (Path(v.path).stem + ".npy"))
            for v in self.videos
        ]

    def load_landmarks(self, idx: int, start=0, nframes: Optional[int] = None) -> NDArray[np.float32]:
        """
        Load facial landmarks for a specific video.

        Returns:
            An array of shape (nframes, nlandmarks, 2) containing the facial landmarks.
        """
        landmarks = np.load(self.landmarks[idx])[start:]
        if nframes is not None:
            landmarks = landmarks[:nframes]
        return landmarks

    def load_bboxes(self, idx: int, start=0, nframes: Optional[int] = None) -> NDArray[np.int32]:
        """
        Load face bounding boxes for a specific video.

        Returns:
            An array of shape (nframes, 4) containing the face bounding boxes in (x1, y1, x2, y2) format.
        """
        bboxes = np.load(self.bboxes[idx])[start:]
        if nframes is not None:
            bboxes = bboxes[:nframes]
        return bboxes
    
    def load_labels(self, idx: int, start=0, nframes: Optional[int] = None) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        bboxes = self.load_bboxes(idx, start=start, nframes=nframes)
        landmarks = self.load_landmarks(idx, start=start, nframes=nframes)
        return bboxes, landmarks