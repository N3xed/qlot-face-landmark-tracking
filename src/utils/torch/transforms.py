from typing import Any

import torch
from kornia.geometry import get_affine_matrix2d
import albumentations as A
import numpy as np
import cv2


class RandomHorizontalFlip(A.HorizontalFlip):
    def __init__(self, p: float = 0.5, invert_replay: bool = True):
        super().__init__(p)
        self.invert_replay = invert_replay

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.invert_replay and self.replay_mode:
            self.applied_in_replay = not self.applied_in_replay
            if self.applied_in_replay and self.params is None:
                params = self.get_params()
                self.params = self.update_transform_params(params=params, data=kwargs)
        res = super().__call__(*args, **kwargs)
        return res

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().apply_with_params(params, *args, **kwargs)
        # Keep track of whether the sample was hflipped, so we can do the reverse
        # (see QueriedFaceDataset for details).
        data["is_hflipped"] = not params.get("is_hflipped", False)
        return data


class Videoify(A.DualTransform):
    """
    Turns a single image (including keypoints) into a video clip by applying a smooth random walk of affine transformations.
    """

    def __init__(
        self,
        clip_len: int,
        effective_fps: float,
        degrees: None | tuple[float, float] = None,  # Range of rotation angles in degrees
        translate: None | tuple[float, float] = None,  # Relative translation (0 to 1)
        scale: None | tuple[float, float] = None,  # Scale range
        shear: None | tuple[float, float] = None,  # Shear range in degrees
        corr_angle=8,  # Box smoothing filter length for angle
        corr_translate=4,  # Box smoothing filter length for translation
        corr_scale=4,  # Box smoothing filter length for scale
        corr_shear=4,  # Box smoothing filter length for shear
        p: float = 1.0,
    ):
        """
        Args:
            clip_len: Number of frames in the output video clip.
            effective_fps: Effective motion induced physical FPS.
            degrees: Range of rotation angles in degrees. If None, no rotation is applied.
            translate: Relative translation (0 to 1) for horizontal and vertical directions. If None, no translation is applied.
            scale: Scale range as (min_scale, max_scale). If None, no scaling is applied.
            shear: Shear degrees as (min_shear, max_shear). If None, no shearing is applied.
            corr_angle: Box smoothing filter length for angle to create smooth random walk.
            corr_translate: Box smoothing filter length for translation to create smooth random walk.
            corr_scale: Box smoothing filter length for scale to create smooth random walk.
            corr_shear: Box smoothing filter length for shear to create smooth random walk.
        """

        super().__init__(p=p)
        self.clip_len = clip_len
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.corr_angle = corr_angle
        self.corr_translate = corr_translate
        self.corr_scale = corr_scale
        self.corr_shear = corr_shear
        self.effective_fps = effective_fps

    def _generate_smooth_walk(self, length: int, limit: tuple[float, float], corr: int) -> np.ndarray:
        if limit == 0:
            return np.zeros(length)
        low, high = limit
        noise = self.random_generator.uniform(low, high, size=length + corr - 1)
        smoothed = np.convolve(noise, np.ones(corr) / corr, mode="valid")
        return smoothed

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image = data.get("image", None)
        assert image is not None, "Videoify requires 'image' in data to determine transformation parameters."
        h, w = image.shape[:2]

        if self.degrees is not None:
            angle = self._generate_smooth_walk(self.clip_len, self.degrees, self.corr_angle)
        else:
            angle = np.zeros(self.clip_len)

        if self.translate is not None:
            tx = self._generate_smooth_walk(self.clip_len, (self.translate[0] * w, self.translate[1] * w), self.corr_translate)
            ty = self._generate_smooth_walk(self.clip_len, (self.translate[0] * h, self.translate[1] * h), self.corr_translate)
            t = np.stack([tx, ty], axis=-1)
        else:
            t = np.zeros((self.clip_len, 2))

        if self.scale is not None:
            scale = self._generate_smooth_walk(self.clip_len, self.scale, self.corr_scale)
            scale = np.repeat(scale[:, np.newaxis], 2, axis=1)  # Make it (clip_len, 2) for x and y
        else:
            scale = np.ones((self.clip_len, 2))

        if self.shear is not None:
            sx = self._generate_smooth_walk(self.clip_len, self.shear, self.corr_shear)
            sy = self._generate_smooth_walk(self.clip_len, self.shear, self.corr_shear)
        else:
            sx = np.zeros(self.clip_len)
            sy = np.zeros(self.clip_len)

        # Accumulate deltas to get absolute parameters for each frame
        angle = np.cumsum(angle, dtype=np.float32)
        t = np.cumsum(t, axis=0, dtype=np.float32)
        scale = np.cumprod(scale, axis=0, dtype=np.float32)
        sx = np.cumsum(sx, dtype=np.float32) * (np.pi / 180.0)  # Shear needs to be in radians
        sy = np.cumsum(sy, dtype=np.float32) * (np.pi / 180.0)

        center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)

        # Build sequence of transforms in batched way
        with torch.no_grad():
            M = get_affine_matrix2d(
                translations=torch.from_numpy(t),
                center=torch.tensor([center], dtype=torch.float32).expand(self.clip_len, 2),
                scale=torch.from_numpy(scale),
                angle=torch.from_numpy(angle),
                sx=torch.from_numpy(sx),
                sy=torch.from_numpy(sy),
            ).numpy(force=True)
            params["transforms"] = M  # (clip_len, 3, 3) homogeneous transformation matrices for each frame

        return params

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform on image."""

        images = np.empty((self.clip_len, *img.shape), dtype=img.dtype)
        transforms = params["transforms"][:, :2, :]  # warpAffine expects 2x3 matrices

        for t in range(self.clip_len):
            M = transforms[t]
            cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, dst=images[t]
            )
        return images

    def apply_to_keypoints(self, keypoints: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        transforms = params["transforms"][:, :2, :2]  # (clip_len, 2, 2) rotation + scale part of the affine transform
        translation = params["transforms"][:, :2, 2]  # (clip_len, 2) translation part of the affine transform

        N = keypoints.shape[0]
        dims = keypoints.shape[-1]

        keypoints_out = np.empty((self.clip_len, N, dims))
        keypoints_out[...] = keypoints[np.newaxis]

        keypoints_out[..., :2] = keypoints_out[..., :2] @ transforms.transpose(0, 2, 1) + translation[:, np.newaxis, :]
        return keypoints_out.reshape(-1, dims)

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().apply_with_params(params, *args, **kwargs)
        # Add clip_len property, which is read in QueriedFaceDataset.__getitem__ to
        # reshape the keypoints into (clip_len, num_keypoints, 2). The images
        # are already in the right shape since albumentations supports `images` directly.
        data["clip_len"] = self.clip_len
        data["images"] = data.pop("image")  # Rename 'image' to 'images' to reflect that it's now a video clip
        data["fps"] = self.effective_fps  # Add effective_fps to the data dictionary
        return data
