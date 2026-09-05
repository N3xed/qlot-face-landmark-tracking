import numpy as np
from imgui_bundle import imgui, implot, immapp, immvision
import argparse
from pathlib import Path
import torch
import threading
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from kornia.geometry.transform.flips import hflip
import data
from model import QLOT
import model as my_model
from model.lmk_features import LearnedTempSoftShrink
import cv2
import queue
import logging
import time
import utils
import utils.torch
from utils.datasets.image import DatasetName
import cv2.data
from model.onnx import OnnxQLOT

logger = logging.getLogger(__name__)


@dataclass
class AnnotationOverlayBuffer:
    size: int
    canvas: NDArray[np.uint8]


@dataclass(slots=True)
class AnnotationCross:
    x: float
    y: float
    color: tuple[int, int, int]


@dataclass(slots=True)
class AnnotationEllipse:
    x: float
    y: float
    axis_x: float
    axis_y: float
    angle_deg: float
    color: tuple[int, int, int]


@dataclass(slots=True)
class AnnotationPrimitives:
    crosses: list[AnnotationCross]
    ellipses: list[AnnotationEllipse]
    bounds: tuple[int, int, int, int]


def create_annotation_overlay_buffer(size: int = 720) -> AnnotationOverlayBuffer:
    return AnnotationOverlayBuffer(size=size, canvas=np.zeros((size, size, 4), dtype=np.uint8))


IM_UP_SIZE: int = 1024
IM_MID_SIZE: int = 512

class Runner:
    def __init__(self, model: QLOT | OnnxQLOT | None, device: torch.device):
        self.model: QLOT | OnnxQLOT | None = model

        self.im_size = 224

        self.device = device
        self.iterations = 1
        self.last_hidden_state = None
        self.last_predictions = None
        self.last_hidden_state2 = None
        self.last_predictions2 = None
        self.min_variance = 1e4
        self.max_variance = 1
        self.mean_variance = 1.0
        self.flip_indices = torch.tensor(data.queries_98_wflw.flip_horizontal_indices(), dtype=torch.long)

        self.gamma_lut = np.arange(256, dtype=np.uint8)
        self.lmk_mask = None

    def reset(self):
        self.last_hidden_state = None
        self.last_hidden_state2 = None
        self.last_predictions = None
        self.last_predictions2 = None
        
    def set_mask_range(self, range: tuple[int, int] | None, count: int):
        if range is None:
            self.lmk_mask = None
        else:
            start, end = range
            if not (0 <= start < end < count):
                logger.warning(f"Invalid mask range: {range} for count {count}, ignoring mask")
            mask = torch.zeros((1, count), dtype=torch.bool)
            mask[:, start:end+1] = True
            self.lmk_mask = mask

    def detect(
        self,
        frame: np.ndarray,
        queries: torch.Tensor,
        crop_size: float,
        crop_x: float,
        crop_y: float,
        gating_radius: float = 0.0,
        gating_cutoff: float = 0.1,
        store_similarity_maps=False,
        tta_flip=False,
        variance_ema_alpha: float = 0.5,
        forward_hidden: bool = True,
        forward_predictions: bool = True,
        naive_correlation: bool = False,
        up_size: int | None = IM_UP_SIZE,
        iterations: int = 1
    ) -> "Sample | None":
        if self.model is None:
            return None

        # Get image dimensions
        H, W, _ = frame.shape

        # Use warpAffine for sub-pixel fractional cropping and scaling in one step
        src_pts = np.array([[crop_x, crop_y], [crop_x + crop_size, crop_y], [crop_x, crop_y + crop_size]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [IM_MID_SIZE, 0], [0, IM_MID_SIZE]], dtype=np.float32)

        M = cv2.getAffineTransform(src_pts, dst_pts)

        # This gives us the target resolution directly, padding out-of-bounds with black (0,0,0)
        image = cv2.warpAffine(
            frame,
            M,
            (IM_MID_SIZE, IM_MID_SIZE),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        image = cv2.LUT(image, self.gamma_lut)
        if up_size is not None:
            image_up = cv2.resize(image, (up_size, up_size), interpolation=cv2.INTER_AREA)
        else:
            image_up = None

        image = cv2.resize(image, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR)

        # Convert BGR frame (H, W, 3) to RGB tensor (1, 3, H, W)
        image_det = (
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        start_time = cv2.getTickCount()
        
        # Make sure the mask shape matches the queries shape (batch_size, num_queries).
        if self.lmk_mask is not None and self.lmk_mask.shape != queries.shape[:2]:
            self.lmk_mask = None

        # Get predictions in pixel coordinates for visualization
        predictions: my_model.LandmarkPrediction
        self.iterations = iterations
        predictions, last_hidden_state = self.model(
            image_det,
            queries.to(self.device),
            iterations=self.iterations,
            return_hidden_state=True,
            prefill_hidden_state=self.last_hidden_state if forward_hidden else None,
            prefill_starting_landmarks=self.last_predictions if forward_predictions else None,
            gating_radius=gating_radius,
            gating_cutoff=gating_cutoff,
            store_similarity_maps=store_similarity_maps,
            use_naive_correlation=naive_correlation,
            landmarks_to_mask=self.lmk_mask.to(self.device) if self.lmk_mask is not None else None,
        )

        # Detach the hidden state and predictions from the computation graph
        # to prevent history accumulation and memory leaks.
        detached_hidden_state = last_hidden_state.detach()

        if self.last_hidden_state is None:
            self.last_hidden_state = detached_hidden_state
        else:
            self.last_hidden_state.copy_(detached_hidden_state)

        if self.last_predictions is None:
            self.last_predictions = predictions
        else:
            self.last_predictions.mean.copy_(predictions.mean)
            self.last_predictions.cov.params.copy_(predictions.cov.params)

        # Test time augmentation with horizontal flip.
        # We flip the input image, and flip the predicted landmarks back, then average the
        # predictions.
        # This is only for testing, the model is actually trained to minimize hflip discrepancies.
        if tta_flip:
            pred2, hidden2 = self.model(
                hflip(image_det),
                queries.to(self.device)[:, self.flip_indices.cuda(), :],
                iterations=self.iterations,
                return_hidden_state=True,
                prefill_hidden_state=self.last_hidden_state2,
                prefill_starting_landmarks=self.last_predictions2,
                gating_radius=gating_radius,
                gating_cutoff=gating_cutoff,
            )

            if self.last_hidden_state2 is None:
                self.last_hidden_state2 = hidden2
            else:
                self.last_hidden_state2.copy_(hidden2)
            if self.last_predictions2 is None:
                self.last_predictions2 = pred2
            else:
                self.last_predictions2.mean.copy_(pred2.mean)
                self.last_predictions2.cov.params.copy_(pred2.cov.params)

            pred2_mean = pred2.mean.clone()
            pred2_mean[..., 0] = image_det.shape[-1] - 1.0 - pred2.mean[..., 0]
            predictions = predictions.clone()
            predictions.mean = 0.5 * (predictions.mean + pred2_mean)

        end_time = cv2.getTickCount()
        duration_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000.0

        self.min_variance = min(self.min_variance, predictions.cov.min_variance.min().item())
        self.max_variance = max(self.max_variance, predictions.cov.max_variance.max().item())
        # Exponential moving average of the mean max variance.
        self.mean_variance = (
            1.0 - variance_ema_alpha
        ) * self.mean_variance + variance_ema_alpha * predictions.cov.max_variance.mean().item()

        preds = predictions[0].clone()
        preds.mean = preds.mean.detach().cpu()
        preds.cov = preds.cov.detach().to("cpu")

        if store_similarity_maps and isinstance(self.model, QLOT) and self.model.prev_similarity_maps is not None:
            sim_maps = self.model.prev_similarity_maps
            sim_maps = [
                sim_map.detach()[0].cpu() for sim_map in sim_maps
            ]  # List of tensors with shape (num_queries, num_heads, H, W)
        else:
            sim_maps = None
            
        if torch.isnan(preds.mean).any() or torch.isnan(preds.cov.params).any():
            logger.warning("NaN values detected in predictions, skipping this frame")
            return None

        return Sample(
            runner=self,
            raw_frame=frame,
            image=image_up,
            predictions=preds,
            duration_ms=duration_ms,
            min_var=self.min_variance,
            max_var=self.max_variance,
            mean_var=self.mean_variance,
            similarity_maps=sim_maps,
        )

    def set_gamma(self, gamma: float):
        vals = np.arange(256) / 255.0
        self.gamma_lut[:] = ((vals**gamma) * 255.0).clip(0, 255).astype(np.uint8)


@dataclass
class Sample:
    runner: Runner

    raw_frame: NDArray
    image: NDArray | None
    predictions: my_model.LandmarkPrediction
    duration_ms: float

    min_var: float
    max_var: float
    mean_var: float

    # List of tensors with shape (num_queries, num_heads, H, W)
    similarity_maps: list[torch.Tensor] | None = None

    def _collect_annotation_primitives(
        self,
        crop_x: float,
        crop_y: float,
        crop_size: float,
        circle_size: int,
        highlight_idx: int | None,
        draw_cov_ellipse: bool,
    ) -> AnnotationPrimitives | None:
        landmarks_2d = self.predictions.mean.cpu().numpy()  # Shape: (num_queries, 2)
        cov_2d = self.predictions.cov.to_matrix().cpu().numpy()  # Shape: (num_queries, 2, 2)

        max_variance = np.log(max(self.max_var, 1e-12))
        min_variance = np.log(max(self.min_var, 1e-12))
        variance_range = max_variance - min_variance
        scale = crop_size / float(self.runner.im_size)

        crosses: list[AnnotationCross] = []
        ellipses: list[AnnotationEllipse] = []

        bounds_pad = max(circle_size, 1) + 3.0
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for j, (x, y) in enumerate(landmarks_2d):
            confidence = float(max(cov_2d[j, 0, 0], cov_2d[j, 1, 1]))
            confidence_norm = min(
                1.0, (np.log(max(confidence, 1e-12)) - min_variance) / max(variance_range, 1e-6)
            )
            color_intensity = int(255 * confidence_norm)
            color = (0, 255 - color_intensity, color_intensity)
            if highlight_idx is not None and j == highlight_idx:
                color = (color_intensity, 0, 255 - color_intensity)

            global_x = crop_x + float(x) * scale
            global_y = crop_y + float(y) * scale
            crosses.append(AnnotationCross(x=global_x, y=global_y, color=color))

            min_x = min(min_x, global_x - bounds_pad)
            min_y = min(min_y, global_y - bounds_pad)
            max_x = max(max_x, global_x + bounds_pad)
            max_y = max(max_y, global_y + bounds_pad)

            if draw_cov_ellipse:
                cov = cov_2d[j] * (scale * scale)
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-9)
                order = eigvals.argsort()[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]

                chi2_95 = 2.447746830680816
                axes = chi2_95 * np.sqrt(eigvals)
                angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
                ellipses.append(
                    AnnotationEllipse(
                        x=global_x,
                        y=global_y,
                        axis_x=float(axes[0]),
                        axis_y=float(axes[1]),
                        angle_deg=angle_deg,
                        color=color,
                    )
                )

                angle_rad = np.radians(angle_deg)
                cos_a = abs(np.cos(angle_rad))
                sin_a = abs(np.sin(angle_rad))
                radius_x = axes[0] * cos_a + axes[1] * sin_a + 2.0
                radius_y = axes[0] * sin_a + axes[1] * cos_a + 2.0
                min_x = min(min_x, global_x - radius_x)
                min_y = min(min_y, global_y - radius_y)
                max_x = max(max_x, global_x + radius_x)
                max_y = max(max_y, global_y + radius_y)

        if not crosses:
            return None

        bounds = (
            int(np.floor(min_x)),
            int(np.floor(min_y)),
            int(np.ceil(max_x)),
            int(np.ceil(max_y)),
        )
        return AnnotationPrimitives(crosses=crosses, ellipses=ellipses, bounds=bounds)

    def _composite_overlay(
        self,
        image: NDArray[np.uint8],
        bounds: tuple[int, int, int, int],
        overlay_region: NDArray[np.uint8],
    ) -> None:
        x0, y0, x1, y1 = bounds

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(image.shape[1], x1)
        dst_y1 = min(image.shape[0], y1)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return

        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        visible = overlay_region[src_y0:src_y1, src_x0:src_x1]
        if visible.size == 0:
            return

        alpha = visible[..., 3:4].astype(np.float32) / 255.0
        if not np.any(alpha > 0.0):
            return

        dst_roi = image[dst_y0:dst_y1, dst_x0:dst_x1]
        blended = dst_roi.astype(np.float32) * (1.0 - alpha) + visible[..., :3].astype(np.float32)
        dst_roi[...] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    def annotate_frame(
        self,
        overlay_buffer: AnnotationOverlayBuffer,
        circle_size=3,
        highlight_idx: int | None = None,
        draw_cov_ellipse: bool = False,
    ) -> NDArray[np.uint8] | None:
        assert self.image is not None, "No image available for annotation"
        return self.annotate_orig_frame(
            self.image,
            crop_x=0.0,
            crop_y=0.0,
            crop_size=float(self.image.shape[0]),
            circle_size=circle_size,
            highlight_idx=highlight_idx,
            draw_cov_ellipse=draw_cov_ellipse,
            overlay_buffer=overlay_buffer,
        )

    def annotate_orig_frame(
        self,
        image: NDArray[np.uint8],
        crop_x: float,
        crop_y: float,
        crop_size: float,
        overlay_buffer: AnnotationOverlayBuffer,
        circle_size=3,
        highlight_idx: int | None = None,
        draw_cov_ellipse: bool = False,
        draw_bounds: bool = False,
    ) -> NDArray[np.uint8] | None:
        """
        Annotate the original (uncropped) frame `image` with the predicted landmarks within the crop
        defined by `crop_x`, `crop_y`, and `crop_size`.
        
        Args:
            image: The original (uncropped) frame to annotate, modified in-place.
            crop_x: The x-coordinate of the top-left corner of the crop in the original frame.
            crop_y: The y-coordinate of the top-left corner of the crop in the original frame.
            crop_size: The size of the square crop in pixels.
            circle_size: The size of the landmarks to draw for landmarks.
            highlight_idx: Optional index of a landmark to highlight with a different color.
            draw_cov_ellipse: Whether to draw covariance ellipses around landmarks based on their uncertainty.
        """
        try:
            primitives = self._collect_annotation_primitives(
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                circle_size=int(circle_size),
                highlight_idx=highlight_idx,
                draw_cov_ellipse=draw_cov_ellipse,
            )
        except OverflowError as e:
            logger.error(f"OverflowError while collecting annotation primitives: {e}", exc_info=True)
            return image

        if primitives is None:
            print("No landmarks to annotate")
            return None

        crosses = primitives.crosses
        ellipses = primitives.ellipses
        bounds = primitives.bounds
        x0, y0, x1, y1 = bounds
        bbox_w = max(1, x1 - x0)
        bbox_h = max(1, y1 - y0)

        overlay_buffer.canvas.fill(0)
        size = overlay_buffer.size
        # Only downscale when the scratch buffer would overflow. Upscaling the
        # tight landmark bounds makes the supersample factor change from frame
        # to frame, which shows up as flicker when the bounds breathe.
        scale = min(1.0, size / float(bbox_w), size / float(bbox_h))
        active_w = max(1, int(np.ceil(bbox_w * scale)))
        active_h = max(1, int(np.ceil(bbox_h * scale)))
        active_left = (size - active_w) // 2
        active_top = (size - active_h) // 2
        offset_x = active_left - x0 * scale
        offset_y = active_top - y0 * scale

        cross_radius = max(1.0, float(circle_size) * scale)
        cross_thickness = max(2, int(round(2.0 * scale)))
        ellipse_thickness = max(1, int(round(scale)))

        for cross in crosses:
            overlay_x = cross.x * scale + offset_x
            overlay_y = cross.y * scale + offset_y
            coords = (
                np.array(
                    [
                        [overlay_x - cross_radius, overlay_y],
                        [overlay_x + cross_radius, overlay_y],
                        [overlay_x, overlay_y - cross_radius],
                        [overlay_x, overlay_y + cross_radius],
                    ]
                )
                * (2**4)
            )
            coords = coords.round().astype(np.int32)
            color_a = (*cross.color, 255)
            cv2.line(overlay_buffer.canvas, coords[0], coords[1], color_a, cross_thickness, cv2.LINE_AA, shift=4)
            cv2.line(overlay_buffer.canvas, coords[2], coords[3], color_a, cross_thickness, cv2.LINE_AA, shift=4)

        for ellipse in ellipses:
            center = (
                int(round((ellipse.x * scale + offset_x) * (2**4))),
                int(round((ellipse.y * scale + offset_y) * (2**4))),
            )
            axes_i = (
                max(1, int(round(ellipse.axis_x * scale * (2**4)))),
                max(1, int(round(ellipse.axis_y * scale * (2**4)))),
            )
            cv2.ellipse(
                overlay_buffer.canvas,
                center,
                axes_i,
                ellipse.angle_deg,
                0,
                360,
                (*ellipse.color, 255),
                ellipse_thickness,
                cv2.LINE_AA,
                shift=4,
            )

        active = overlay_buffer.canvas[active_top : active_top + active_h, active_left : active_left + active_w]
        if active.shape[1] != bbox_w or active.shape[0] != bbox_h:
            active = cv2.resize(active, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        self._composite_overlay(image, bounds, active)
        if draw_bounds:
            p1 = np.array((crop_x, crop_y))
            p2 = p1 + np.array((crop_size, crop_size))

            p1 = (p1 * (2**4)).round().astype(np.int32)
            p2 = (p2 * (2**4)).round().astype(np.int32)

            cv2.rectangle(image, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2, cv2.LINE_AA, shift=4)
        return image


@dataclass
class Tracker:
    # The maximum crop size in pixels.
    max_size: int

    # Actual crop position and size in pixels.
    crop_size: float
    crop_x: float
    crop_y: float

    # Margin of the crop box to the landmark bounding box as a ratio of it
    margin_ratio: float = 0.12
    # Variance threshold above which we consider the tracking to be lost (after some timeout).
    tracking_lost_var: float = 80.0
    # Max velocity in pixels per frame that the tracker can move the crop box, to prevent large jumps.
    max_velocity: float = 40.0
    # Max velocity in pixels per frame that the tracker can change the crop size, to prevent large jumps.
    max_size_velocity: float = 20.0
    # Additional smoothing factor multiplied by the crop position target delta.
    smoothing_translation: float = 0.12
    # Additional smoothing factor multiplied by the crop size target delta.
    smoothing_size: float = 0.12
    # Additional smoothing factor multiplied by the crop position target delta when in lost state, to slowly move back to center.
    lost_smoothing_translation: float = 0.05
    # Exponential moving average alpha for the velocity, to smooth out the velocity over time.
    velocity_ema_alpha: float = 0.4
    # Minimum crop size in pixels.
    min_size: float = 30.0
    # Exponential moving average alpha for the variance used to determine lost state.
    variance_ema_alpha: float = 0.5
    # Minimum number of frames in lost state, after face detector found a new face
    # which the landmark detector sees to determine that we have successfully relocked onto the new face.
    # If we have not relocked in those frames, we will remain in lost state and move back
    # to a square crop of the whole frame.
    face_detected_frames: int = 10
    # Exponential decay factor of current velocities when variance is above the tracking_lost_var threshold,
    # to slowly reduce velocity when we are likely lost even before we switch to lost state.
    lost_velocity_decay: float = 0.5

    # Lost State variables
    lost_grace_period: int = 60  # Number of frames of high variance before lost state triggers
    mean_var: float | None = None

    # Current velocities of the crop in pixels per frame.
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    size_velocity: float = 0.0

    # Whether we are currently in lost state, note that the tracker may have lost the face
    # well before is_lost becomes True due to the lost_grace_period and variance EMA.
    is_lost = True

    _frames_lost_counter: int = 0
    _detect_cooldown: int = 0
    _reinit_mean_var_next_frame: bool = False
    _was_face_detected: bool = False
    _width: int = 1920
    _height: int = 1080

    def __post_init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def reset_velocity(self):
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.size_velocity = 0.0

    def _smooth_saturate(self, value: float, max_value: float) -> float:
        if max_value <= 0.0:
            return 0.0
        return value / (1.0 + abs(value) / max_value)

    def track(self, sample: Sample):
        next_var = sample.predictions.cov.generalized_variance.mean().item()

        if self._reinit_mean_var_next_frame or self.mean_var is None:
            self.mean_var = next_var
            self._reinit_mean_var_next_frame = False
        else:
            # EMA update of the mean variance, including while in lost state.
            self.mean_var = (1.0 - self.variance_ema_alpha) * self.mean_var + self.variance_ema_alpha * next_var

        if not self.is_lost:
            if self.mean_var > self.tracking_lost_var:
                self._frames_lost_counter += 1
                self.velocity_x *= self.lost_velocity_decay
                self.velocity_y *= self.lost_velocity_decay
                self.size_velocity *= self.lost_velocity_decay
            else:
                self._frames_lost_counter = 0

            if self._frames_lost_counter > self.lost_grace_period:
                self.reset_velocity()
                self.is_lost = True
                self._frames_lost_counter = 0
                self._detect_cooldown = 0

        # Relock from variance recovery even without detector snap.
        if self.mean_var <= self.tracking_lost_var:
            self.is_lost = False
            self._detect_cooldown = 0

        if self.is_lost:
            if self._detect_cooldown <= 0:
                # We are lost - run global face detector at cooldown intervals.
                gray_frame = cv2.cvtColor(sample.raw_frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(30, 30),
                )

                self._detect_cooldown = self.lost_grace_period

                if len(faces) > 0:
                    # Naively take random face
                    fx, fy, fw, fh = faces[np.random.randint(0, len(faces))]

                    # Jump the crop to the newfound face, applying margins
                    target_size = max(fw, fh) * (1 + self.margin_ratio * 2)
                    target_size = min(max(target_size, self.min_size), self.max_size)

                    target_x = (fx + fw / 2.0) - (target_size / 2.0)
                    target_y = (fy + fh / 2.0) - (target_size / 2.0)

                    # Jump to the new face and reinitialize mean_var on next frame.
                    self.crop_size = target_size
                    self.crop_x = target_x
                    self.crop_y = target_y
                    self._reinit_mean_var_next_frame = True
                    self._was_face_detected = True
                    self.reset_velocity()
                    return
            if not self._was_face_detected:

                def smooth_step(error: float) -> float:
                    # Smooth step function that saturates as error increases, to prevent large jumps
                    max_error = min(self._width, self._height) / 2
                    return error / (1 + abs(error) / max_error)

                # If no faces are found, slowly reset boundaries to full screen to catch anyone walking in
                self.crop_size += smooth_step(self.max_size - self.crop_size) * self.smoothing_size
                self.crop_x += smooth_step(self._width / 5 - self.crop_x) * self.lost_smoothing_translation
                self.crop_y += smooth_step(0.0 - self.crop_y) * self.lost_smoothing_translation
            elif self._detect_cooldown < self.lost_grace_period - self.face_detected_frames:
                self._was_face_detected = False

            self._detect_cooldown -= 1

        else:
            # We have the face
            # Calculate the bounding box of the landmarks in the crop space
            landmarks = sample.predictions.mean.cpu().numpy()

            # Convert to global coordinates
            im_size = sample.runner.im_size
            scale = self.crop_size / float(im_size)

            landmarks_global_x = self.crop_x + landmarks[:, 0] * scale
            landmarks_global_y = self.crop_y + landmarks[:, 1] * scale

            # 2. Derive scale size
            min_x, max_x = landmarks_global_x.min(), landmarks_global_x.max()
            min_y, max_y = landmarks_global_y.min(), landmarks_global_y.max()
            w = max_x - min_x
            h = max_y - min_y
            center_x = min_x + w / 2.0
            center_y = min_y + h / 2.0

            padded_w = w * (1 + self.margin_ratio * 2)
            padded_h = h * (1 + self.margin_ratio * 2)
            target_size = max(padded_w, padded_h, self.min_size)
            target_size = min(target_size, self.max_size)

            # 3. Calculate target coordinates
            target_x = center_x - target_size / 2.0
            target_y = center_y - target_size / 2.0

            # Smoothly interpolate towards fractional values
            base_size_step = (target_size - self.crop_size) * self.smoothing_size
            self.size_velocity = (1.0 - self.velocity_ema_alpha) * self.size_velocity + self.velocity_ema_alpha * base_size_step
            size_step = self._smooth_saturate(self.size_velocity, self.max_size_velocity)
            self.crop_size = min(max(self.crop_size + size_step, self.min_size), self.max_size)
            base_step_x = (target_x - self.crop_x) * self.smoothing_translation
            base_step_y = (target_y - self.crop_y) * self.smoothing_translation

            self.velocity_x = (1.0 - self.velocity_ema_alpha) * self.velocity_x + self.velocity_ema_alpha * base_step_x
            self.velocity_y = (1.0 - self.velocity_ema_alpha) * self.velocity_y + self.velocity_ema_alpha * base_step_y

            step_x = self._smooth_saturate(self.velocity_x, self.max_velocity)
            step_y = self._smooth_saturate(self.velocity_y, self.max_velocity)

            self.crop_x += step_x
            self.crop_y += step_y


class Gui:
    def __init__(self, device: torch.device):
        self.device = device
        self.camera_path: str | None = None
        self.model_path: str = ""

        self.model: QLOT | None = None
        self.model_queries = data.queries_opt.clone()
        self.runner = Runner(self.model, self.device)

        self.queue = queue.Queue(maxsize=10)
        self.processing_thread: None | threading.Thread = None
        self.running = False
        self.annotation_overlay = create_annotation_overlay_buffer()

        self.capture = None
        self.width = 1920
        self.height = 1080
        self.fps = 30

        max_size = min(self.width, self.height)
        self.tracker = Tracker(max_size=max_size, crop_size=float(max_size), crop_x=0.0, crop_y=0.0)
        self.tracker._width = self.width
        self.tracker._height = self.height

        self.tracker_enabled = False
        self.crop_size = float(max_size)
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.hflip = False
        self.vflip = False
        self.full_frame = False
        self.naive_correlation = False
        self.forward_hidden = True
        self.forward_predictions = True
        self.mask_start = 0
        self.mask_end = 0
        self.iterations = 1

        self.selected_queries: str = DatasetName.WFLW
        self.avaiable_queries = {
            DatasetName.WFLW: lambda: data.queries_opt.get(DatasetName.WFLW),
            DatasetName.WFLW_V: lambda: data.queries_opt.get(DatasetName.WFLW_V),
            DatasetName.FaceSynth: lambda: data.queries_opt.get(DatasetName.FaceSynth),
            DatasetName.Ibug: lambda: data.queries_opt.get(DatasetName.Ibug),
            "Mesh": lambda: torch.from_numpy(data.face_mesh.points[::8].copy()).to(torch.float32),
            "WFLW-2P": lambda: data.queries_opt.get(DatasetName.WFLW)[[60, 72]],
            "WFLW+FaceSynth": lambda: torch.cat([
                data.queries_opt.get(DatasetName.WFLW)[:-2],
                data.queries_opt.get(DatasetName.FaceSynth)
            ], dim=0),
        }

        self.query_points: torch.Tensor = torch.from_numpy(data.queries_98_wflw.points).clone().unsqueeze(0).to(self.device)
        self.next_query_points: torch.Tensor | None = None

        self.gating_radius = 0.5
        self.gating_cutoff = 0.05
        self.variance_ema_alpha = 0.5
        self.store_similarity_maps = False
        self.tta_flip = False
        self.cross_size = 3
        self.show_cov_ellipse = False
        self.selected_query_idx: int = 0
        self.curr_cmap = implot.Colormap_.hot
        self.gamma = 1.3

        self.show_bypass_fraction = False
        self.show_var_graph = False
        self.show_basis_graph = False
        self.show_cov_graph = False
        self.show_output_graph = False

        self.curr_sample: Sample | None = None
        self.curr_frame: NDArray | None = None
        self.act: None | list[LearnedTempSoftShrink] = None

    @torch.inference_mode()
    def _process_frames(self):
        while self.running:
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if self.hflip:
                frame = frame[:, ::-1]
            if self.vflip:
                frame = frame[::-1, :]
            if self.hflip or self.vflip:
                frame = frame.copy()

            if self.runner.model is None:
                self.curr_frame = frame
                continue

            if self.next_query_points is not None:
                self.query_points = self.next_query_points.to(self.device)
                self.next_query_points = None
                self.runner.reset()
                
            sample = self.runner.detect(
                frame,
                self.query_points,
                crop_size=float(self.crop_size),
                crop_x=float(self.crop_x),
                crop_y=float(self.crop_y),
                gating_radius=self.gating_radius,
                gating_cutoff=self.gating_cutoff,
                store_similarity_maps=self.store_similarity_maps,
                tta_flip=self.tta_flip,
                variance_ema_alpha=self.variance_ema_alpha,
                forward_hidden=self.forward_hidden,
                forward_predictions=self.forward_predictions,
                naive_correlation=self.naive_correlation,
                up_size=None if self.full_frame else IM_UP_SIZE,
                iterations=self.iterations
            )
            if sample is None:
                print("Detection result is None, skipping frame")
                continue

            self.curr_sample = sample
            if sample.image is None:
                self.curr_frame = sample.annotate_orig_frame(
                    sample.raw_frame,
                    crop_x=float(self.tracker.crop_x if self.tracker_enabled else self.crop_x),
                    crop_y=float(self.tracker.crop_y if self.tracker_enabled else self.crop_y),
                    crop_size=float(self.tracker.crop_size if self.tracker_enabled else self.crop_size),
                    circle_size=self.cross_size,
                    draw_cov_ellipse=self.show_cov_ellipse,
                    highlight_idx=self.selected_query_idx if self.store_similarity_maps else None,
                    overlay_buffer=self.annotation_overlay,
                    draw_bounds=True
                )
            else:
                self.curr_frame = sample.annotate_frame(
                    circle_size=self.cross_size,
                    draw_cov_ellipse=self.show_cov_ellipse,
                    highlight_idx=self.selected_query_idx if self.store_similarity_maps else None,
                    overlay_buffer=self.annotation_overlay,
                )

            if self.tracker_enabled:
                self.tracker.track(sample)
                # Allow it to go out of bounds (crop will padding with black borders natively)
                self.tracker.crop_size = min(self.tracker.crop_size, self.tracker.max_size)

                # Apply back to the GUI properties
                self.crop_size = float(self.tracker.crop_size)
                self.crop_x = float(self.tracker.crop_x)
                self.crop_y = float(self.tracker.crop_y)

        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def gui(self):
        imgui.begin("Controls")

        start_btn_text = "Stop Capture" if self.running else "Start Capture"
        if imgui.button(start_btn_text):
            if not self.running:
                self.start_capture(self.camera_path)
            else:
                self.running = False

        imgui.same_line()
        changed, self.camera_path = imgui.input_text("Camera Path", self.camera_path or "", 256)

        if imgui.button("Load Model"):
            p = Path(self.model_path)
            if p.is_file():
                self.load_model(p)
        imgui.same_line()
        changed, self.model_path = imgui.input_text("Model Path", self.model_path, 256)

        changed, self.crop_size = imgui.drag_float("Crop Size", self.crop_size, 1.0, 30.0, float(min(self.width, self.height)))
        changed, self.crop_x = imgui.drag_float("Crop X", self.crop_x, 1.0, float(-self.width), float(self.width))
        changed, self.crop_y = imgui.drag_float("Crop Y", self.crop_y, 1.0, float(-self.height), float(self.height))
        self.tracker.crop_size = self.crop_size
        self.tracker.crop_x = self.crop_x
        self.tracker.crop_y = self.crop_y

        if imgui.begin_combo("Queries", str(self.selected_queries)):
            for dataset in self.avaiable_queries.keys():
                is_selected = dataset == self.selected_queries
                changed, selected = imgui.selectable(str(dataset), is_selected)
                if changed and not is_selected:
                    self.selected_queries = dataset
                    self.next_query_points = self.avaiable_queries[dataset]().clone().unsqueeze(0)
                if changed:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.set_next_item_width(70)
        changed, self.cross_size = imgui.drag_int("Cross Size", self.cross_size, 0.25, 1, 10)
        imgui.same_line()
        changed, self.show_cov_ellipse = imgui.checkbox("Show Covariance Ellipse", self.show_cov_ellipse)
        changed, self.hflip = imgui.checkbox("Horizontal Flip", self.hflip)
        imgui.same_line()
        changed, self.vflip = imgui.checkbox("Vertical Flip", self.vflip)
        imgui.same_line()
        imgui.set_next_item_width(70)
        changed, self.gamma = imgui.drag_float("Gamma", self.gamma, 0.01, 0.1, 3.0)
        if changed:
            self.runner.set_gamma(self.gamma)
        imgui.same_line()
        changed, self.full_frame = imgui.checkbox("Full Frame", self.full_frame)
        changed, self.forward_hidden = imgui.checkbox("Forward Hidden", self.forward_hidden)
        imgui.same_line()
        changed, self.forward_predictions = imgui.checkbox("Forward Predictions", self.forward_predictions)
        changed, self.naive_correlation = imgui.checkbox("Naive Correlation", self.naive_correlation)
        imgui.same_line()
        imgui.set_next_item_width(70)
        changed, self.mask_start = imgui.drag_int("Mask Start", self.mask_start, 0.25, 0, self.query_points.shape[1] - 1)
        imgui.same_line()
        imgui.set_next_item_width(70)
        changed2, self.mask_end = imgui.drag_int("Mask End", self.mask_end, 0.25, 0, self.query_points.shape[1] - 1)
        if changed or changed2:
            self.runner.set_mask_range((self.mask_start, self.mask_end), count=self.query_points.shape[1])
        imgui.same_line()
        imgui.set_next_item_width(70)
        changed, self.iterations = imgui.drag_int("Iterations", self.iterations, 0.1, 1, 10)

        if imgui.collapsing_header("Tracker Settings", flags=imgui.TreeNodeFlags_.default_open):
            changed, self.tracker_enabled = imgui.checkbox("Enable Tracker", self.tracker_enabled)
            changed, self.tracker.min_size = imgui.drag_float("Tracker Min Size", self.tracker.min_size, 1.0, 10.0, self.width)
            changed, self.tracker.margin_ratio = imgui.drag_float(
                "Tracker Margin Ratio", self.tracker.margin_ratio, 0.01, 0.0, 1.0
            )
            changed, self.tracker.tracking_lost_var = imgui.drag_float(
                "Tracker Lost Variance", self.tracker.tracking_lost_var, 1.0, 1.0, 1000.0
            )
            changed, self.tracker.max_velocity = imgui.drag_float(
                "Tracker Max Velocity (px)", self.tracker.max_velocity, 0.5, 0.0, 200.0
            )
            changed, self.tracker.max_size_velocity = imgui.drag_float(
                "Tracker Max Size Velocity (px)", self.tracker.max_size_velocity, 0.5, 0.0, 200.0
            )
            changed, self.tracker.smoothing_translation = imgui.drag_float(
                "Smoothing Trans", self.tracker.smoothing_translation, 0.001, 0.0, 1.0
            )
            changed, self.tracker.smoothing_size = imgui.drag_float(
                "Smoothing Size", self.tracker.smoothing_size, 0.001, 0.0, 1.0
            )
            changed, self.tracker.lost_smoothing_translation = imgui.drag_float(
                "Lost Smoothing Trans", self.tracker.lost_smoothing_translation, 0.001, 0.0, 5.0
            )
            changed, self.tracker.velocity_ema_alpha = imgui.drag_float(
                "Velocity EMA Alpha", self.tracker.velocity_ema_alpha, 0.001, 0.0, 1.0
            )
            changed, self.tracker.lost_grace_period = imgui.drag_int(
                "Lost Grace Period (Frames)", self.tracker.lost_grace_period, 1, 0, 100
            )
            changed, self.tracker.face_detected_frames = imgui.drag_int(
                "Face Detected Frames", self.tracker.face_detected_frames, 1, 0, self.tracker.lost_grace_period
            )
            changed, self.tracker.lost_velocity_decay = imgui.drag_float(
                "Lost Velocity Decay", self.tracker.lost_velocity_decay, 0.01, 0.0, 1.0
            )
            changed, self.tracker.variance_ema_alpha = imgui.drag_float(
                "Tracker Variance EMA Alpha", self.tracker.variance_ema_alpha, 0.01, 0.0, 1.0
            )
            imgui.label_text("##tracker_var", f"Variance Mean: {self.tracker.mean_var or 0.0:.3f}")
            imgui.label_text("##tracker_state", f"Tracker State: {'LOST' if self.tracker.is_lost else 'LOCKED'}")
            imgui.label_text("##tracker_detect_cooldown", f"Face Detect Cooldown: {self.tracker._detect_cooldown}")

        imgui.separator_text("Covariance Gating")

        changed, self.gating_radius = imgui.drag_float("Gating Radius", self.gating_radius, 0.01, 0.0, 10.0)
        changed, self.gating_cutoff = imgui.drag_float("Gating Cutoff", self.gating_cutoff, 0.005, 0.0, 1.0)
        changed, self.variance_ema_alpha = imgui.drag_float("Variance EMA Alpha", self.variance_ema_alpha, 0.01, 0.0, 1.0)
        changed, self.tta_flip = imgui.checkbox("TTA Flip", self.tta_flip)

        imgui.separator()

        changed, self.store_similarity_maps = imgui.checkbox("Store Similarity Maps", self.store_similarity_maps)
        changed, self.selected_query_idx = imgui.drag_int(
            "Query Index", self.selected_query_idx, v_min=0, v_max=self.query_points.shape[1] - 1, v_speed=0.1
        )
        if implot.colormap_button("Colormap", (255, 0), self.curr_cmap):
            self.curr_cmap = (self.curr_cmap + 1) % implot.get_colormap_count()

        imgui.separator()

        if self.curr_sample is not None:
            imgui.label_text("##duration", f"Inference Time: {self.curr_sample.duration_ms:.1f} ms")
            imgui.label_text("##variance", f"Variance Min: {self.curr_sample.min_var:.3f}")
            imgui.label_text("##variance2", f"Variance Max: {self.curr_sample.max_var:.3f}")
            imgui.label_text("##variance3", f"Variance Mean: {self.curr_sample.mean_var:.3f}")

        if imgui.collapsing_header("Plots", flags=imgui.TreeNodeFlags_.default_open) and self.model is not None:
            _, self.show_bypass_fraction = imgui.checkbox("Bypass", self.show_bypass_fraction)
            imgui.same_line()
            _, self.show_var_graph = imgui.checkbox("Var", self.show_var_graph)
            imgui.same_line()
            _, self.show_basis_graph = imgui.checkbox("Basis", self.show_basis_graph)
            imgui.same_line()
            _, self.show_cov_graph = imgui.checkbox("Cov##covgraph", self.show_cov_graph)
            imgui.same_line()
            _, self.show_output_graph = imgui.checkbox("Output", self.show_output_graph)

            
            if self.show_bypass_fraction and self.model.update_predictor.mixer.bypass_fraction is not None:
                if self.store_similarity_maps:
                    frac = self.model.update_predictor.mixer.bypass_fraction.detach()[0, self.selected_query_idx].flatten().cpu().numpy()
                else:
                    frac = self.model.update_predictor.mixer.bypass_fraction.detach()[0].flatten().cpu().numpy()

                if implot.begin_plot(f"##lmk_feat_w"):
                    # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                    implot.plot_bars(f"##bypass_frac", frac)
                    implot.end_plot()

                if self.model.update_predictor.mixer.conf is not None:
                    if self.store_similarity_maps:
                        frac = self.model.update_predictor.mixer.conf.detach()[0, self.selected_query_idx].flatten().cpu().numpy()
                    else:
                        frac = self.model.update_predictor.mixer.conf.detach()[0].flatten().cpu().numpy()

                    if implot.begin_plot(f"##lmk_feat_off"):
                        # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                        implot.plot_bars(f"##offset", frac)
                        implot.end_plot()

            if self.store_similarity_maps and self.show_var_graph:
                if self.model.update_predictor.mixer.local_var is not None:
                    vals = self.model.update_predictor.mixer.local_var.detach()[0, self.selected_query_idx].flatten().cpu().numpy()

                    if implot.begin_plot(f"Local Var"):
                        # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                        implot.plot_bars(f"##localVar", vals)
                        implot.end_plot()

                if self.model.update_predictor.mixer.global_var is not None:
                    vals = self.model.update_predictor.mixer.global_var.detach()[0, self.selected_query_idx].flatten().cpu().numpy()

                    if implot.begin_plot(f"Global Var"):
                        # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                        implot.plot_bars(f"##globalVar", vals)
                        implot.end_plot()

            if self.store_similarity_maps and self.show_basis_graph:
                if self.model.update_predictor.mixer.write is not None:
                    vals = self.model.update_predictor.mixer.write.detach()[0, self.selected_query_idx].flatten().cpu().numpy()

                    if implot.begin_plot(f"Write"):
                        # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                        implot.plot_bars(f"##write_avg", vals)
                        implot.end_plot()

                if self.model.update_predictor.mixer.read is not None:
                    vals = self.model.update_predictor.mixer.read.detach()[0, self.selected_query_idx].flatten().cpu().numpy()

                    if implot.begin_plot(f"Read"):
                        # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                        implot.plot_bars(f"##read", vals)
                        implot.end_plot()

            
            if self.curr_sample is not None and self.show_cov_graph:
                scale = 2.0 / 224.0
                vals = self.curr_sample.predictions.cov.scale_clamp(scale).params.detach().flatten().cpu().numpy()
                if implot.begin_plot(f"Covariance"):
                    # implot.plot_heatmap(f"##lmk_feat_w_", lmk_feat_w, scale_min=0.0, scale_max=1.0)
                    implot.plot_bars(f"##covariance", vals)
                    implot.end_plot()

            if self.show_output_graph:
                if self.model.update_predictor.cand_mult is not None:
                    if self.store_similarity_maps:
                        vals = self.model.update_predictor.cand_mult.detach()[0, self.selected_query_idx].flatten().cpu().numpy()
                    else:
                        vals = self.model.update_predictor.cand_mult.detach()[0].flatten().cpu().numpy()
                    if implot.begin_plot(f"CandMult"):
                        implot.plot_bars(f"##candMult", vals)
                        implot.end_plot()
                if self.model.update_predictor.cand_offset is not None:
                    if self.store_similarity_maps:
                        vals = self.model.update_predictor.cand_offset.detach()[0, self.selected_query_idx].flatten().cpu().numpy()
                    else:
                        vals = self.model.update_predictor.cand_offset.detach()[0].flatten().cpu().numpy()
                    if implot.begin_plot(f"CandOffset"):
                        implot.plot_bars(f"##candOffset", vals)
                        implot.end_plot()
                if self.model.update_predictor.last_hidden_state is not None:
                    if self.store_similarity_maps:
                        vals = self.model.update_predictor.last_hidden_state.detach()[0, self.selected_query_idx].flatten().cpu().numpy()
                    else:
                        vals = self.model.update_predictor.last_hidden_state.detach()[0].square().mean(dim=-1).sqrt().flatten().cpu().numpy()
                    if implot.begin_plot(f"LastHidden"):
                        implot.plot_bars(f"##lastHidden", vals)
                        implot.end_plot()


            imgui.label_text(
                "##temps",
                f"Temps: {torch.exp2(self.model.update_predictor.mixer.write_temperature.detach().cpu().flatten() / 3.0).numpy()}",
            )

        imgui.end()

        if self.curr_frame is not None:
            immvision.use_bgr_color_order()
            immvision.image_display_resizable("frame", self.curr_frame, refresh_image=True)

        if self.curr_sample is not None:
            sample = self.curr_sample
            if sample.similarity_maps is not None:

                axis_flags = (
                    implot.AxisFlags_.lock
                    | implot.AxisFlags_.no_decorations
                    | implot.AxisFlags_.no_grid_lines
                    | implot.AxisFlags_.no_label
                )
                plot_flags = implot.Flags_.no_legend | implot.Flags_.no_frame | implot.Flags_.no_legend
                implot.push_colormap(self.curr_cmap)

                MAP_SIZES = [10, 12, 16]
                for i, sim_map in enumerate(sample.similarity_maps):
                    with torch.no_grad():
                        if self.act is None or len(self.act) != len(sample.similarity_maps):
                            if self.act is None:
                                self.act = []
                            act = LearnedTempSoftShrink(sim_map.shape[-2])
                            act.log_temp.data = self.model.encoder.corr_feat_head_convs[i][0].log_temp.data.detach().cpu().clone() # type: ignore
                            self.act.append(act)
                        sim_map = self.act[i](sim_map[self.selected_query_idx].detach().cpu())[0]

                    head_map: NDArray
                    for j, head_map in enumerate(sim_map.numpy()):
                        B = sim_map.shape[0]
                        sim_map_flat = sim_map.view(B, -1)
                        min_v = sim_map_flat.quantile(0.01, dim=-1).numpy()
                        max_v = sim_map_flat.quantile(0.99, dim=-1).numpy()

                        if j != 0:
                            imgui.same_line()
                        H, W = head_map.shape

                        # imgui.label_text(f"##similarity_map_{i}_{j}", f"Similarity Map {i}, Head {j}, Min: {min_v[j]:.3f}, Max: {max_v[j]:.3f}")
                        temp = self.act[i].log_temp[0, j, 0, 0].exp().item()
                        if implot.begin_plot(f"Min: {min_v[j]:.3f}, Max: {max_v[j]:.3f}, T={temp:.3f}##similarity_map_plot_{i}_{j}", size=(W * MAP_SIZES[i], H * MAP_SIZES[i]), flags=plot_flags):
                            implot.setup_axes("", "", axis_flags, axis_flags)
                            implot.plot_heatmap(f"##hm{i}_{j}", head_map, scale_min=min_v[j], scale_max=max_v[j], label_fmt="")
                            implot.end_plot()
                implot.pop_colormap()

    def load_model(self, path: Path):
        if path.suffix == ".pth":
            self.act = None
            if self.runner.model is not None:
                self.runner.model = None

            def weights_filter(state_dict: dict) -> dict:
                state_dict = my_model.QLOT.translate_weights(state_dict)
                return state_dict

            def load_queries(state_dict: dict):
                nonlocal self
                self.model_queries.load_state_dict(state_dict)

            if self.model is None:
                self.model = QLOT(feature_extractor_pretrained=False)

            steps, self.cfg = utils.torch.misc.load(
                path, self.model, optimizer=None, scheduler=None, model_func=weights_filter, query_points=load_queries
            )
            self.model.eval()

            logger.info(f"Loaded model at step {steps}: {self.cfg}")
            self.model = self.model.to(self.device)
            self.model.eval()
            self.runner.model = self.model
        elif path.suffix == ".onnx" or path.suffix == ".onnx_cpu":
            is_cpu = path.suffix == ".onnx_cpu"
            path = path.with_suffix(".onnx") if is_cpu else path

            use_cuda = not is_cpu and torch.cuda.is_available()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
            self.model = None
            # use_cuda_graph=True captures a CUDA graph per distinct num_queries
            # value (i.e. per landmark topology) the first time it's used --
            # this happens automatically on the first frame after a new
            # query-point set is selected below, since that's exactly when
            # num_queries changes. See OnnxQLOT's docstring for details/caveats.
            self.runner.model = OnnxQLOT(str(path), providers=providers, use_cuda_graph=use_cuda)
        else:
            print(f"Unsupported model format: {path.suffix}")

        self.query_points = self.avaiable_queries[self.selected_queries]().clone().unsqueeze(0).to(self.device)
        self.model_path = str(path) if not path.is_absolute() else str(path.relative_to(Path.cwd(), walk_up=True))

    def close(self):
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join()
            self.processing_thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def start_capture(self, camera_path: str | None = None):
        self.close()
        self.camera_path = camera_path
        try:
            cap = cv2.VideoCapture(camera_path or 0, cv2.CAP_V4L2)

            if not cap.isOpened():
                cap = None
        except Exception as e:
            logger.exception(e)
            cap = None

        if cap is None:
            logger.error(f"Failed to open capture: {self.camera_path or 'default webcam'}")
            return

        # Webcam settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Get camera frame dimensions
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture = cap

        max_size = min(self.width, self.height)
        self.tracker.max_size = max_size
        self.tracker.crop_size = float(max_size)
        self.tracker.crop_x = 0.0
        self.tracker.crop_y = 0.0
        self.tracker._width = self.width
        self.tracker._height = self.height

        self.crop_size = float(max_size)
        self.crop_x = 0.0
        self.crop_y = 0.0

        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()

    def _get_frame(self) -> NDArray[np.uint8] | None:
        if self.capture is None:
            return None
        try:
            ret, frame = self.capture.read()
            if not ret:
                return None
            return frame  #  type: ignore
        except:
            return None


def main():
    parser = argparse.ArgumentParser(description="Model demo")
    parser.add_argument("--model", type=Path, default=None, help="Path to the model file")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu', 'cuda')")
    parser.add_argument("--camera", type=str, default=None, help="Path to the camera file")

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    gui = Gui(device=device)
    if args.model is not None:
        gui.load_model(args.model)
    if args.camera is not None:
        gui.start_capture(args.camera)
        
    run_params = immapp.RunnerParams()
    run_params.callbacks.show_gui = gui.gui
    run_params.fps_idling.vsync_to_monitor = False
    run_params.fps_idling.fps_idle = 40
    run_params.fps_idling.enable_idling = True
    run_params.fps_idling.fps_max = 60
    run_params.app_window_params.window_title = "Model Demo"
    run_params.app_window_params.restore_previous_geometry = True
    addons_params = immapp.AddOnsParams(with_implot=True)

    immapp.run(
        run_params,
        addons_params,
    )
    gui.close()


if __name__ == "__main__":
    main()
