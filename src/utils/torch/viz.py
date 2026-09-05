import utils.torch
from utils.torch.datasets import Batch, QueriedFaceDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
from PIL import ImageColor
import torch
import math
import cv2


@torch.no_grad()
def display_sample(data_sample: dict | Batch, show_queries=False, show_weights=True, sample_idx=None):
    """
    Display a sample with its landmarks, weights, and canonical query landmarks.

    Args:
        data_sample (dict): A data sample from a QueriedFaceDataset, or dataloader.
        show_queries (bool): Whether to display the canonical query landmarks.
        show_weights (bool): Whether to display the landmark weights.
        sample_idx (int, optional): Index of the sample for title display.
        clip_len (int): Number of images to display in the clip.
    """

    # Select a sample index
    if isinstance(data_sample, dict):
        sample: utils.torch.datasets.Batch = QueriedFaceDataset.wrap(data_sample)
        print(sample.clip_len, sample.batch_size)
    else:
        sample = data_sample
    canonical_landmarks = sample.dataset.canonical_landmarks

    if (scales := sample.get_scales()) is not None:
        print(scales)

    first_sample = sample
    if sample.batch_size > 0:
        first_sample = sample[0]
    if first_sample.clip_len > 0:
        first_sample = first_sample[0]

    groups = {k: v for k, v in canonical_landmarks.indices.__dict__.items() if k != "all" and len(v) > 0}

    # Create figure with subplots
    if show_queries or show_weights:
        if show_queries and show_weights:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), squeeze=True)
        elif show_weights != show_queries:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), squeeze=True)
            ax2 = ax1
        else:
            assert False

        if show_weights:
            # First subplot: Weights bar chart
            ax1.bar(groups.keys(), [first_sample.weights[vals[0]] for vals in groups.values()])  # type: ignore
            ax1.set_title("Landmark Weights by Group")
            ax1.set_xlabel("Landmark Groups")
            ax1.set_ylabel("Weight")
            ax1.tick_params(axis="x", rotation=45)

        # Third subplot: Canonical query landmarks (if requested)
        if show_queries:
            xq, yq = first_sample.queries[:, 0], first_sample.queries[:, 1]
            ax2.scatter(xq, yq, c="g", s=10, label="Query Landmarks")

            # ax3.scatter(face_synthetics.canonical_landmarks.points[:, 0], face_synthetics.canonical_landmarks.points[:, 1], c='b', s=5, label='Canonical Landmarks')
            # for idx, (x, y) in enumerate(zip(face_synthetics.canonical_landmarks.points[:, 0], face_synthetics.canonical_landmarks.points[:, 1])):
            #     ax3.text(x, y, str(idx), color='blue', fontsize=6, ha='center', va='bottom')

            # Annotate each query landmark with its index
            for idx, (x, y) in enumerate(zip(xq, yq)):
                ax2.text(float(x), float(y+0.005), str(idx), color="purple", fontsize=8, ha="center", va="bottom")

            ax2.set_title("Canonical Query Landmarks")
            # ax3.axis('off')
            ax2.legend()

        fig.tight_layout()
        fig.show()

    for bi in range(max(1, sample.batch_size)):
        if sample.batch_size > 0:
            curr = sample[bi]
        else:
            curr = sample

        num = curr.clip_len if curr.clip_len > 0 else 1
        fig, axes = plt.subplots(1, num, figsize=(num * 6, 6), squeeze=True)

        if num == 1:
            curr = [curr]  # type: ignore
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(num):
            s = curr[i]
            ax = axes[i]

            img_np = s.images.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            ax.imshow(img_np)
            ax.scatter(s.labels[:, 0], s.labels[:, 1], c=np.log(s.weights), s=10, label="Landmarks")

            # Annotate each landmark with its index
            for idx, (x, y) in enumerate(s.labels):
                ax.text(x, y, str(idx), color="red", fontsize=8, ha="center", va="bottom")

        fig.suptitle(f"Sample {str(sample_idx)+' ' if not None else ''}with Landmarks")
        # ax2.axis('off')
        fig.tight_layout()
        fig.show()


def chi_2_squared(p: float) -> float:
    """
    Compute the chi-squared value for 2 degrees of freedom and a given probability p.
    """
    return -2.0 * math.log(1.0 - p)


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: list[torch.Tensor],
    variances: Optional[list[torch.Tensor]] = None,
    connectivity: Optional[list[tuple[int, int]]] = None,
    colors: Optional[Union[str, tuple[int, int, int]] | list[str] | list[tuple[int, int, int]]] = None,
    radius: int | float = 2,
    width: int = 1,
    visibility: Optional[list[torch.Tensor]] = None,
    probability_threshold: float = 0.95,
    scale_to: int | None = None,
) -> torch.Tensor:
    """
    Draws Keypoints on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    Keypoints can be drawn for multiple instances at a time.

    This method allows that keypoints and their connectivity are drawn based on the visibility of this keypoint.

    Adapted from torchvision.utils.draw_keypoints() with added support for covariance ellipses.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
        keypoints (List[Tensor]): List of Tensors, where each Tensor is of shape (K, 2) for each instance,
            with the K keypoint locations in the format [x, y]. Or a Tensor of shape (num_instances, K, 2)
            containing the K keypoint locations for each of the N instances.
        variances (List[Tensor]): List of Tensors, where each Tensor is of shape (K, 3) for each instance,
            specifying the 2D covariance matrix parameters of the K keypoints as (var_x, var_y, cov_xy).
            Or a Tensor of shape (num_instances, K, 3) specifying the 2D covariance matrix parameters
            of the K keypoints for each of the N instances as (var_x, var_y, cov_xy).
        connectivity (List[Tuple[int, int]]]): A List of tuple where each tuple contains a pair of keypoints
            to be connected.
            If at least one of the two connected keypoints has a ``visibility`` of False,
            this specific connection is not drawn.
            Exclusions due to invisibility are computed per-instance.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
        visibility (List[Tensor]): List of Tensors, where each Tensor is of shape (K,) or (K, 1)
            specifying the visibility of the K keypoints for each of the N instances.
            True means that the respective keypoint is visible and should be drawn.
            False means invisible, so neither the point nor possible connections containing it are drawn.
            The input tensors will be cast to bool.
            Default ``None`` means that all the keypoints are visible.
            For more details, see :ref:`draw_keypoints_with_visibility`.
        probability_threshold (float): The probability threshold for drawing probability ellipses.

    Returns:
        img (Tensor[C, H, W]): Image Tensor with keypoints drawn.
    """

    # validate image
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # validate keypoints
    num_inst = len(keypoints)
    assert num_inst > 0, "At least one instance of keypoints must be provided."
    assert all(
        kp.ndim == 2 and kp.shape[1] == 2 for kp in keypoints
    ), "Each keypoints tensor must be of shape (K, 2) for K keypoints."

    if variances is not None:
        assert len(variances) == num_inst, "Number of variance tensors must match number of keypoint instances."
        assert all(
            var.ndim == 2 and var.shape[1] == 3 for var in variances
        ), "Each variances tensor must be of shape (K, 3) for K keypoints with (var_x, var_y, cov_xy)."

        chi2 = chi_2_squared(probability_threshold)
    else:
        chi2 = 0.0
        variances = [torch.zeros(*kp.shape[:-1], 3, dtype=torch.float32) for kp in keypoints]

    # validate visibility
    if visibility is None:  # set default
        visibility = [torch.ones(kp.shape[0], dtype=torch.bool) for kp in keypoints]
    if any(v.ndim == 2 for v in visibility):
        # If visibility was passed as pred.split([2, 1], dim=-1), it will be of shape (num_instances, K, 1).
        # We make sure it is of shape (num_instances, K). This isn't documented, we're just being nice.
        visibility = [v.squeeze(-1) if v.ndim == 2 else v for v in visibility]
    assert len(visibility) == num_inst, "Number of visibility tensors must match number of keypoint instances."

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        from torchvision.transforms.v2.functional import to_dtype  # noqa

        image = to_dtype(image, dtype=torch.uint8, scale=True)

    img_to_draw = image.permute(1, 2, 0).cpu().contiguous().numpy()
    if scale_to is not None:
        scale_factor = scale_to / min(img_to_draw.shape[:2])
        img_to_draw = cv2.resize(
            img_to_draw,
            dsize=None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale_factor = 1.0

    SHIFT = 4
    SHIFT_VAL = 1 << SHIFT
    img_kpts = [(k * SHIFT_VAL * scale_factor).round().to(torch.int64).tolist() for k in keypoints]
    img_variances = [v.to(torch.double).tolist() for v in variances]
    img_vis = [v.cpu().bool().tolist() for v in visibility]
    radius = int(round(radius * SHIFT_VAL))
    max_size = min(*img_to_draw.shape[:2]) // 4

    def cvt_color(color: Union[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        if isinstance(color, str):
            r, g, b, *_ = ImageColor.getrgb(color)
            return (r, g, b)
        else:
            return color

    n_instances = len(img_kpts)
    if colors is None:
        colors = [(0, 0, 255)] * n_instances  # Default color is red
    if isinstance(colors, list):
        assert len(colors) == n_instances, "Length of colors list must match number of instances."
        colors = [cvt_color(c) for c in colors]
    else:
        colors = [cvt_color(colors)] * n_instances  # Single color for all instances

    overlay = np.zeros_like(img_to_draw)
    ELLIPSE_ALPHA = 0.4

    for kpt_inst, vis_inst, var_inst, col_inst in zip(img_kpts, img_vis, img_variances, colors):
        for kpt_coord, kp_vis, kp_var in zip(kpt_inst, vis_inst, var_inst):
            if not kp_vis:
                continue
            cv2.circle(
                img_to_draw,
                (kpt_coord[0], kpt_coord[1]),
                radius,
                color=col_inst,
                thickness=-1,  # Filled circle
                lineType=cv2.LINE_AA,
                shift=SHIFT,
            )

            var_x, var_y, cov_xy = kp_var
            if var_x > 0 and var_y > 0:
                try:
                    discriminant = math.sqrt(((var_x - var_y) ** 2) / 4.0 + cov_xy**2)
                    a = math.sqrt(((var_x + var_y) / 2.0 + discriminant) * chi2)
                    b = math.sqrt(((var_x + var_y) / 2.0 - discriminant) * chi2)
                    theta = 0.5 * math.atan2(2 * cov_xy, var_x - var_y)

                    if a <= max_size and b <= max_size:
                        a = int(round(a * SHIFT_VAL * scale_factor))
                        b = int(round(b * SHIFT_VAL * scale_factor))
                        cv2.ellipse(
                            overlay,
                            (kpt_coord[0], kpt_coord[1]),
                            (a, b),
                            angle=math.degrees(theta),
                            startAngle=0,
                            endAngle=360,
                            color=col_inst,
                            thickness=width,
                            lineType=cv2.LINE_AA,
                            shift=SHIFT,
                        )
                except Exception as e:
                    print(f"Error drawing confidence ellipse for keypoint {kpt_coord} with variance {kp_var}: {e}")

        if connectivity:
            for connection in connectivity:
                if (not vis_inst[connection[0]]) or (not vis_inst[connection[1]]):
                    continue
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                cv2.line(
                    img_to_draw,
                    (start_pt_x, start_pt_y),
                    (end_pt_x, end_pt_y),
                    color=col_inst,
                    thickness=width,
                    lineType=cv2.LINE_AA,
                    shift=SHIFT,
                )

    # Blend the overlay with the original image
    img_to_draw = cv2.addWeighted(img_to_draw, 1.0, overlay, ELLIPSE_ALPHA, 0)

    out = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
    if original_dtype.is_floating_point:
        out = to_dtype(out, dtype=original_dtype, scale=True)  # type: ignore
    return out


def map_to_color(
    x: torch.Tensor, cmap: str = "inferno", norm_per_batch=True, use_quantiles=True
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies a colormap to a batch of spatial maps.

    Args:
        x: Tensor of shape (B, H, W) with arbitrary floating point range.
        cmap: Name of the matplotlib colormap to use (e.g. "inferno", "viridis").

    Returns:
        Tensor of shape (B, 3, H, W) with uint8 values in [0, 255].
        The color channels are RGB.
        And a tuple (min_value, max_value) used for normalization.
    """
    import matplotlib.pyplot as plt

    B, H, W = x.shape

    # Normalize per map to [0, 1]
    # Flatten spatial dims

    if norm_per_batch:
        x_flat = x.view(-1)
        if use_quantiles:
            min_v = torch.quantile(x_flat, 0.01)
            max_v = torch.quantile(x_flat, 0.99)
        else:
            min_v = x_flat.min()
            max_v = x_flat.max()
        res_min = min_v.cpu().expand(B)
        res_max = max_v.cpu().expand(B)
    else:
        x_flat = x.view(B, -1)
        if use_quantiles:
            min_v = torch.quantile(x_flat, 0.01, dim=1, keepdim=True).unsqueeze(-1)
            max_v = torch.quantile(x_flat, 0.99, dim=1, keepdim=True).unsqueeze(-1)
        else:
            min_v = x_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
            max_v = x_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
        res_min = min_v.cpu().squeeze(-1)
        res_max = max_v.cpu().squeeze(-1)
    x_norm = (x - min_v) / (max_v - min_v + 1e-8)

    # Map to indices [0, 2**24 - 1]
    n_bins = 2**24
    indices = (x_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)

    # Cache colormap
    cache_name = f"_cmap_{cmap}_{n_bins}"
    if not hasattr(map_to_color, cache_name):
        # Get colormap from matplotlib
        c = plt.get_cmap(cmap)
        # Create colors, discard alpha. Use float32 to save memory
        steps = np.linspace(0, 1, n_bins, dtype=np.float32)
        colors = c(steps)[:, :3]
        # Convert to tensor suitable for embedding
        cmap_tensor = torch.from_numpy(colors).float() * 255
        setattr(map_to_color, cache_name, cmap_tensor.byte())

    # Retrieve and move to device
    cmap_tensor: torch.Tensor = getattr(map_to_color, cache_name)
    if cmap_tensor.device != x.device:
        cmap_tensor = cmap_tensor.to(x.device)
        setattr(map_to_color, cache_name, cmap_tensor)

    # Apply colormap using embedding
    # indices: (B, H, W), cmap_tensor: (256, 3) -> output: (B, H, W, 3)
    rgb = torch.nn.functional.embedding(indices, cmap_tensor)

    # Permute to (B, 3, H, W)
    return rgb.permute(0, 3, 1, 2), (res_min, res_max)
