from typing import Optional, Any, Iterator
from pathlib import Path
import hashlib
import os
from . import determine_cache_dir
from tqdm import tqdm
import itertools
import cv2
import numpy as np
from dataclasses import dataclass

def create_video_from_images_cached(
        images: list[str | Path] | Iterator[str | Any],
        fps=30,
        filename=None,
        resolution: None | tuple[int, int] = None,
        force=False,
        digest: Optional[str] = None,
        nframes: Optional[int] = None,
        codec="h264") -> tuple[str, tuple[int, int]]:
    """
    Create an MP4 video from a list of image paths or iterator of images or paths, and cache it.

    Args:
        images: List of paths to image files, or an iterator that yields image paths or BGR24 images.
        fps: Frames per second for the video
        filename: Optional filename (without extension). If None, uses SHA1 hash of all image paths.
        resolution: Optional (width, height) tuple to resize images. If None, uses the resolution of the first image.
        force: If True, force re-creation of the video even if cached version exists.
        digest: Optional SHA1 digest to use for caching. If None, a digest is computed from image paths and parameters.
        nframes: Optional number of frames to process from the iterator. If None, processes all frames.
        codec: Codec to use for video encoding. Default is "h264".

    Returns:
        video_path: Path to the created (or cached) video file
        resolution: (width, height) of the video
    """
    import ffmpeg
    assert (images is not None)
    assert (fps > 0)
    assert (nframes is None or nframes > 0)

    def img_id(img_path) -> str:
        try:
            mtime = str(os.path.getmtime(img_path))
        except:
            mtime = ""
        return str(img_path) + mtime

    def handle_img(img, width=None, height=None):
        if isinstance(img, str) or isinstance(img, Path):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
        if width is not None and height is not None and ((img.shape[1] != width or img.shape[0] != height)):
            img = cv2.resize(img, (width, height))
        return img

    # Create cache directory for videos
    cache_videos_dir = Path(determine_cache_dir()) / "videos"
    cache_videos_dir.mkdir(parents=True, exist_ok=True)

    # Create SHA1 hash for knowing whether we need to recreate the video
    hash_obj = hashlib.sha1()

    first_img = None
    if digest is not None:
        hash_obj.update(digest.encode('utf-8'))
        if filename is None:
            filename = hash_obj.hexdigest()
    if isinstance(images, list):
        images = sorted(images)
        assert (len(images) > 0)
        assert (isinstance(images[0], str) or isinstance(images[0], Path))
        if filename is None:
            for img_path in images:
                hash_obj.update(img_id(img_path).encode('utf-8'))
            filename = hash_obj.hexdigest()
        first_img = images.pop(0)
    if filename is None:
        assert isinstance(images, Iterator)
        first_img = next(images)
        assert (first_img is not None)
        if isinstance(first_img, str) or isinstance(first_img, Path):
            images = list(images)

            if filename is None:
                for img_path in [first_img] + images:
                    hash_obj.update(img_id(img_path).encode('utf-8'))
                filename = hash_obj.hexdigest()
        else:
            if filename is None:
                raise ValueError(
                    "When providing an iterator not over image paths, the filename parameter must be set to a non-None value, or digest must be provided.")

    # Full path to cached video
    video_path = cache_videos_dir / f"{filename}.mp4"

    if resolution is None:
        if first_img is None:
            width = None
            height = None
        else:
            first_img = handle_img(first_img)
            height, width = first_img.shape[:2]
    else:
        width, height = resolution

    # Create hash for parameters to know whether we need to recreate the video
    hash_obj.update(str(fps).encode('utf-8'))
    hash_obj.update(str(width).encode('utf-8'))
    hash_obj.update(str(height).encode('utf-8'))
    hash_obj.update(str(nframes).encode('utf-8'))
    digest = hash_obj.hexdigest()

    try:
        file_digest = os.getxattr(video_path, 'user.hash').decode('utf-8')
    except:
        file_digest = ""

    # Check if video already exists in cache
    if not force and video_path.exists() and file_digest == digest:
        print(f"Video already cached at: {video_path}")
        print("Loading from cache...")

        # Get the resolution from the existing video
        if resolution is None:
            try:
                probe = ffmpeg.probe(str(video_path))
                video_streams = [
                    stream for stream in probe['streams'] if stream['codec_type'] == 'video']
                if not video_streams:
                    raise ValueError(f"No video streams found in {video_path}")
                video_stream = video_streams[0]
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                resolution = (width, height)
            except Exception as e:
                print(
                    f"Could not determine video resolution from file. Error: {e}")
                width = None
                height = None
        if resolution is not None and resolution[0] == width and resolution[1] == height:
            return str(video_path), resolution

    if video_path.exists():
        os.remove(video_path)

    if first_img is None:
        assert isinstance(images, Iterator)
        first_img = handle_img(next(images), width, height)
        if height is None or width is None:
            height, width = first_img.shape[:2]

    # Define the codec and create VideoWriter object
    try:
        n = nframes if nframes is not None else len(images) + 1  # type: ignore
        print(f"Creating video with {n} frames...")
    except:
        print(f"Creating video...")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Cache path: {video_path}")

    # Process images with progress bar
    # Create input stream from raw video data piped in
    input_stream = ffmpeg.input(
        'pipe:',
        format='rawvideo',
        pix_fmt='bgr24',
        s=f'{width}x{height}',
        r=fps
    )

    # Should have been handled above.
    assert not isinstance(first_img, str) and not isinstance(first_img, Path)
    assert isinstance(width, int) and isinstance(height, int)

    # Create output stream with H.264 codec
    output_stream = ffmpeg.output(
        input_stream,
        str(video_path),
        vcodec=codec,
        pix_fmt='yuv420p',
        crf=23,
        preset='medium'
    )

    # Run ffmpeg process
    process = ffmpeg.run_async(output_stream, pipe_stdin=True, quiet=True)
    assert process.stdin is not None
    process.stdin.write(first_img.tobytes())
    # Feed frames to ffmpeg
    for img_maybe_path in tqdm(itertools.islice(images, nframes), desc="Processing frames", initial=1, total=None if nframes is None else nframes):
        img = handle_img(img_maybe_path, width, height)
        process.stdin.write(img.tobytes())

    # Close stdin and wait for process to complete
    process.stdin.close()
    process.wait()

    os.setxattr(video_path, 'user.hash', digest.encode('utf-8'))

    # Return the path to the created video and its resolution
    print("Video created!")
    return str(video_path), (width, height)

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    nframes: int
    frame_start: int
    duration: float
    
def probe_video(video_path: str | Path) -> VideoInfo:
    """
    Probe video file to get its metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        Metadata about the video.
    """
    import ffmpeg
    import numpy as np

    try:
        probe = ffmpeg.probe(str(video_path))
        video_streams = [
            stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if not video_streams:
            raise ValueError(f"No video streams found in {video_path}")
        duration = float(probe['format']['duration'])
        video_stream = video_streams[0]
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps_str = video_stream['r_frame_rate']
        fps_str_parts = fps_str.split('/')
        assert (len(fps_str_parts) == 2), f"Could not parse fps string: {fps_str}"
        fps = float(fps_str_parts[0]) / float(fps_str_parts[1])
        nframes = int(video_stream.get('nb_frames', None))
        if nframes is None:
            nframes = int(duration * fps)
        else:
            assert (nframes > 0), f"Video has non-positive number of frames: {nframes}"
        
        if video_stream.get("side_data_list", None) is not None:
            for side_data in video_stream["side_data_list"]:
                if side_data.get("rotation", None) is not None:
                    rotation = int(side_data["rotation"])
                    if rotation == 90 or rotation == 270:
                        width, height = height, width
                    break 
    except Exception as e:
        raise ValueError(f"Failed to read video '{str(video_path)}' metadata. Error: {e}")
    
    return VideoInfo(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frame_start=0,
        nframes=nframes,
    )
    

def decode_video_frames(video: VideoInfo | str | Path, start: int = 0, nframes: Optional[int] = None, progress = False) -> tuple[Iterator[np.typing.NDArray[np.uint8]], VideoInfo]:
    """
    Decode video file into frames.

    Args:
        video: VideoInfo object containing metadata about the video, or path to the video file.

    Yields:
        Frames as numpy arrays in BGR format.
    """
    import ffmpeg
    import numpy as np
    import datetime
    
    if isinstance(video, str) or isinstance(video, Path):
        video = probe_video(video)

    video.nframes = video.nframes - start if nframes is None else min(video.nframes - start, nframes)
    assert (video.nframes > 0), f"Video has no frames to decode (start={start}, nframes={nframes})"
    video.frame_start = start
    
    def get_frames(info: VideoInfo) -> Iterator[np.typing.NDArray[np.uint8]]:
        input_args = {}
        output_args = {}
        if start > 0:
            input_args["ss"] = str(datetime.timedelta(seconds=info.frame_start / info.fps))
        if nframes is not None:
            output_args["frames:v"] = nframes
        process = (
            ffmpeg
            .input(info.path, **input_args)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', **output_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        frame_size = info.width * info.height * 3  # 3 bytes per pixel for bgr24

        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([info.height, info.width, 3])
            )
            yield frame

        process.stdout.close()
        process.wait()
    if progress:
        return tqdm(get_frames(video), total=nframes, desc="Decoding frames"), video  # type: ignore
    else:
        return get_frames(video), video