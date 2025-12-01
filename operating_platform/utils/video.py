import json
import logging
import subprocess
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Literal
import re

import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


# =============================================================================
# NPU Encoder Resource Management
# =============================================================================
# Ascend 310B typically has 2-4 hardware encoding channels.
# We use a semaphore to limit concurrent NPU encoding and prevent channel exhaustion.

# Default NPU channel limit (Ascend 310B has ~2-4 channels)
NPU_ENCODER_CHANNELS = 2

# Semaphore to limit concurrent NPU encoding
_npu_semaphore: Optional[threading.Semaphore] = None
_npu_semaphore_lock = threading.Lock()


def _get_npu_semaphore() -> threading.Semaphore:
    """Get or create the NPU encoder semaphore (thread-safe lazy init)."""
    global _npu_semaphore
    if _npu_semaphore is None:
        with _npu_semaphore_lock:
            if _npu_semaphore is None:
                _npu_semaphore = threading.Semaphore(NPU_ENCODER_CHANNELS)
                logging.info(f"[VideoEncoder] Initialized NPU semaphore with {NPU_ENCODER_CHANNELS} channels")
    return _npu_semaphore


def set_npu_encoder_channels(num_channels: int) -> None:
    """
    Configure the number of NPU encoder channels available.

    Call this before any encoding operations if your hardware has
    a different number of channels than the default (2).

    Args:
        num_channels: Number of concurrent NPU encoding channels (typically 2-4)
    """
    global _npu_semaphore, NPU_ENCODER_CHANNELS
    with _npu_semaphore_lock:
        NPU_ENCODER_CHANNELS = num_channels
        _npu_semaphore = threading.Semaphore(num_channels)
        logging.info(f"[VideoEncoder] Set NPU encoder channels to {num_channels}")


def get_available_encoders():
    """
    获取当前系统中 ffmpeg 支持的视频编码器列表。
    """
    try:
        # 执行 ffmpeg -encoders 命令
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        output = result.stdout

        # # 调试：打印完整输出
        # print("ffmpeg output:")
        # print(output)

        encoders = set()
        in_encoders = False

        for line in output.splitlines():
            line = line.strip()

            # 检测到 Encoders: 行，开始解析编码器
            if line == "Encoders:":
                in_encoders = True
                continue

            if not in_encoders:
                continue

            # 匹配视频编码器行：V + 5个非空白字符 + 空格 + 编码器名称
            match = re.match(r"^\s*V\S{5}\s+(\S+)", line)
            if match:
                encoders.add(match.group(1))

        logging.info(f"[VideoEncoder] Available ffmpeg encoders: {encoders}")

        return encoders

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg not found or failed to execute. "
            "Please ensure ffmpeg is installed and available in your PATH."
        )
    
# 缓存编码器列表，避免重复调用 ffmpeg
_AVAILABLE_ENCODERS = None

def _ensure_encoders_loaded():
    global _AVAILABLE_ENCODERS
    if _AVAILABLE_ENCODERS is None:
        _AVAILABLE_ENCODERS = get_available_encoders()

def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def _build_ffmpeg_cmd(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str = "yuv420p",
    g: int | None = None,
    crf: int | None = None,
    fast_decode: int = 0,
    log_level: Optional[str] = "error",
    overwrite: bool = False,
    # libx264 specific parameters
    preset: str = "ultrafast",
    # Ascend encoder specific parameters
    device_id: int = 0,
    channel_id: int = 0,
    profile: int = 1,
    rc_mode: int = 0,
    max_bit_rate: int = 10000,
    movement_scene: int = 0,
    # Progress output
    show_progress: bool = False,
) -> list[str]:
    """Build ffmpeg command for video encoding."""
    ffmpeg_args = OrderedDict(
        [
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(imgs_dir / "frame_%06d.png")),
            ("-vcodec", vcodec),
            ("-pix_fmt", pix_fmt),
        ]
    )

    if vcodec == "h264_ascend":
        # Ascend encoder specific parameters
        ffmpeg_args["-device_id"] = str(device_id)
        ffmpeg_args["-channel_id"] = str(channel_id)
        ffmpeg_args["-profile"] = str(profile)
        ffmpeg_args["-rc_mode"] = str(rc_mode)
        ffmpeg_args["-max_bit_rate"] = str(max_bit_rate)
        ffmpeg_args["-movement_scene"] = str(movement_scene)
        ffmpeg_args["-frame_rate"] = str(fps)
        if g is not None:
            ffmpeg_args["-gop"] = str(g)
    elif vcodec == "libx264":
        # libx264 specific parameters for fast encoding
        ffmpeg_args["-preset"] = preset
        if g is not None:
            ffmpeg_args["-g"] = str(g)
        else:
            ffmpeg_args["-g"] = str(fps * 2)  # Default: keyframe every 2 seconds
        if crf is not None:
            ffmpeg_args["-crf"] = str(crf)
        else:
            ffmpeg_args["-crf"] = "23"  # Default CRF for libx264 (good quality/size balance)
        if fast_decode:
            ffmpeg_args["-tune"] = "fastdecode"
    else:
        # Other software encoders (libopenh264, etc.)
        if g is not None:
            ffmpeg_args["-g"] = str(g)
        if crf is not None:
            ffmpeg_args["-crf"] = str(crf)
        if fast_decode:
            key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
            value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
            ffmpeg_args[key] = value

    if log_level is not None and not show_progress:
        ffmpeg_args["-loglevel"] = str(log_level)
    elif show_progress:
        # Show progress info
        ffmpeg_args["-loglevel"] = "info"
        ffmpeg_args["-stats"] = ""

    ffmpeg_args_list = []
    for key, value in ffmpeg_args.items():
        ffmpeg_args_list.append(key)
        if value:  # Skip empty values (like -stats which has no value)
            ffmpeg_args_list.append(value)

    if overwrite:
        ffmpeg_args_list.append("-y")

    return ["ffmpeg"] + ffmpeg_args_list + [str(video_path)]


def _encode_with_cpu_fallback(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    pix_fmt: str,
    fast_decode: int,
) -> None:
    """
    Fallback to libx264 CPU encoder with progress output.

    Uses ultrafast preset for speed on ARM CPUs.
    """
    frame_count = len(list(imgs_dir.glob("frame_*.png")))
    logging.info(f"[VideoEncoder] Encoding {frame_count} frames with libx264 (preset=ultrafast)...")

    fallback_cmd = _build_ffmpeg_cmd(
        imgs_dir=imgs_dir,
        video_path=video_path,
        fps=fps,
        vcodec="libx264",
        pix_fmt=pix_fmt,
        g=None,  # Default: keyframe every 2 seconds
        crf=None,  # Default: 23 (balanced quality)
        fast_decode=fast_decode,
        log_level=None,
        overwrite=True,
        preset="ultrafast",
        show_progress=True,
    )

    logging.info(f"[VideoEncoder] Running: {' '.join(fallback_cmd)}")

    try:
        subprocess.run(fallback_cmd, check=True, stdin=subprocess.DEVNULL)
        logging.info(f"[VideoEncoder] CPU encoding successful: {video_path.name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"[VideoEncoder] CPU encoding failed with return code: {e.returncode}")
        raise


def _is_npu_channel_error(stderr: str) -> bool:
    """Check if the error is due to NPU channel exhaustion."""
    error_indicators = [
        "Failed to create venc channel",
        "Error initializing output stream",
        "venc channel",
        "resource busy",
    ]
    stderr_lower = stderr.lower() if stderr else ""
    return any(indicator.lower() in stderr_lower for indicator in error_indicators)


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: Literal["h264_ascend", "libopenh264", "libx264"] = "h264_ascend",
    pix_fmt: str = "yuv420p",
    g: int | None = None,
    crf: int | None = None,
    fast_decode: int = 0,
    log_level: Optional[str] = "error",
    overwrite: bool = False,
    # Ascend encoder specific parameters
    device_id: int = 0,
    channel_id: int = 0,
    profile: int = 1,  # 0: baseline, 1: main, 2: high
    rc_mode: int = 0,  # 0: CBR, 1: VBR
    max_bit_rate: int = 10000,  # kbps
    movement_scene: int = 0,  # 0: static, 1: movement
    # Retry and fallback parameters
    npu_retry_timeout: float = 30.0,  # Max time to wait for NPU channel (seconds)
    npu_retry_interval: float = 1.0,  # Initial retry interval (seconds)
    npu_retry_max_interval: float = 5.0,  # Max retry interval (seconds)
) -> None:
    """
    Encode image frames to video with NPU priority and automatic fallback.

    Encoding Strategy:
    1. Use semaphore to limit concurrent NPU encoding (prevents channel exhaustion)
    2. If NPU fails, retry with exponential backoff (wait for channel to be freed)
    3. After timeout, fallback to libx264 CPU encoder as last resort

    NPU encoding is much faster than CPU (~10-50x), so we prioritize waiting for
    NPU availability over immediately falling back to slow CPU encoding.

    Args:
        imgs_dir: Directory containing frame_XXXXXX.png images
        video_path: Output video file path
        fps: Frames per second
        vcodec: Video codec (h264_ascend, libx264, libopenh264)
        pix_fmt: Pixel format (default: yuv420p)
        g: Keyframe interval (None = encoder default)
        crf: Constant rate factor for quality (None = encoder default)
        fast_decode: Enable fast decode optimization
        log_level: FFmpeg log level
        overwrite: Overwrite existing output file
        device_id: Ascend device ID
        channel_id: Ascend channel ID
        profile: H.264 profile (0=baseline, 1=main, 2=high)
        rc_mode: Rate control mode (0=CBR, 1=VBR)
        max_bit_rate: Maximum bitrate in kbps
        movement_scene: Scene type (0=static, 1=movement)
        npu_retry_timeout: Maximum time to retry NPU encoding (seconds)
        npu_retry_interval: Initial retry interval (seconds)
        npu_retry_max_interval: Maximum retry interval (seconds)
    """
    _ensure_encoders_loaded()
    available_encoders = _AVAILABLE_ENCODERS

    # Determine encoder to use
    if vcodec not in available_encoders:
        supported_candidates = {"h264_ascend", "libopenh264", "libx264"} & set(available_encoders)
        if not supported_candidates:
            raise ValueError(
                "None of the supported encoders are available. "
                "Please ensure at least one of 'libopenh264' or 'libx264' is supported."
            )
        # Prefer h264_ascend > libx264 > libopenh264
        if "h264_ascend" in supported_candidates:
            selected_vcodec = "h264_ascend"
        elif "libx264" in supported_candidates:
            selected_vcodec = "libx264"
        else:
            selected_vcodec = "libopenh264"

        warnings.warn(
            f"vcodec '{vcodec}' not available. Automatically switched to '{selected_vcodec}'.",
            UserWarning
        )
        vcodec = selected_vcodec

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # For non-NPU encoders, encode directly without semaphore
    if vcodec != "h264_ascend":
        ffmpeg_cmd = _build_ffmpeg_cmd(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
            log_level=log_level,
            overwrite=overwrite,
        )
        logging.info(f"[VideoEncoder] Encoding video: {video_path.name} (encoder={vcodec})")
        subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL,
                      capture_output=True, text=True)
        if not video_path.exists():
            raise OSError(f"Video encoding failed. File not found: {video_path}")
        return

    # === NPU Encoding with Semaphore and Retry ===
    semaphore = _get_npu_semaphore()
    has_cpu_fallback = "libx264" in available_encoders

    # Build NPU ffmpeg command
    ffmpeg_cmd = _build_ffmpeg_cmd(
        imgs_dir=imgs_dir,
        video_path=video_path,
        fps=fps,
        vcodec="h264_ascend",
        pix_fmt=pix_fmt,
        g=g,
        crf=crf,
        fast_decode=fast_decode,
        log_level=log_level,
        overwrite=overwrite,
        device_id=device_id,
        channel_id=channel_id,
        profile=profile,
        rc_mode=rc_mode,
        max_bit_rate=max_bit_rate,
        movement_scene=movement_scene,
    )

    # Try to acquire semaphore (wait for NPU channel)
    logging.info(f"[VideoEncoder] Waiting for NPU channel: {video_path.name}")
    acquired = semaphore.acquire(timeout=npu_retry_timeout)

    if not acquired:
        # Semaphore timeout - all channels busy for too long
        logging.warning(
            f"[VideoEncoder] NPU channels busy for {npu_retry_timeout}s, "
            f"falling back to CPU: {video_path.name}"
        )
        if has_cpu_fallback:
            _encode_with_cpu_fallback(imgs_dir, video_path, fps, pix_fmt, fast_decode)
            return
        else:
            raise RuntimeError(
                f"NPU channels busy and no CPU fallback available. "
                f"Consider increasing npu_retry_timeout or reducing concurrent encoding."
            )

    # Semaphore acquired - try NPU encoding with retry on failure
    try:
        start_time = time.time()
        retry_interval = npu_retry_interval
        attempt = 0

        while True:
            attempt += 1
            elapsed = time.time() - start_time

            logging.info(
                f"[VideoEncoder] NPU encoding attempt {attempt}: {video_path.name}"
            )

            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                )
                # Success!
                logging.info(f"[VideoEncoder] NPU encoding successful: {video_path.name}")
                break

            except subprocess.CalledProcessError as e:
                stderr = str(e.stderr) if e.stderr else ""

                # Check if it's a channel exhaustion error (can retry)
                if _is_npu_channel_error(stderr):
                    remaining = npu_retry_timeout - elapsed

                    if remaining <= 0:
                        # Timeout reached
                        logging.warning(
                            f"[VideoEncoder] NPU retry timeout after {attempt} attempts, "
                            f"falling back to CPU: {video_path.name}"
                        )
                        if has_cpu_fallback:
                            _encode_with_cpu_fallback(
                                imgs_dir, video_path, fps, pix_fmt, fast_decode
                            )
                            return
                        else:
                            raise RuntimeError(
                                f"NPU encoding failed after {attempt} attempts "
                                f"and no CPU fallback available."
                            ) from e

                    # Wait and retry
                    wait_time = min(retry_interval, remaining)
                    logging.info(
                        f"[VideoEncoder] NPU channel busy, retrying in {wait_time:.1f}s "
                        f"({remaining:.1f}s remaining): {video_path.name}"
                    )
                    time.sleep(wait_time)

                    # Exponential backoff (capped)
                    retry_interval = min(retry_interval * 1.5, npu_retry_max_interval)

                else:
                    # Non-recoverable error
                    logging.error(f"[VideoEncoder] NPU encoding failed: {stderr}")
                    raise

    finally:
        # Always release semaphore
        semaphore.release()

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}. "
            f"Try running the command manually to debug: `{' '.join(ffmpeg_cmd)}`"
        )

@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }


def get_video_info(video_path: Path | str) -> dict:
    ffprobe_video_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,codec_name,nb_frames,duration,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream_info = info["streams"][0]

    # Calculate fps from r_frame_rate
    r_frame_rate = video_stream_info["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    fps = num / denom

    pixel_channels = get_video_pixel_channels(video_stream_info["pix_fmt"])

    video_info = {
        "video.fps": fps,
        "video.height": video_stream_info["height"],
        "video.width": video_stream_info["width"],
        "video.channels": pixel_channels,
        "video.codec": video_stream_info["codec_name"],
        "video.pix_fmt": video_stream_info["pix_fmt"],
        "video.is_depth_map": False,
        **get_audio_info(video_path),
    }

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
