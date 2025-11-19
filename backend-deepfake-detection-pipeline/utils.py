import tempfile
import os
import subprocess
import shutil
import numpy as np
import cv2
import torch
from fastapi import HTTPException
from typing import Optional

def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _guess_suffix_from_filename(filename: Optional[str]) -> str:
    if not filename:
        return ".mp4"
    _, ext = os.path.splitext(filename)
    return ext if ext else ".mp4"

def extract_frames(video_bytes: bytes, max_frames: int = 8, size: int = 224, filename: Optional[str] = None) -> torch.Tensor:
    """
    Robust frame extractor:
      - writes bytes to a temp file (with suffix from filename if present)
      - attempts cv2.VideoCapture
      - if VideoCapture fails, tries to transcode to mp4 using ffmpeg and tries again
      - returns tensor [N,3,size,size] dtype float32 with values in [0,1]

    Raises HTTPException with helpful details on failure.
    """
    # quick diagnostics
    length = len(video_bytes)
    prefix_hex = video_bytes[:12].hex()
    suffix = _guess_suffix_from_filename(filename)

    # 1) write original bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpf:
        tmpf.write(video_bytes)
        tmpf.flush()
        tmp_path = tmpf.name

    try:
        # 2) try to open with cv2
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            cap.release()
            # attempt ffmpeg transcode if available
            if _has_ffmpeg():
                # transcode to a new mp4 file with libx264 baseline profile (max compatibility)
                transcoded = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                transcoded_name = transcoded.name
                transcoded.close()
                # Build ffmpeg command: copy audio, re-encode video to baseline H264
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", tmp_path,
                    "-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "baseline",
                    "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-strict", "-2",
                    transcoded_name
                ]
                try:
                    subprocess.run(cmd, check=True, timeout=30)
                except subprocess.CalledProcessError as e:
                    # ffmpeg failed to transcode; raise helpful error
                    raise HTTPException(status_code=400, detail=(
                        "ffmpeg failed to transcode uploaded file. "
                        "Ensure ffmpeg is installed and the uploaded file is a valid video. "
                        f"file_len={length}, filename={filename}, prefix_hex={prefix_hex}"
                    ))
                except subprocess.TimeoutExpired:
                    raise HTTPException(status_code=500, detail="ffmpeg transcoding timed out.")
                # retry VideoCapture on transcoded file
                cap = cv2.VideoCapture(transcoded_name)
                if not cap.isOpened():
                    cap.release()
                    # cleanup transcoded file
                    try:
                        os.remove(transcoded_name)
                    except Exception:
                        pass
                    raise HTTPException(status_code=400, detail=(
                        "Unable to open transcoded video with OpenCV after ffmpeg. "
                        f"file_len={length}, filename={filename}, prefix_hex={prefix_hex}"
                    ))
                else:
                    # we will remember to remove transcoded_name later
                    cleanup_transcoded = transcoded_name
            else:
                # No ffmpeg installed — instruct user
                raise HTTPException(status_code=400, detail=(
                    "OpenCV cannot open uploaded file and ffmpeg is not available for transcoding. "
                    "Install ffmpeg or upload a standard mp4/h264 file. "
                    f"file_len={length}, filename={filename}, prefix_hex={prefix_hex}"
                ))
        else:
            cleanup_transcoded = None

        # 3) read frames (favor frame count if present)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        frames = []

        if frame_count and frame_count > 0:
            indices = np.linspace(0, frame_count - 1, num=min(max_frames, frame_count), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
        else:
            # read sequentially
            tmp_frames = []
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                tmp_frames.append(frame)
                if len(tmp_frames) > max_frames * 10:
                    break
            if len(tmp_frames) == 0:
                cap.release()
                raise HTTPException(status_code=400, detail=(
                    "No frames available after reading the video. file may be corrupted. "
                    f"file_len={length}, filename={filename}, prefix_hex={prefix_hex}"
                ))
            take = min(max_frames, len(tmp_frames))
            indices = np.linspace(0, len(tmp_frames) - 1, num=take, dtype=int)
            for idx in indices:
                f = tmp_frames[int(idx)]
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
                frames.append(f)

        cap.release()
        # cleanup transcoded if created
        if 'cleanup_transcoded' in locals() and cleanup_transcoded:
            try:
                os.remove(cleanup_transcoded)
            except Exception:
                pass

    finally:
        # remove the original temp file always
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if len(frames) == 0:
        raise HTTPException(status_code=400, detail=(
            "Unable to extract frames from the uploaded video after trying VideoCapture and ffmpeg. "
            f"file_len={length}, filename={filename}, prefix_hex={prefix_hex}"
        ))

    arr = np.stack(frames, axis=0).astype("float32") / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))  # [N,3,H,W]
    return torch.from_numpy(arr)
