import os
import subprocess
import json
from pathlib import Path
from config import ALL_MEDIA_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in ALL_MEDIA_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def scan_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        if is_media_file(p):
            return [p]
        return []
    if p.is_dir():
        files = []
        for ext in ALL_MEDIA_EXTENSIONS:
            files.extend(p.rglob(f"*{ext}"))
            files.extend(p.rglob(f"*{ext.upper()}"))
        return sorted(set(files), key=lambda f: f.name.lower())
    return []


def get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(path),
            ],
            capture_output=True, timeout=30,
        )
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        if result.returncode == 0 and stdout.strip():
            data = json.loads(stdout)
            duration = data.get("format", {}).get("duration")
            if duration:
                return float(duration)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, ValueError):
        pass
    return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_output_path(source_path: Path, output_dir: str, output_format: str,
                    same_as_source: bool) -> Path:
    if same_as_source:
        return source_path.parent / f"{source_path.stem}.{output_format}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{source_path.stem}.{output_format}"


def extract_audio(video_path: Path, output_path: Path) -> Path:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_path),
        ],
        capture_output=True, timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {stderr[:200]}")
    return output_path
