import shutil
import subprocess
import sys


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_preflight(settings: dict | None = None) -> list[str]:
    settings = settings or {}
    missing: list[str] = []

    if not _have("ffmpeg"):
        missing.append(
            "ffmpeg is not on PATH \u2014 required for video-to-audio extraction "
            "and VAD filtering. Install from https://ffmpeg.org/."
        )
    if not _have("ffprobe"):
        missing.append(
            "ffprobe is not on PATH \u2014 required to read media duration. "
            "It ships with ffmpeg."
        )

    if settings.get("background_music_removal"):
        if getattr(sys, "frozen", False):
            try:
                import demucs
                _ = demucs.__version__
            except ImportError:
                missing.append(
                    "Demucs is enabled in settings but is not available in the "
                    "bundled application. This feature may require a custom build."
                )
        else:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "demucs", "--help"],
                    capture_output=True, timeout=10,
                )
                if result.returncode != 0:
                    missing.append(
                        "Demucs is enabled in settings but is not installed "
                        "(run: pip install demucs)."
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                missing.append(
                    "Demucs is enabled in settings but could not be invoked "
                    "(run: pip install demucs)."
                )

    return missing
