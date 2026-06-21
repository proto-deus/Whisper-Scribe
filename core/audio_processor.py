import tempfile
import subprocess
import traceback
from pathlib import Path


def remove_background_music(audio_path: Path, output_path: Path) -> Path:
    try:
        import torch
        import torchaudio
        import soundfile as sf
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(name="htdemucs")
        model.to(device)
        model.eval()

        mix, sr = torchaudio.load(str(audio_path))

        if sr != model.samplerate:
            mix = torchaudio.functional.resample(mix, sr, model.samplerate)

        if mix.shape[0] == 1:
            mix = mix.repeat(2, 1)
        elif mix.shape[0] > 2:
            mix = mix[:2] 

        mix = mix.unsqueeze(0).to(device)

        with torch.no_grad():
            sources = apply_model(
                model, mix, shifts=0, num_workers=0, progress=False,
            )

        vocals_idx = model.sources.index("vocals")
        vocals = sources[0, vocals_idx].cpu().numpy()
        vocals = vocals.mean(axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), vocals, model.samplerate)
        return output_path
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Demucs background music removal failed: {e}\n\nTraceback:\n{tb}"
        ) from e


def run_silero_vad(audio_path: Path, threshold: float = 0.5) -> list[dict]:
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
    try:
        model = load_silero_vad(onnx=False)
    except Exception as e:
        raise RuntimeError(f"Silero VAD failed to load: {e}") from e
    wav = read_audio(str(audio_path), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav, model, threshold=threshold, sampling_rate=16000,
    )
    segments = []
    for ts in speech_timestamps:
        segments.append({
            "start": ts["start"] / 16000.0,
            "end": ts["end"] / 16000.0,
        })
    return segments


def run_diarization(audio_path: Path, hf_token: str) -> list[dict]:
    from pyannote.audio import Pipeline
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except OSError as e:
        msg = str(e)
        if "local_files_only" in msg or "outgoing traffic" in msg:
            raise RuntimeError(
                "Cannot download diarization model \u2014 no internet connection or "
                "HuggingFace Hub is unreachable. Connect to the internet first, "
                "or ensure the model is already cached locally."
            ) from e
        raise
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    diarization = pipeline(str(audio_path))
    segments = []
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
    except TypeError:
        for turn, _, _ in diarization.itertracks():
            label = diarization.argmax(turn) if hasattr(diarization, "argmax") else "speaker"
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": str(label),
            })
    return segments


def apply_vad_segments(audio_path: Path, segments: list[dict],
                       output_path: Path) -> Path:
    filter_parts = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        filter_parts.append(f"between(t,{start:.3f},{end:.3f})")
    filter_str = "+".join(filter_parts)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-af", f"aselect='{filter_str}',asetpts=N/SR/TB",
            "-ar", "16000", "-ac", "1",
            str(output_path),
        ],
        capture_output=True, timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg VAD failed: {stderr[:200]}")
    return output_path


def filter_voice_segments(audio_path: Path, segments: list[dict],
                          speakers: list[str] | None = None) -> list[dict]:
    if speakers is None:
        return segments
    return [s for s in segments if s.get("speaker") in speakers]
