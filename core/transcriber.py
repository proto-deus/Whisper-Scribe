import os
from pathlib import Path
from config import MODEL_SERIES


class Transcriber:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    @staticmethod
    def _get_temperature(settings):
        temp = settings.get("temperature", 0.0)
        if settings.get("temperature_fallback", False):
            return (temp, 0.2, 0.4, 0.6, 0.8, 1.0)
        return temp

    @staticmethod
    def _get_initial_prompt(settings):
        prompt = settings.get("initial_prompt", "").strip()
        return prompt if prompt else None

    def transcribe(self, audio_path: Path, settings: dict,
                   progress_callback=None, preset_key=None) -> list[dict]:
        series = settings["model_series"]
        model_name = settings["model"]
        language = settings.get("language")
        device = settings.get("device", "auto")
        compute_type = settings.get("compute_type", "float16")

        if preset_key is not None:
            cached = self.model_manager.get_loaded(preset_key)
            if cached is not None:
                model = cached
            else:
                model = self.model_manager.load_model(
                    series, model_name, device, compute_type, language,
                )
        else:
            model = self.model_manager.load_model(
                series, model_name, device, compute_type, language,
            )

        if series == "Faster Whisper":
            return self._transcribe_faster_whisper(
                model, audio_path, settings, progress_callback,
            )
        elif series == "OpenAI Whisper":
            return self._transcribe_openai_whisper(
                model, audio_path, settings, progress_callback,
            )
        elif series == "Insanely Fast Whisper":
            return self._transcribe_insanely_fast_whisper(
                model, audio_path, settings, progress_callback,
            )
        return []

    def _transcribe_faster_whisper(self, model, audio_path, settings,
                                    progress_callback=None):
        language = settings.get("language")
        translate_to = settings.get("translate_to")
        task = "translate" if translate_to and translate_to != "None" else "transcribe"

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=settings.get("beam_size", 5),
            best_of=settings.get("best_of", 5),
            temperature=self._get_temperature(settings),
            patience=settings.get("patience", 1.0),
            compression_ratio_threshold=settings.get("compression_ratio_threshold", 2.4),
            log_prob_threshold=settings.get("log_prob_threshold", -1.0),
            no_speech_threshold=settings.get("no_speech_threshold", 0.6),
            condition_on_previous_text=settings.get("condition_on_previous_text", True),
            vad_filter=settings.get("vad_filter", True),
            vad_parameters={"threshold": settings.get("vad_threshold", 0.5)},
            word_timestamps=settings.get("word_timestamps", True),
            initial_prompt=self._get_initial_prompt(settings),
            prepend_punctuations=settings.get("prepend_punctuations", "\"'([{-"),
            append_punctuations=settings.get(
                "append_punctuations",
                '"\'.\u3002\uFF0C!\uFF01?\uFF1F:\uFF1A\u201D)\u005D}\u3001',
            ),
        )

        segments = []
        total_duration = info.duration if info.duration else 0
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            if progress_callback and total_duration > 0:
                progress_callback(min(seg.end / total_duration, 1.0))

        return segments

    def _transcribe_openai_whisper(self, model, audio_path, settings,
                                    progress_callback=None):
        import whisper
        options = {
            "beam_size": settings.get("beam_size", 5),
            "best_of": settings.get("best_of", 5),
            "temperature": self._get_temperature(settings),
            "patience": settings.get("patience", 1.0),
            "compression_ratio_threshold": settings.get("compression_ratio_threshold", 2.4),
            "logprob_threshold": settings.get("log_prob_threshold", -1.0),
            "no_speech_threshold": settings.get("no_speech_threshold", 0.6),
            "condition_on_previous_text": settings.get("condition_on_previous_text", True),
            "word_timestamps": settings.get("word_timestamps", True),
        }
        initial_prompt = self._get_initial_prompt(settings)
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
        language = settings.get("language")
        if language:
            options["language"] = language
        if settings.get("translate_to") and settings["translate_to"] != "None":
            options["task"] = "translate"

        result = model.transcribe(str(audio_path), **options)
        segments = []
        for seg in (result.get("segments") or []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })
        if progress_callback:
            progress_callback(1.0)
        return segments

    def _transcribe_insanely_fast_whisper(self, pipe, audio_path, settings,
                                           progress_callback=None):
        generate_kwargs = {}
        language = settings.get("language")
        if language:
            generate_kwargs["language"] = language
        if settings.get("translate_to") and settings["translate_to"] != "None":
            generate_kwargs["task"] = "translate"
        generate_kwargs["temperature"] = self._get_temperature(settings)
        initial_prompt = self._get_initial_prompt(settings)
        if initial_prompt:
            generate_kwargs["initial_prompt"] = initial_prompt

        import soundfile as sf
        audio_array, sr = sf.read(str(audio_path))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        audio_input = {"raw": audio_array, "sampling_rate": sr}

        result = pipe(
            audio_input,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        if isinstance(result, list):
            result = result[0] if result else {}

        segments = []
        chunks = result.get("chunks") or []
        for chunk in chunks:
            ts = chunk.get("timestamp")
            if not ts or len(ts) != 2:
                continue
            start, end = ts
            if start is not None and end is not None:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", "").strip(),
                })
        if progress_callback:
            progress_callback(1.0)
        return segments


def format_segments_srt(segments: list[dict], include_timestamps: bool = True) -> str:
    if not include_timestamps:
        return _format_text_only(segments)
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"]
        speaker = seg.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_segments_webvtt(segments: list[dict], include_timestamps: bool = True) -> str:
    if not include_timestamps:
        return _format_text_only(segments)
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments, 1):
        start = _format_vtt_time(seg["start"])
        end = _format_vtt_time(seg["end"])
        text = seg["text"]
        speaker = seg.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_segments_txt(segments: list[dict], include_timestamps: bool = True) -> str:
    if not include_timestamps:
        return _format_text_only(segments)
    lines = []
    for seg in segments:
        start = _format_txt_time(seg["start"])
        end = _format_txt_time(seg["end"])
        text = seg["text"]
        speaker = seg.get("speaker")
        if speaker:
            lines.append(f"[{start} - {end}] [{speaker}] {text}")
        else:
            lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def _format_text_only(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        text = seg["text"]
        speaker = seg.get("speaker")
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def format_segments(segments: list[dict], fmt: str,
                    include_timestamps: bool = True) -> str:
    if fmt == "srt":
        return format_segments_srt(segments, include_timestamps)
    elif fmt == "webvtt":
        return format_segments_webvtt(segments, include_timestamps)
    elif fmt == "txt":
        return format_segments_txt(segments, include_timestamps)
    return format_segments_srt(segments, include_timestamps)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_txt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def save_transcription(segments: list[dict], output_path: Path, fmt: str,
                       include_timestamps: bool = True):
    content = format_segments(segments, fmt, include_timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
