import gc
import shutil
import tempfile
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from core.file_utils import is_video_file, extract_audio, get_output_path
from core.audio_processor import (
    remove_background_music, run_silero_vad, run_diarization,
    apply_vad_segments,
)
from core.transcriber import Transcriber, save_transcription


class _StopRequested(Exception):
    pass


class TranscriptionWorker(QThread):
    progress = pyqtSignal(int, float)
    file_started = pyqtSignal(int)
    file_completed = pyqtSignal(int, str)
    file_error = pyqtSignal(int, str)
    overall_progress = pyqtSignal(float)
    finished_all = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self, files: list[tuple[int, Path]], settings: dict, model_manager):
        super().__init__()
        self.files = files
        self.settings = settings
        self.model_manager = model_manager
        self._stop_requested = False
        self._active_key = None
        self._completed_outputs: dict[int, Path] = {}

    def request_stop(self):
        self._stop_requested = True

    def _check_stop(self):
        if self._stop_requested:
            raise _StopRequested()

    def get_output_path(self, index: int) -> Path | None:
        return self._completed_outputs.get(index)

    def run(self):
        transcriber = Transcriber(self.model_manager)
        total = len(self.files)
        stopped = False

        try:
            for worker_idx, (table_idx, file_path) in enumerate(self.files):
                if self._stop_requested:
                    stopped = True
                    for rem_idx in range(worker_idx, total):
                        self.file_completed.emit(self.files[rem_idx][0], "Skipped")
                    self.overall_progress.emit(1.0)
                    break

                self.file_started.emit(table_idx)
                try:
                    self._process_file(table_idx, file_path, transcriber)
                    self.file_completed.emit(table_idx, "Done")
                except _StopRequested:
                    self.file_error.emit(table_idx, "Stopped by user")
                    stopped = True
                    for rem_idx in range(worker_idx + 1, total):
                        self.file_completed.emit(self.files[rem_idx][0], "Skipped")
                    self.overall_progress.emit(1.0)
                    break
                except Exception as e:
                    self.file_error.emit(table_idx, str(e))
                    self.log_message.emit(f"  Error: {e}")

                self.overall_progress.emit((worker_idx + 1) / total)
        finally:
            self._release_active_model()
            if self.settings.get("unload_after_batch", True) and not self.model_manager.busy():
                before = len(self.model_manager.get_loaded_models())
                self.model_manager.unload_unused()
                after = len(self.model_manager.get_loaded_models())
                freed = before - after
                if freed > 0:
                    self.log_message.emit(
                        f"Auto-unloaded {freed} model(s) from memory."
                    )
                elif before > 0:
                    self.log_message.emit(
                        f"Auto-unload: {before} model(s) still in use, nothing to free."
                    )

        if stopped:
            self.log_message.emit("Processing stopped by user.")
        self.finished_all.emit()

    def _process_file(self, idx: int, file_path: Path, transcriber: Transcriber):
        self.log_message.emit(f"Processing: {file_path.name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = file_path
            vocal_path = None
            music_path = None

            if is_video_file(file_path):
                self._check_stop()
                self.log_message.emit(f"  Extracting audio from video...")
                tmp_audio = Path(tmp_dir) / f"{file_path.stem}.wav"
                audio_path = extract_audio(file_path, tmp_audio)

            if self.settings.get("background_music_removal"):
                self._check_stop()
                self.log_message.emit("  Removing background music...")
                tmp_vocal = Path(tmp_dir) / f"{file_path.stem}_vocals.wav"
                vocal_path, music_path = remove_background_music(audio_path, tmp_vocal)
                audio_path = vocal_path

            if self.settings.get("voice_detection_filter"):
                self._check_stop()
                self.log_message.emit("  Applying voice detection filter...")
                try:
                    segments_vad = run_silero_vad(audio_path)
                    if segments_vad:
                        filtered = Path(tmp_dir) / f"{file_path.stem}_filtered.wav"
                        audio_path = apply_vad_segments(audio_path, segments_vad, filtered)
                    del segments_vad
                except Exception as e:
                    self.log_message.emit(f"  Voice detection filter failed: {e}")
                    self.log_message.emit("  Continuing without VAD filtering...")

            def file_progress(p):
                self._check_stop()
                self.progress.emit(idx, p)

            self._check_stop()
            self.log_message.emit(
                f"  Transcribing with {self.settings['model_series']} - "
                f"{self.settings['model']}..."
            )
            segments = self._transcribe_with_release(
                transcriber, audio_path, file_progress,
            )

            if self.settings.get("extract_speakers") and self.settings.get("hf_token"):
                self._check_stop()
                self.log_message.emit("  Running speaker diarization...")
                try:
                    diar_segments = run_diarization(audio_path, self.settings["hf_token"])
                    segments = self._merge_diarization(segments, diar_segments)
                    del diar_segments
                except _StopRequested:
                    raise
                except Exception as e:
                    self.log_message.emit(f"  Diarization failed: {e}")

            self._check_stop()
            output_format = self.settings.get("output_format", "srt")
            output_path = get_output_path(
                file_path,
                self.settings.get("output_path", ""),
                output_format,
                self.settings.get("output_same_as_source", True),
            )

            save_transcription(segments, output_path, output_format,
                               self.settings.get("include_timestamps", True))
            self._completed_outputs[idx] = output_path
            self.log_message.emit(f"  Saved: {output_path}")
            del segments

            if self.settings.get("save_background_removal_tracks") and vocal_path is not None:
                dest_vocal = file_path.parent / f"{file_path.stem}_vocals.wav"
                shutil.copy2(str(vocal_path), str(dest_vocal))
                self.log_message.emit(f"  Saved vocals track: {dest_vocal}")
                if music_path is not None:
                    dest_music = file_path.parent / f"{file_path.stem}_music.wav"
                    shutil.copy2(str(music_path), str(dest_music))
                    self.log_message.emit(f"  Saved music track: {dest_music}")

        self.progress.emit(idx, 1.0)
        gc.collect()

    def _transcribe_with_release(self, transcriber, audio_path, file_progress):
        series = self.settings["model_series"]
        model_name = self.settings["model"]
        device = self.settings.get("device", "auto")
        compute_type = self.settings.get("compute_type", "float16")
        key = self.model_manager.acquire(series, model_name, device, compute_type)
        self._active_key = key
        try:
            return transcriber.transcribe(
                audio_path, self.settings, file_progress, key,
            )
        finally:
            self.model_manager.release(key)
            self._active_key = None

    def _release_active_model(self):
        if self._active_key is not None:
            self.model_manager.release(self._active_key)
            self._active_key = None

    @staticmethod
    def _merge_diarization(transcript_segments, diar_segments):
        merged = []
        for seg in transcript_segments:
            mid = (seg["start"] + seg["end"]) / 2
            speaker = "Unknown"
            for d in diar_segments:
                if d["start"] <= mid <= d["end"]:
                    speaker = d["speaker"]
                    break
            merged.append({**seg, "speaker": speaker})
        return merged
