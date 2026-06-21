from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLabel, QFileDialog, QDialogButtonBox, QScrollArea,
    QWidget, QMessageBox, QTextEdit,
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QWheelEvent
from config import (
    MODEL_SERIES, LANGUAGES, OUTPUT_FORMATS, DEVICES, COMPUTE_TYPES, DEFAULT_SETTINGS,
)
from core.model_manager import ModelManager
from core.secrets import get_hf_token, set_hf_token, delete_hf_token


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(750)
        self.settings = dict(settings)
        self.setStyleSheet("""
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 20px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        main_layout = QVBoxLayout(container)

        main_layout.addWidget(self._create_output_group())
        main_layout.addWidget(self._create_language_group())
        main_layout.addWidget(self._create_prompt_group())
        main_layout.addWidget(self._create_model_group())
        main_layout.addWidget(self._create_model_settings_group())
        main_layout.addWidget(self._create_processing_group())
        main_layout.addWidget(self._create_system_group())

        scroll.setWidget(container)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._reset_defaults
        )

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)
        outer.addWidget(buttons)

        for widget in self.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(event, QWheelEvent) and isinstance(
            obj, (QSpinBox, QDoubleSpinBox, QComboBox)
        ):
            if not obj.hasFocus():
                return True
        return super().eventFilter(obj, event)

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Output")
        layout = QVBoxLayout()

        path_row = QHBoxLayout()
        self.output_path_edit = QLineEdit(self.settings.get("output_path", ""))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_output)
        self.same_source_cb = QCheckBox("Same as source")
        self.same_source_cb.setChecked(self.settings.get("output_same_as_source", True))
        self.same_source_cb.toggled.connect(self._toggle_output_path)
        path_row.addWidget(QLabel("Path:"))
        path_row.addWidget(self.output_path_edit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)
        layout.addWidget(self.same_source_cb)
        self._toggle_output_path(self.same_source_cb.isChecked())

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(OUTPUT_FORMATS)
        idx = OUTPUT_FORMATS.index(self.settings.get("output_format", "srt"))
        self.format_combo.setCurrentIndex(idx)
        fmt_row.addWidget(self.format_combo)
        layout.addLayout(fmt_row)

        self.timestamps_cb = QCheckBox("Include timestamps in output")
        self.timestamps_cb.setChecked(self.settings.get("include_timestamps", True))
        layout.addWidget(self.timestamps_cb)

        group.setLayout(layout)
        return group

    def _create_language_group(self) -> QGroupBox:
        group = QGroupBox("Language")
        layout = QFormLayout()

        self.language_combo = QComboBox()
        for label, code in LANGUAGES:
            self.language_combo.addItem(label, code)
        lang_code = self.settings.get("language")
        for i, (_, code) in enumerate(LANGUAGES):
            if code == lang_code:
                self.language_combo.setCurrentIndex(i)
                break
        layout.addRow("Source Language:", self.language_combo)

        self.translate_combo = QComboBox()
        self.translate_combo.addItem("None", None)
        for label, code in LANGUAGES:
            if code is not None:
                self.translate_combo.addItem(label, code)
        trans_code = self.settings.get("translate_to")
        for i in range(self.translate_combo.count()):
            if self.translate_combo.itemData(i) == trans_code:
                self.translate_combo.setCurrentIndex(i)
                break
        layout.addRow("Translate to:", self.translate_combo)

        group.setLayout(layout)
        return group

    def _create_prompt_group(self) -> QGroupBox:
        group = QGroupBox("Initial Prompt")
        layout = QVBoxLayout()

        hint = QLabel(
            "Provide context to help Whisper recognize names, terms, or topics. "
            "Treat this as text that would appear just before the audio begins."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "e.g. The speaker's name is John Smith. They discuss Project Alpha and the Acme Corporation."
        )
        self.prompt_edit.setPlainText(self.settings.get("initial_prompt", ""))
        self.prompt_edit.setMaximumHeight(100)
        layout.addWidget(self.prompt_edit)

        group.setLayout(layout)
        return group

    def _create_model_group(self) -> QGroupBox:
        group = QGroupBox("Model Selection")
        layout = QFormLayout()

        self.series_combo = QComboBox()
        self.series_combo.addItems(ModelManager.get_all_series())
        series = self.settings.get("model_series", "Faster Whisper")
        self.series_combo.setCurrentText(series)
        self.series_combo.currentTextChanged.connect(self._on_series_changed)
        layout.addRow("Model Series:", self.series_combo)

        self.model_combo = QComboBox()
        self._update_model_list(series)
        self.model_combo.setCurrentText(self.settings.get("model", "large-v3"))
        layout.addRow("Model:", self.model_combo)

        group.setLayout(layout)
        return group

    def _create_model_settings_group(self) -> QGroupBox:
        group = QGroupBox("Model Settings")
        layout = QFormLayout()

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 20)
        self.beam_spin.setValue(self.settings.get("beam_size", 5))
        layout.addRow("Beam Size:", self.beam_spin)

        self.best_of_spin = QSpinBox()
        self.best_of_spin.setRange(1, 20)
        self.best_of_spin.setValue(self.settings.get("best_of", 5))
        layout.addRow("Best Of:", self.best_of_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.settings.get("temperature", 0.0))
        layout.addRow("Temperature:", self.temp_spin)

        self.temp_fallback_cb = QCheckBox("Temperature fallback (retry with higher temps on repetition)")
        self.temp_fallback_cb.setChecked(self.settings.get("temperature_fallback", False))
        layout.addRow(self.temp_fallback_cb)

        self.patience_spin = QDoubleSpinBox()
        self.patience_spin.setRange(0.0, 10.0)
        self.patience_spin.setSingleStep(0.1)
        self.patience_spin.setValue(self.settings.get("patience", 1.0))
        layout.addRow("Patience:", self.patience_spin)

        self.compression_spin = QDoubleSpinBox()
        self.compression_spin.setRange(0.0, 10.0)
        self.compression_spin.setSingleStep(0.1)
        self.compression_spin.setValue(self.settings.get("compression_ratio_threshold", 2.4))
        layout.addRow("Compression Threshold:", self.compression_spin)

        self.logprob_spin = QDoubleSpinBox()
        self.logprob_spin.setRange(-10.0, 0.0)
        self.logprob_spin.setSingleStep(0.1)
        self.logprob_spin.setValue(self.settings.get("log_prob_threshold", -1.0))
        layout.addRow("Log Prob Threshold:", self.logprob_spin)

        self.nospeech_spin = QDoubleSpinBox()
        self.nospeech_spin.setRange(0.0, 1.0)
        self.nospeech_spin.setSingleStep(0.05)
        self.nospeech_spin.setValue(self.settings.get("no_speech_threshold", 0.6))
        layout.addRow("No Speech Threshold:", self.nospeech_spin)

        self.condition_cb = QCheckBox("Condition on previous text")
        self.condition_cb.setChecked(self.settings.get("condition_on_previous_text", True))
        layout.addRow(self.condition_cb)

        self.vad_cb = QCheckBox("Enable VAD filter (Faster Whisper only)")
        self.vad_cb.setChecked(self.settings.get("vad_filter", True))
        layout.addRow(self.vad_cb)

        self.vad_thresh_spin = QDoubleSpinBox()
        self.vad_thresh_spin.setRange(0.0, 1.0)
        self.vad_thresh_spin.setSingleStep(0.05)
        self.vad_thresh_spin.setValue(self.settings.get("vad_threshold", 0.5))
        layout.addRow("VAD Threshold:", self.vad_thresh_spin)

        self.word_ts_cb = QCheckBox("Word timestamps")
        self.word_ts_cb.setChecked(self.settings.get("word_timestamps", True))
        layout.addRow(self.word_ts_cb)

        group.setLayout(layout)
        return group

    def _create_processing_group(self) -> QGroupBox:
        group = QGroupBox("Audio Processing")
        layout = QVBoxLayout()

        self.diarize_cb = QCheckBox("Extract speakers (pyannote diarization)")
        self.diarize_cb.setChecked(self.settings.get("extract_speakers", False))
        layout.addWidget(self.diarize_cb)

        self.demucs_cb = QCheckBox("Background music removal (demucs)")
        self.demucs_cb.setChecked(self.settings.get("background_music_removal", False))
        layout.addWidget(self.demucs_cb)

        self.save_tracks_cb = QCheckBox("Save separated audio tracks (vocals & music) next to source files")
        self.save_tracks_cb.setChecked(self.settings.get("save_background_removal_tracks", False))
        layout.addWidget(self.save_tracks_cb)

        self.voice_filter_cb = QCheckBox("Voice detection filter (Silero VAD)")
        self.voice_filter_cb.setChecked(self.settings.get("voice_detection_filter", False))
        layout.addWidget(self.voice_filter_cb)

        group.setLayout(layout)
        return group

    def _create_system_group(self) -> QGroupBox:
        group = QGroupBox("System")
        layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICES)
        device = self.settings.get("device", "auto")
        self.device_combo.setCurrentText(device)
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        layout.addRow("Device:", self.device_combo)

        self.compute_combo = QComboBox()
        device_key = device if device in COMPUTE_TYPES else "CUDA"
        self.compute_combo.addItems(COMPUTE_TYPES.get(device_key, COMPUTE_TYPES["CUDA"]))
        ct = self.settings.get("compute_type", "float16")
        self.compute_combo.setCurrentText(ct)
        layout.addRow("Compute Type:", self.compute_combo)

        self.unload_cb = QCheckBox("Unload model from memory after each batch completes")
        self.unload_cb.setChecked(self.settings.get("unload_after_batch", True))
        layout.addRow(self.unload_cb)

        self.hf_token_edit = QLineEdit(self.settings.get("hf_token", "") or (get_hf_token() or ""))
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText(
            "HuggingFace token (stored in OS keyring; required for diarization)"
        )
        layout.addRow("HF Token:", self.hf_token_edit)

        group.setLayout(layout)
        return group

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_path_edit.setText(d)

    def _toggle_output_path(self, same_source: bool):
        self.output_path_edit.setEnabled(not same_source)

    def _on_series_changed(self, series: str):
        self._update_model_list(series)

    def _update_model_list(self, series: str):
        self.model_combo.clear()
        models = ModelManager.get_models_for_series(series)
        self.model_combo.addItems(models)

    def _on_device_changed(self, device: str):
        self.compute_combo.clear()
        key = device if device in COMPUTE_TYPES else "CUDA"
        self.compute_combo.addItems(COMPUTE_TYPES.get(key, COMPUTE_TYPES["CUDA"]))

    def _reset_defaults(self):
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to their defaults? Your current choices will "
            "be lost when you save.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.settings = dict(DEFAULT_SETTINGS)
        self._reload_widgets()

    def _reload_widgets(self):
        self.output_path_edit.setText(self.settings.get("output_path", ""))
        self.same_source_cb.setChecked(self.settings.get("output_same_as_source", True))
        self._toggle_output_path(self.same_source_cb.isChecked())
        fmt = self.settings.get("output_format", "srt")
        if fmt in OUTPUT_FORMATS:
            self.format_combo.setCurrentIndex(OUTPUT_FORMATS.index(fmt))
        self.timestamps_cb.setChecked(self.settings.get("include_timestamps", True))

        lang_code = self.settings.get("language")
        for i, (_, code) in enumerate(LANGUAGES):
            if code == lang_code:
                self.language_combo.setCurrentIndex(i)
                break
        trans_code = self.settings.get("translate_to")
        for i in range(self.translate_combo.count()):
            if self.translate_combo.itemData(i) == trans_code:
                self.translate_combo.setCurrentIndex(i)
                break

        series = self.settings.get("model_series", "Faster Whisper")
        self.series_combo.setCurrentText(series)
        self._update_model_list(series)
        self.model_combo.setCurrentText(self.settings.get("model", "large-v3"))

        self.beam_spin.setValue(self.settings.get("beam_size", 5))
        self.best_of_spin.setValue(self.settings.get("best_of", 5))
        self.temp_spin.setValue(self.settings.get("temperature", 0.0))
        self.patience_spin.setValue(self.settings.get("patience", 1.0))
        self.compression_spin.setValue(self.settings.get("compression_ratio_threshold", 2.4))
        self.logprob_spin.setValue(self.settings.get("log_prob_threshold", -1.0))
        self.nospeech_spin.setValue(self.settings.get("no_speech_threshold", 0.6))
        self.condition_cb.setChecked(self.settings.get("condition_on_previous_text", True))
        self.temp_fallback_cb.setChecked(self.settings.get("temperature_fallback", False))
        self.vad_cb.setChecked(self.settings.get("vad_filter", True))
        self.vad_thresh_spin.setValue(self.settings.get("vad_threshold", 0.5))
        self.word_ts_cb.setChecked(self.settings.get("word_timestamps", True))

        self.prompt_edit.setPlainText(self.settings.get("initial_prompt", ""))

        self.diarize_cb.setChecked(self.settings.get("extract_speakers", False))
        self.demucs_cb.setChecked(self.settings.get("background_music_removal", False))
        self.save_tracks_cb.setChecked(self.settings.get("save_background_removal_tracks", False))
        self.voice_filter_cb.setChecked(self.settings.get("voice_detection_filter", False))

        device = self.settings.get("device", "auto")
        self.device_combo.setCurrentText(device)
        self._on_device_changed(device)
        ct = self.settings.get("compute_type", "float16")
        if ct in [self.compute_combo.itemText(i) for i in range(self.compute_combo.count())]:
            self.compute_combo.setCurrentText(ct)
        self.unload_cb.setChecked(self.settings.get("unload_after_batch", True))
        self.hf_token_edit.setText(self.settings.get("hf_token", "") or (get_hf_token() or ""))

    def _save_and_accept(self):
        self.settings["output_path"] = self.output_path_edit.text()
        self.settings["output_same_as_source"] = self.same_source_cb.isChecked()
        self.settings["output_format"] = self.format_combo.currentText()
        self.settings["language"] = self.language_combo.currentData()
        self.settings["translate_to"] = self.translate_combo.currentData()
        self.settings["model_series"] = self.series_combo.currentText()
        self.settings["model"] = self.model_combo.currentText()
        self.settings["beam_size"] = self.beam_spin.value()
        self.settings["best_of"] = self.best_of_spin.value()
        self.settings["temperature"] = self.temp_spin.value()
        self.settings["patience"] = self.patience_spin.value()
        self.settings["compression_ratio_threshold"] = self.compression_spin.value()
        self.settings["log_prob_threshold"] = self.logprob_spin.value()
        self.settings["no_speech_threshold"] = self.nospeech_spin.value()
        self.settings["condition_on_previous_text"] = self.condition_cb.isChecked()
        self.settings["temperature_fallback"] = self.temp_fallback_cb.isChecked()
        self.settings["vad_filter"] = self.vad_cb.isChecked()
        self.settings["vad_threshold"] = self.vad_thresh_spin.value()
        self.settings["word_timestamps"] = self.word_ts_cb.isChecked()
        self.settings["include_timestamps"] = self.timestamps_cb.isChecked()
        self.settings["extract_speakers"] = self.diarize_cb.isChecked()
        self.settings["background_music_removal"] = self.demucs_cb.isChecked()
        self.settings["save_background_removal_tracks"] = self.save_tracks_cb.isChecked()
        self.settings["voice_detection_filter"] = self.voice_filter_cb.isChecked()
        self.settings["initial_prompt"] = self.prompt_edit.toPlainText().strip()
        self.settings["device"] = self.device_combo.currentText()
        self.settings["compute_type"] = self.compute_combo.currentText()
        self.settings["unload_after_batch"] = self.unload_cb.isChecked()

        hf_token = self.hf_token_edit.text().strip()
        if hf_token:
            set_hf_token(hf_token)
            self.settings["hf_token"] = ""
        else:
            delete_hf_token()
            self.settings["hf_token"] = ""

        self.accept()

    def get_settings(self) -> dict:
        self.settings["hf_token"] = ""
        return dict(self.settings)
