import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QProgressBar, QLabel, QFileDialog, QPlainTextEdit,
    QSplitter, QSizePolicy, QApplication, QMessageBox, QStatusBar,
    QDialog,
)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QFont, QTextCursor, QDragEnterEvent, QDropEvent

from config import ALL_MEDIA_EXTENSIONS
from settings import SettingsManager
from core.file_utils import scan_files
from core.model_manager import ModelManager
from core.preflight import run_preflight
from core.secrets import get_hf_token
from gui.file_table import FileTable
from gui.settings_dialog import SettingsDialog
from gui.model_manager_dialog import ModelManagerDialog
from workers.transcription_worker import TranscriptionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Scribe")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        self.setAcceptDrops(True)

        self.settings_mgr = SettingsManager()
        self.model_manager = ModelManager()
        self.worker: TranscriptionWorker | None = None
        self._preview_cache: dict[int, str] = {}

        self._init_ui()
        self._apply_style()

        self.model_manager.load_started.connect(self._on_model_load_started)
        self.model_manager.load_finished.connect(self._on_model_load_finished)
        self.model_manager.unload_finished.connect(self._on_unload_finished)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status_bar)
        self._status_timer.start()

        self._run_startup_checks()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._create_top_bar())
        layout.addWidget(self._create_file_section(), stretch=1)
        layout.addWidget(self._create_controls())
        layout.addWidget(self._create_progress_section())

        self._create_status_bar()

    def _create_top_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a file or folder, or drag-and-drop here...")
        self.path_edit.setReadOnly(True)
        layout.addWidget(self.path_edit, stretch=1)

        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.clicked.connect(self._open_file)
        layout.addWidget(self.open_file_btn)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._open_folder)
        layout.addWidget(self.open_folder_btn)

        self.settings_btn = QPushButton("\u2699")
        self.settings_btn.setFixedSize(36, 36)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(self.settings_btn)

        self.models_btn = QPushButton("Models")
        self.models_btn.setToolTip("Model Management")
        self.models_btn.clicked.connect(self._open_model_manager)
        layout.addWidget(self.models_btn)

        return widget

    def _create_file_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.file_table = FileTable()
        self.file_table.itemSelectionChanged.connect(self._on_row_selection_changed)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Files:"))
        header_row.addStretch()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(lambda: self.file_table.select_all(True))
        header_row.addWidget(self.select_all_btn)
        self.deselect_btn = QPushButton("Deselect All")
        self.deselect_btn.clicked.connect(lambda: self.file_table.select_all(False))
        header_row.addWidget(self.deselect_btn)
        self.remove_btn = QPushButton("Remove Unchecked")
        self.remove_btn.clicked.connect(self.file_table.remove_unchecked)
        header_row.addWidget(self.remove_btn)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        header_row.addWidget(self.clear_btn)
        layout.addLayout(header_row)

        layout.addWidget(self.file_table, stretch=1)

        return widget

    def _create_controls(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._start_processing)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        layout.addWidget(self.stop_btn)

        return widget

    def _create_progress_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("Overall: %p%")
        layout.addWidget(self.overall_progress)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Log:"))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_output)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("Output Preview:"))
        self.preview_output = QPlainTextEdit()
        self.preview_output.setReadOnly(True)
        self.preview_output.setPlaceholderText(
            "Select a completed file from the table above to preview its transcription."
        )
        self.preview_output.setMaximumHeight(150)
        self.preview_output.setFont(QFont("Consolas", 9))
        preview_layout.addWidget(self.preview_output)

        bottom_splitter.addWidget(log_panel)
        bottom_splitter.addWidget(preview_panel)
        bottom_splitter.setSizes([500, 500])

        layout.addWidget(bottom_splitter)

        return widget

    def _create_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.model_status_label = QLabel("Model: \u2014")
        self.memory_label = QLabel("VRAM: 0 MB")
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setMaximumWidth(200)
        self.download_progress.setVisible(False)
        self.download_progress.setTextVisible(True)

        sb.addPermanentWidget(self.model_status_label)
        sb.addPermanentWidget(QLabel(" | "))
        sb.addPermanentWidget(self.memory_label)
        sb.addPermanentWidget(QLabel(" | "))
        sb.addPermanentWidget(self.download_progress)

    def _refresh_status_bar(self):
        loaded = self.model_manager.get_loaded_models()
        if loaded:
            series_name, model_name = loaded[0]
            extras = ""
            if len(loaded) > 1:
                extras = f" (+{len(loaded) - 1} more)"
            self.model_status_label.setText(
                f"Model: {series_name} - {model_name}{extras}"
            )
        else:
            self.model_status_label.setText("Model: \u2014")

        gpu_mb, cpu_mb = self.model_manager.get_memory_usage()
        if gpu_mb > 0:
            self.memory_label.setText(
                f"VRAM: {_fmt_mb(gpu_mb)}  RAM: {_fmt_mb(cpu_mb)}"
            )
        else:
            self.memory_label.setText(f"RAM: {_fmt_mb(cpu_mb)}")

    def _on_model_load_started(self, series: str, model_name: str):
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 0)
        self.download_progress.setFormat(f"Loading {model_name}...")
        self._log(f"Loading model: {series} - {model_name}...")

    def _on_model_load_finished(self, series: str, model_name: str):
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(100)
        self.download_progress.setFormat("Ready")
        self._refresh_status_bar()
        QTimer.singleShot(1500, lambda: self.download_progress.setVisible(False))

    def _on_unload_finished(self, series: str):
        self._refresh_status_bar()
        self._log(f"Unloaded {series} model(s) from memory.")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #569cd6;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 16px;
                color: #d4d4d4;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #569cd6;
            }
            QPushButton:pressed {
                background-color: #264f78;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #606060;
            }
            QTableWidget {
                background-color: #252525;
                gridline-color: #333333;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #264f78;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 4px;
                text-align: center;
                background-color: #252525;
                color: #d4d4d4;
            }
            QProgressBar::chunk {
                background-color: #569cd6;
                border-radius: 3px;
            }
            QPlainTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #d4d4d4;
                font-family: Consolas, monospace;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #404040;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #569cd6;
                border-color: #569cd6;
            }
            QGroupBox {
                border: 1px solid #404040;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #569cd6;
            }
            QScrollArea {
                border: none;
            }
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #d4d4d4;
            }
            QStatusBar {
                background-color: #252525;
                color: #d4d4d4;
            }
            QSplitter::handle {
                background-color: #404040;
            }
        """)

    def _run_startup_checks(self):
        if self.settings_mgr.was_malformed:
            QMessageBox.warning(
                self,
                "Settings Corrupted",
                "Your settings.json file was malformed and could not be read.\n\n"
                "Defaults have been restored. Your previous settings will be "
                "overwritten when you save new ones.",
            )
        needs = run_preflight(self.settings_mgr.as_dict())
        if needs:
            lines = "\n".join(f"  \u2022 {msg}" for msg in needs)
            QMessageBox.warning(
                self,
                "Missing Dependencies",
                f"The following required tools were not found:\n\n{lines}\n\n"
                f"Some features may not work. Install them and restart the app.",
            )

    def _open_file(self):
        ext_str = " ".join(f"*{e}" for e in sorted(ALL_MEDIA_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Media File", "",
            f"Media Files ({ext_str});;All Files (*)",
        )
        if path:
            files = scan_files(path)
            self._add_files_with_prompt(files, str(path))

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            files = scan_files(folder)
            self._add_files_with_prompt(files, folder)

    def _add_files_with_prompt(self, files: list[Path], display_path: str):
        if not files:
            self._log(f"No media files found at {display_path}")
            return
        if self.file_table.rowCount() > 0:
            reply = QMessageBox.question(
                self,
                "Add Files",
                f"Add {len(files)} new file(s) to the current list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.No:
                self.file_table.clear_all()
                self.path_edit.clear()
                self.overall_progress.setValue(0)
                self.log_output.clear()
                self.preview_output.clear()
                self._preview_cache.clear()
        self.path_edit.setText(display_path)
        self.file_table.add_files(files)
        self._log(f"Loaded {len(files)} file(s)")

    def _open_settings(self):
        dlg = SettingsDialog(self.settings_mgr.as_dict(), self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            new_settings = dlg.get_settings()
            self.settings_mgr.update(new_settings)
            self.settings_mgr.save()
            self._log("Settings saved")

    def _open_model_manager(self):
        dlg = ModelManagerDialog(self.model_manager, self)
        dlg.exec()

    def _start_processing(self):
        files = self.file_table.get_selected_files_with_indices()
        if not files:
            self._log("No files selected for processing")
            return

        settings = self.settings_mgr.as_dict()
        token = get_hf_token()
        settings["hf_token"] = token or ""
        self._log(f"Starting transcription of {len(files)} file(s)...")
        self._log(f"Model: {settings['model_series']} - {settings['model']}")
        self._log(f"Device: {settings['device']}, Compute: {settings['compute_type']}")

        self._preview_cache.clear()
        self.preview_output.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.overall_progress.setValue(0)

        self.worker = TranscriptionWorker(files, settings, self.model_manager)
        self.worker.progress.connect(self._on_file_progress)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_completed.connect(self._on_file_completed)
        self.worker.file_error.connect(self._on_file_error)
        self.worker.overall_progress.connect(self._on_overall_progress)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.log_message.connect(self._log)
        self.worker.start()

    def _stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.stop_btn.setEnabled(False)
            self.worker.request_stop()
            self._log("Stopping... please wait for the current operation to finish")

    def _on_file_progress(self, index: int, value: float):
        self.file_table.update_file_progress(index, value)

    def _on_file_started(self, index: int):
        self.file_table.mark_file_started(index)
        self._log(f"Started file {index + 1}")

    def _on_file_completed(self, index: int, status: str):
        self.file_table.mark_file_status(index, status)
        self.file_table.update_file_progress(index, 1.0)
        if status == "Done" and self.worker is not None:
            out_path = self.worker.get_output_path(index)
            if out_path is not None:
                self.file_table.set_output_path(index, out_path)

    def _on_file_error(self, index: int, error: str):
        self.file_table.mark_file_status(index, f"Error: {error}")
        self._log(f"Error on file {index + 1}: {error}")

    def _on_overall_progress(self, value: float):
        self.overall_progress.setValue(int(value * 100))

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log("Processing complete")
        self._refresh_status_bar()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _on_row_selection_changed(self):
        row = self.file_table.get_current_row()
        if row < 0:
            return
        if self.file_table.get_status(row) not in ("Done",):
            self.preview_output.clear()
            return
        if row in self._preview_cache:
            self.preview_output.setPlainText(self._preview_cache[row])
            return
        out_path = self.file_table.get_output_path(row)
        if out_path is None:
            return
        try:
            text = out_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            self._log(f"Could not read preview for row {row + 1}: {e}")
            return
        if len(text) > 200_000:
            text = text[:200_000] + "\n\n[... truncated for preview ...]"
        self._preview_cache[row] = text
        self.preview_output.setPlainText(text)

    def _clear_all(self):
        self.file_table.clear_all()
        self.path_edit.clear()
        self.overall_progress.setValue(0)
        self.log_output.clear()
        self.preview_output.clear()
        self._preview_cache.clear()

    def _log(self, message: str):
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = []
        for url in urls:
            local = url.toLocalFile()
            if local:
                paths.append(Path(local))
        if not paths:
            return
        all_files: list[Path] = []
        display_path = ""
        for p in paths:
            if p.is_dir():
                sub = scan_files(str(p))
                all_files.extend(sub)
                display_path = str(p)
            elif p.is_file():
                sub = scan_files(str(p))
                all_files.extend(sub)
                display_path = str(p)
        if not all_files:
            self._log("No media files in dropped items")
            return
        self._add_files_with_prompt(all_files, display_path)
        event.acceptProposedAction()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            if not self.worker.wait(5000):
                self.worker.terminate()
                self.worker.wait(2000)
        self.settings_mgr.save()
        super().closeEvent(event)


def _fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"
