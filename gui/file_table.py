from pathlib import Path
from PyQt6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QCheckBox, QProgressBar,
    QWidget, QHBoxLayout, QHeaderView,
)
from PyQt6.QtCore import Qt, QElapsedTimer
from core.file_utils import format_duration, get_duration


class FileTable(QTableWidget):
    COL_CHECK = 0
    COL_NAME = 1
    COL_FORMAT = 2
    COL_LENGTH = 3
    COL_ELAPSED = 4
    COL_PROGRESS = 5
    COL_COUNT = 6

    HEADERS = ["", "File Name", "Format", "Length", "Elapsed", "Progress"]

    def __init__(self, parent=None):
        super().__init__(0, self.COL_COUNT, parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)

        header = self.horizontalHeader()
        header.setSectionResizeMode(self.COL_CHECK, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_FORMAT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_LENGTH, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_ELAPSED, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(self.COL_CHECK, 30)
        self.setColumnWidth(self.COL_ELAPSED, 80)
        self.setColumnWidth(self.COL_PROGRESS, 140)

        self._files: list[Path] = []
        self._checkboxes: list[QCheckBox] = []
        self._timers: list[QElapsedTimer] = []
        self._elapsed_items: list[QTableWidgetItem] = []
        self._progress_bars: list[QProgressBar] = []
        self._status: list[str] = []
        self._output_paths: list[Path | None] = []

    def add_files(self, paths: list[Path]):
        for p in paths:
            if p in self._files:
                continue
            self._files.append(p)
            row = self.rowCount()
            self.insertRow(row)

            cb = QCheckBox()
            cb.setChecked(True)
            self._checkboxes.append(cb)
            cb_widget = QWidget()
            layout = QHBoxLayout(cb_widget)
            layout.addWidget(cb)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.setCellWidget(row, self.COL_CHECK, cb_widget)

            self.setItem(row, self.COL_NAME, QTableWidgetItem(p.name))
            self.setItem(row, self.COL_FORMAT, QTableWidgetItem(p.suffix.upper().lstrip(".")))

            duration = get_duration(p)
            self.setItem(row, self.COL_LENGTH, QTableWidgetItem(format_duration(duration)))

            elapsed_item = QTableWidgetItem("--:--")
            elapsed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, self.COL_ELAPSED, elapsed_item)
            self._elapsed_items.append(elapsed_item)

            timer = QElapsedTimer()
            self._timers.append(timer)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setFixedHeight(16)
            self.setCellWidget(row, self.COL_PROGRESS, progress)
            self._progress_bars.append(progress)

            self._status.append("Pending")
            self._output_paths.append(None)

    def get_selected_files(self) -> list[Path]:
        selected = []
        for i, cb in enumerate(self._checkboxes):
            if cb.isChecked():
                selected.append(self._files[i])
        return selected

    def get_all_files(self) -> list[Path]:
        return list(self._files)

    def get_index_for_path(self, path: Path) -> int | None:
        try:
            return self._files.index(path)
        except ValueError:
            return None

    def get_current_row(self) -> int:
        return self.currentRow()

    def update_file_progress(self, index: int, value: float):
        widget = self.cellWidget(index, self.COL_PROGRESS)
        if isinstance(widget, QProgressBar):
            widget.setValue(int(value * 100))
        if 0 <= index < len(self._timers) and self._timers[index].isValid():
            elapsed_ms = self._timers[index].elapsed()
            self._elapsed_items[index].setText(_format_elapsed(elapsed_ms))

    def mark_file_started(self, index: int):
        if 0 <= index < len(self._timers):
            self._timers[index].restart()
        if 0 <= index < len(self._elapsed_items):
            self._elapsed_items[index].setText("00:00")
        if 0 <= index < len(self._progress_bars):
            self._progress_bars[index].setValue(0)
        self._set_status(index, "Processing...")

    def mark_file_status(self, index: int, status: str):
        self._set_status(index, status)
        name_item = self.item(index, self.COL_NAME)
        if name_item:
            name_item.setToolTip(status)

    def get_status(self, index: int) -> str:
        if 0 <= index < len(self._status):
            return self._status[index]
        return ""

    def set_output_path(self, index: int, path: Path | None):
        if 0 <= index < len(self._output_paths):
            self._output_paths[index] = path

    def get_output_path(self, index: int) -> Path | None:
        if 0 <= index < len(self._output_paths):
            return self._output_paths[index]
        return None

    def clear_all(self):
        self.setRowCount(0)
        self._files.clear()
        self._checkboxes.clear()
        self._timers.clear()
        self._elapsed_items.clear()
        self._progress_bars.clear()
        self._status.clear()
        self._output_paths.clear()

    def select_all(self, checked: bool):
        for cb in self._checkboxes:
            cb.setChecked(checked)

    def remove_unchecked(self):
        to_remove = []
        for i, cb in enumerate(self._checkboxes):
            if not cb.isChecked():
                to_remove.append(i)
        for i in reversed(to_remove):
            self.removeRow(i)
            del self._files[i]
            del self._checkboxes[i]
            del self._timers[i]
            del self._elapsed_items[i]
            del self._progress_bars[i]
            del self._status[i]
            del self._output_paths[i]

    def _set_status(self, index: int, status: str):
        if 0 <= index < len(self._status):
            self._status[index] = status


def _format_elapsed(ms: int) -> str:
    s = ms // 1000
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"
