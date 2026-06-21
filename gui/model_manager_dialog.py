from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QMessageBox, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from core.model_manager import ModelManager


class ModelManagerDialog(QDialog):
    COL_SERIES = 0
    COL_NAME = 1
    COL_SIZE = 2
    COL_STATUS = 3
    COL_UNLOAD = 4
    COL_DELETE = 5
    COL_COUNT = 6

    HEADERS = ["Series", "Model", "Size", "Status", "", ""]

    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.setWindowTitle("Model Management")
        self.setMinimumSize(700, 450)
        self._init_ui()
        self._populate()

        self.model_manager.unload_finished.connect(lambda *_: self._populate())

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(self.COL_COUNT)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_SERIES, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_UNLOAD, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_DELETE, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self.COL_UNLOAD, 90)
        self.table.setColumnWidth(self.COL_DELETE, 90)

        layout.addWidget(self.table)

        bottom = QHBoxLayout()
        bottom.addStretch()

        self.unload_all_btn = QPushButton("Unload All From Memory")
        self.unload_all_btn.clicked.connect(self._unload_all)
        bottom.addWidget(self.unload_all_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)

        layout.addLayout(bottom)

    def _populate(self):
        loaded = self.model_manager.get_loaded_models()
        downloaded = self.model_manager.list_downloaded_models()
        busy = self.model_manager.busy()

        self.table.setRowCount(len(downloaded))

        for row, model in enumerate(downloaded):
            series_item = QTableWidgetItem(model["series"])
            self.table.setItem(row, self.COL_SERIES, series_item)

            name_item = QTableWidgetItem(model["name"])
            self.table.setItem(row, self.COL_NAME, name_item)

            size_text = f"{model['size_mb']:.0f} MB"
            size_item = QTableWidgetItem(size_text)
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, self.COL_SIZE, size_item)

            in_use = self.model_manager.is_in_use(model["series"], model["name"])
            is_loaded = any(
                s == model["series"] and n == model["name"]
                for s, n in loaded
            )
            if in_use:
                status = "In Use"
            elif is_loaded:
                status = "Loaded"
            else:
                status = "On Disk"
            status_item = QTableWidgetItem(status)
            self.table.setItem(row, self.COL_STATUS, status_item)

            if is_loaded and not in_use:
                unload_btn = QPushButton("Unload")
                unload_btn.setMinimumHeight(28)
                unload_btn.clicked.connect(
                    lambda checked, s=model["series"], n=model["name"]:
                        self._unload_one(s, n)
                )
                self.table.setCellWidget(row, self.COL_UNLOAD, unload_btn)
            elif in_use:
                placeholder = QLabel("In Use")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder.setStyleSheet("color: #888888;")
                placeholder.setToolTip(
                    "This model is currently being used by the transcription worker."
                )
                self.table.setCellWidget(row, self.COL_UNLOAD, placeholder)
            else:
                self.table.setCellWidget(row, self.COL_UNLOAD, QLabel(""))

            delete_btn = QPushButton("Delete")
            delete_btn.setMinimumHeight(28)
            delete_btn.clicked.connect(
                lambda checked, s=model["series"], n=model["name"], r=row: self._delete(s, n, r)
            )
            self.table.setCellWidget(row, self.COL_DELETE, delete_btn)

        self.unload_all_btn.setEnabled(not busy and bool(loaded))
        if busy:
            self.unload_all_btn.setToolTip(
                "Cannot unload all while a worker is using a model."
            )
        else:
            self.unload_all_btn.setToolTip("")

    def _unload_one(self, series: str, name: str):
        self.model_manager.unload_one(series, name)
        self._populate()

    def _unload_all(self):
        self.model_manager.unload_all()
        self._populate()

    def _delete(self, series: str, name: str, row: int):
        is_loaded = any(
            s == series and n == name
            for s, n in self.model_manager.get_loaded_models()
        )
        if is_loaded:
            QMessageBox.warning(
                self, "Model Loaded",
                "Cannot delete a model that is currently loaded in memory. "
                "Unload it first."
            )
            return

        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete {name} from disk?\n\nThis will free disk space but the "
            f"model will need to be re-downloaded to use it again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            ModelManager.delete_model(series, name)
            self._populate()
