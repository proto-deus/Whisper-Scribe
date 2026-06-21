import sys
import os

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

from config import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "torch").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "whisper").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "local").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
os.environ["WHISPER_CACHE_DIR"] = str(MODELS_DIR / "whisper")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from settings import SettingsManager
from core.secrets import get_hf_token, set_hf_token


def _migrate_hf_token_to_keyring():
    settings = SettingsManager()
    current = settings.get("hf_token", "")
    if current and not get_hf_token():
        set_hf_token(current)
        settings.set("hf_token", "")
        settings.save()
    elif current:
        settings.set("hf_token", "")
        settings.save()


def main():
    _migrate_hf_token_to_keyring()
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Scribe")
    app.setFont(QFont("Segoe UI", 10))
    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
