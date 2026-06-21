import os
from pathlib import Path

_data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Whisper Scribe"
_models_dir = _data_dir / "Models"
(_models_dir / "huggingface").mkdir(parents=True, exist_ok=True)
(_models_dir / "torch").mkdir(parents=True, exist_ok=True)
(_models_dir / "whisper").mkdir(parents=True, exist_ok=True)
(_models_dir / "local").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(_models_dir / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_models_dir / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(_models_dir / "torch")
os.environ["WHISPER_CACHE_DIR"] = str(_models_dir / "whisper")
