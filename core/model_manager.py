import os
import sys
import gc
import shutil
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal

from config import MODEL_SERIES, COMPUTE_TYPES, MODELS_DIR


def _wrap_hf_error(e: Exception, model_name: str) -> str:
    msg = str(e)
    if "local_files_only" in msg or "outgoing traffic" in msg:
        return (
            f"Cannot download model '{model_name}' \u2014 no internet connection or "
            f"HuggingFace Hub is unreachable. Either connect to the internet to "
            f"download the model first, or set HF_HUB_OFFLINE=1 and ensure the "
            f"model is already cached locally."
        )
    return f"Failed to load model '{model_name}': {e}"


class ModelManager(QObject):
    download_progress = pyqtSignal(str, float)
    load_started = pyqtSignal(str, str)
    load_finished = pyqtSignal(str, str)
    unload_finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._loaded_models = {}
        self._in_use = set()

    def acquire(self, series: str, model_name: str, device: str,
                compute_type: str):
        key = (series, model_name, device, compute_type)
        self._in_use.add(key)
        return key

    def release(self, key):
        self._in_use.discard(key)

    def busy(self) -> bool:
        return bool(self._in_use)

    def is_in_use(self, series: str, model_name: str) -> bool:
        return any(k[0] == series and k[1] == model_name for k in self._in_use)

    def load_model(self, series: str, model_name: str, device: str = "auto",
                   compute_type: str = "float16", language=None):
        key = (series, model_name, device, compute_type)
        if key in self._loaded_models:
            return self._loaded_models[key]

        self.load_started.emit(series, model_name)
        try:
            if series == "Faster Whisper":
                model = self._load_faster_whisper(model_name, device, compute_type)
            elif series == "OpenAI Whisper":
                model = self._load_openai_whisper(model_name, device)
            elif series == "Insanely Fast Whisper":
                model = self._load_insanely_fast_whisper(model_name, device, compute_type)
            else:
                raise ValueError(f"Unknown model series: {series}")
        except OSError as e:
            raise RuntimeError(_wrap_hf_error(e, model_name)) from e

        self._loaded_models[key] = model
        self.load_finished.emit(series, model_name)
        return model

    def _load_faster_whisper(self, model_name: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        if device == "cuda" and compute_type not in COMPUTE_TYPES["CUDA"]:
            compute_type = "float16"
        elif device == "cpu" and compute_type not in COMPUTE_TYPES["CPU"]:
            compute_type = "int8"
        local_dir = self._ensure_model_local(model_name)
        return WhisperModel(str(local_dir), device=device, compute_type=compute_type)

    def _load_openai_whisper(self, model_name: str, device: str):
        import whisper
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        return whisper.load_model(model_name, device=device)

    def _load_insanely_fast_whisper(self, model_name: str, device: str,
                                     compute_type: str):
        sys.modules["torchcodec"] = None
        sys.modules["torchcodec.decoders"] = None
        sys.modules["torchcodec._internally_replaced_utils"] = None

        from transformers import pipeline
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" and "16" in compute_type else torch.float32
        local_dir = self._ensure_model_local(model_name)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(local_dir),
            torch_dtype=dtype,
            device=device if device == "cuda" else -1,
            ignore_warning=True,
        )
        return pipe

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _local_models_root() -> Path:
        return MODELS_DIR / "local"

    @staticmethod
    def _resolve_faster_whisper_repo_id(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        try:
            from faster_whisper.utils import _MODELS as _FW_MODELS
            return _FW_MODELS.get(model_name, model_name)
        except Exception:
            return model_name

    @classmethod
    def _local_model_dir(cls, repo_id: str) -> Path:
        return cls._local_models_root() / repo_id.replace("/", "--")

    @classmethod
    def _ensure_model_local(cls, model_name: str) -> Path:
        from huggingface_hub import snapshot_download
        repo_id = cls._resolve_faster_whisper_repo_id(model_name)
        local_dir = cls._local_model_dir(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        return local_dir

    def unload_all(self):
        for k in list(self._loaded_models.keys()):
            if k not in self._in_use:
                self._unload_key(k)

    def unload_series(self, series: str):
        for k in list(self._loaded_models.keys()):
            if k[0] == series and k not in self._in_use:
                self._unload_key(k)

    def unload_unused(self):
        for k in list(self._loaded_models.keys()):
            if k not in self._in_use:
                self._unload_key(k)

    def unload_one(self, series: str, model_name: str):
        for k in list(self._loaded_models.keys()):
            if k[0] == series and k[1] == model_name and k not in self._in_use:
                self._unload_key(k)

    def _unload_key(self, key):
        model = self._loaded_models.pop(key, None)
        if model is None:
            return
        try:
            self._release_model_resources(model)
        except Exception:
            pass
        del model
        gc.collect()
        self._empty_cuda_cache()
        self.unload_finished.emit(key[0])

    @staticmethod
    def _release_model_resources(model):
        for attr in ("model", "feature_extractor", "tokenizer", "processor"):
            obj = getattr(model, attr, None)
            if obj is not None:
                try:
                    obj.to("cpu")
                except Exception:
                    pass

    @staticmethod
    def _empty_cuda_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def get_memory_usage(self) -> tuple[float, float]:
        gpu_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu_mb = (total - free) / (1024 * 1024)
        except Exception:
            pass
        cpu_mb = 0.0
        try:
            import psutil
            cpu_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        return gpu_mb, cpu_mb

    @staticmethod
    def get_models_for_series(series: str) -> list[str]:
        return MODEL_SERIES.get(series, {}).get("models", [])

    @staticmethod
    def get_all_series() -> list[str]:
        return list(MODEL_SERIES.keys())

    def get_loaded_models(self) -> list[tuple[str, str]]:
        return [(k[0], k[1]) for k in self._loaded_models]

    def get_loaded(self, key):
        return self._loaded_models.get(key)

    @staticmethod
    def _classify_hf_model(org: str, name: str) -> tuple[str, str] | None:
        if org in ("Systran", "deepdml"):
            series = "Faster Whisper"
        elif org in ("openai", "distil-whisper"):
            series = "Insanely Fast Whisper"
        else:
            return None

        short_name = name
        full_name = f"{org}/{name}"

        for model_id in MODEL_SERIES.get(series, {}).get("models", []):
            if model_id == full_name or model_id == short_name:
                return (series, model_id)

        candidates = sorted(
            MODEL_SERIES.get(series, {}).get("models", []),
            key=len, reverse=True,
        )
        for model_id in candidates:
            idx = short_name.find(model_id)
            if idx >= 0:
                before_ok = idx == 0 or short_name[idx - 1] == "-"
                after_end = idx + len(model_id)
                after_ok = after_end >= len(short_name) or short_name[after_end] == "-"
                if before_ok and after_ok:
                    return (series, model_id)

        return None

    @staticmethod
    def list_downloaded_models() -> list[dict]:
        results = []
        local_root = ModelManager._local_models_root()
        whisper_cache = MODELS_DIR / "whisper"

        for model_name in MODEL_SERIES.get("OpenAI Whisper", {}).get("models", []):
            pt_file = whisper_cache / f"{model_name}.pt"
            if pt_file.exists():
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                results.append({
                    "series": "OpenAI Whisper",
                    "name": model_name,
                    "size_mb": size_mb,
                })

        if local_root.exists():
            for d in local_root.iterdir():
                if not d.is_dir():
                    continue
                parts = d.name.split("--", 1)
                if len(parts) != 2:
                    continue
                org, name = parts
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)

                classified = ModelManager._classify_hf_model(org, name)
                if classified:
                    series, matched_name = classified
                    results.append({
                        "series": series,
                        "name": matched_name,
                        "size_mb": size_mb,
                    })

        return results

    @staticmethod
    def delete_model(series: str, model_name: str):
        whisper_cache = MODELS_DIR / "whisper"
        local_root = ModelManager._local_models_root()

        if series == "OpenAI Whisper":
            pt_file = whisper_cache / f"{model_name}.pt"
            if pt_file.exists():
                pt_file.unlink()
            return

        candidates = set()
        if "/" in model_name:
            candidates.add(model_name.replace("/", "--"))
        else:
            resolved = model_name
            try:
                from faster_whisper.utils import _MODELS as _FW_MODELS
                resolved = _FW_MODELS.get(model_name, model_name)
            except Exception:
                pass
            candidates.add(resolved.replace("/", "--"))

        for entry in local_root.iterdir() if local_root.exists() else []:
            if not entry.is_dir():
                continue
            if entry.name in candidates:
                shutil.rmtree(entry)
