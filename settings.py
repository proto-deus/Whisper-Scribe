import json
import logging
from pathlib import Path
from config import SETTINGS_FILE, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


class SettingsManager:
    def __init__(self):
        self._settings = dict(DEFAULT_SETTINGS)
        self._malformed = False
        self._load()

    def _load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._settings.update(saved)
            except json.JSONDecodeError as e:
                logger.warning("settings.json is malformed (%s); using defaults.", e)
                self._malformed = True
            except OSError as e:
                logger.warning("Could not read settings.json (%s); using defaults.", e)

    def save(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self._settings.items() if k != "hf_token"}
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        return self._settings.get(key, default)

    def set(self, key, value):
        self._settings[key] = value

    def update(self, data: dict):
        self._settings.update(data)

    def reset(self):
        self._settings = dict(DEFAULT_SETTINGS)

    def as_dict(self):
        return dict(self._settings)

    @property
    def was_malformed(self) -> bool:
        return self._malformed
