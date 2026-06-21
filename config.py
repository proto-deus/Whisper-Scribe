import os
from pathlib import Path

APP_NAME = "Whisper Scribe"
APP_VERSION = "1.0.0"

APP_DIR = Path(__file__).parent

_data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Whisper Scribe"
CONFIG_DIR = _data_dir
MODELS_DIR = _data_dir / "Models"

SETTINGS_FILE = CONFIG_DIR / "settings.json"

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma",
    ".opus", ".aiff", ".aif", ".ape", ".alac", ".mka", ".ra",
    ".tta", ".wv", ".tak", ".dsf", ".dff",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    ".m4v", ".ts", ".mts", ".vob", ".3gp", ".ogv", ".rm",
    ".rmvb", ".asf", ".divx",
}

ALL_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

LANGUAGES = [
    ("Auto Detect", None),
    ("Afrikaans", "af"), ("Albanian", "sq"), ("Amharic", "am"), ("Arabic", "ar"),
    ("Armenian", "hy"), ("Assamese", "as"), ("Azerbaijani", "az"), ("Bashkir", "ba"),
    ("Basque", "eu"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"),
    ("Breton", "br"), ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"),
    ("Chinese", "zh"), ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"),
    ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"), ("Faroese", "fo"),
    ("Finnish", "fi"), ("French", "fr"), ("Galician", "gl"), ("Georgian", "ka"),
    ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), ("Haitian", "ht"),
    ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"),
    ("Icelandic", "is"), ("Indonesian", "id"), ("Italian", "it"), ("Japanese", "ja"),
    ("Javanese", "jv"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"),
    ("Korean", "ko"), ("Lao", "lo"), ("Latin", "la"), ("Latvian", "lv"),
    ("Lithuanian", "lt"), ("Luxembourgish", "lb"), ("Macedonian", "mk"),
    ("Malagasy", "mg"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"),
    ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"),
    ("Norwegian", "no"), ("Occitan", "oc"), ("Pashto", "ps"), ("Persian", "fa"),
    ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"),
    ("Russian", "ru"), ("Sanskrit", "sa"), ("Serbian", "sr"), ("Sinhala", "si"),
    ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Spanish", "es"),
    ("Sundanese", "su"), ("Swahili", "sw"), ("Swedish", "sv"), ("Tagalog", "tl"),
    ("Tajik", "tg"), ("Tamil", "ta"), ("Tatar", "tt"), ("Telugu", "te"),
    ("Thai", "th"), ("Tibetan", "bo"), ("Turkish", "tr"), ("Turkmen", "tk"),
    ("Ukrainian", "uk"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"),
    ("Welsh", "cy"), ("Yiddish", "yi"), ("Yoruba", "yo"),
]

OUTPUT_FORMATS = ["txt", "srt", "webvtt"]

MODEL_SERIES = {
    "OpenAI Whisper": {
        "models": [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v2", "large-v3", "turbo",
        ],
        "library": "openai-whisper",
    },
    "Faster Whisper": {
        "models": [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3",
            "distil-small.en", "distil-medium.en", "distil-large-v2",
            "distil-large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2",
            "deepdml/faster-distil-whisper-large-v3-turbo",
        ],
        "library": "faster-whisper",
    },
    "Insanely Fast Whisper": {
        "models": [
            "openai/whisper-tiny", "openai/whisper-tiny.en",
            "openai/whisper-base", "openai/whisper-base.en",
            "openai/whisper-small", "openai/whisper-small.en",
            "openai/whisper-medium", "openai/whisper-medium.en",
            "openai/whisper-large-v2", "openai/whisper-large-v3",
            "openai/whisper-large-v3-turbo",
            "distil-whisper/distil-large-v2", "distil-whisper/distil-large-v3",
            "distil-whisper/distil-medium.en", "distil-whisper/distil-small.en",
        ],
        "library": "insanely-fast-whisper",
    },
}

COMPUTE_TYPES = {
    "CUDA": ["float16", "float32", "int8_float16", "int8"],
    "CPU": ["int8", "float32"],
}

DEVICES = ["auto", "cuda", "cpu"]

DEFAULT_SETTINGS = {
    "output_path": "",
    "output_same_as_source": True,
    "output_format": "srt",
    "language": None,
    "translate_to": None,
    "model_series": "Faster Whisper",
    "model": "large-v3",
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,
    "patience": 1.0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "vad_filter": True,
    "vad_threshold": 0.5,
    "extract_speakers": False,
    "background_music_removal": False,
    "voice_detection_filter": False,
    "device": "auto",
    "compute_type": "float16",
    "hf_token": "",
    "word_timestamps": True,
    "include_timestamps": True,
    "prepend_punctuations": "\"'([{-",
    "append_punctuations": '"\'.\u3002\uFF0C!\uFF01?\uFF1F:\uFF1A\u201D)\u005D}\u3001',
    "unload_after_batch": True,
    "temperature_fallback": False,
    "initial_prompt": "",
}
