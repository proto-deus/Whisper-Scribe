# Whisper Scribe

A desktop GUI application for batch audio/video transcription powered by OpenAI's Whisper speech-to-text models.

![GUI](https://raw.githubusercontent.com/proto-deus/Whisper-Scribe/refs/heads/main/gui/gui.jpg)

## Features

- **Multiple transcription engines** -- Choose between OpenAI Whisper, Faster Whisper (default), or Insanely Fast Whisper
- **Batch processing** -- Transcribe multiple audio and video files in one session
- **Drag-and-drop** -- Add files or folders by dragging them into the window
- **Output formats** -- Export as SRT, WebVTT, or plain TXT (with or without timestamps)
- **Translation** -- Translate transcription output to another language
- **Language detection** -- Auto-detect the source language or manually select from 90+ languages
- **Speaker diarization** -- Identify and label different speakers using pyannote.audio
- **Background music removal** -- Isolate vocals before transcription using Demucs
- **Voice activity detection** -- Filter out non-speech segments with Silero VAD
- **Video support** -- Automatically extracts audio from video files via ffmpeg
- **Model management** -- Load, unload, and delete downloaded models with memory monitoring
- **Configurable inference** -- Fine-tune beam size, temperature, patience, VAD thresholds, and more

## Supported Formats

**Audio:** MP3, WAV, FLAC, OGG, M4A, AAC, WMA, OPUS, AIFF, APE, ALAC, MKA, RA, TTA, WV, TAK, DSF, DFF

**Video:** MP4, MKV, AVI, MOV, WebM, FLV, WMV, M4V, TS, MTS, VOB, 3GP, OGV, RM, RMVB, ASF, DIVX

## Requirements

- **Python 3.11**
- **ffmpeg** and **ffprobe** installed and available on your PATH

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/whisper-scribe.git
   cd whisper-scribe
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ffmpeg is installed:
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg` (or equivalent)

## Usage

Launch the application:

```bash
python main.py
```

1. Add audio or video files using the file picker or drag-and-drop
2. Configure transcription settings (engine, model, language, output format)
3. Click **Start** to begin transcription
4. Output files are saved alongside the source files (or to a custom directory)

## Speaker Diarization

To use speaker diarization, you need a HuggingFace account token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the [pyannote model terms](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Enter the token in **Settings > HuggingFace Token**

The token is stored securely in your operating system's keyring.

## Models

Models are downloaded automatically on first use and cached in `%LOCALAPPDATA%\Whisper Scribe\Models\` (or `~/Whisper Scribe/Models/` on non-Windows). The default model is **Faster Whisper -- large-v3**.

Available model families:

| Engine | Models |
|---|---|
| **OpenAI Whisper** | tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3, turbo |
| **Faster Whisper** | tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1/v2/v3, distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3, turbo variants |
| **Insanely Fast Whisper** | openai/whisper-tiny through large-v3-turbo, distil-whisper variants |

Click the **Models** button in the top bar to load, unload, or delete cached models.

## Configuration

Settings are persisted to `%LOCALAPPDATA%\Whisper Scribe\settings.json` (or `~/Whisper Scribe/settings.json` on non-Windows).

Key settings include:

| Setting | Default | Description |
|---|---|---|
| Output format | SRT | TXT, SRT, or WebVTT |
| Language | Auto Detect | Source language or auto-detect |
| Model engine | Faster Whisper | Transcription backend |
| Model | large-v3 | Whisper model variant |
| Beam size | 5 | Number of beams for decoding |
| VAD filter | Enabled | Silero VAD or faster-whisper built-in |
| Speaker diarization | Disabled | Label speakers (requires HF token) |
| Background music removal | Disabled | Isolate vocals with Demucs |
| Device | Auto | CUDA or CPU |
| Compute type | float16 | Inference precision |

