# VoiceForge

Azerbaijani speech-to-text pipeline. Transcribes audio/video files using Whisper, then corrects ASR errors using a local Qwen LLM.

## How it works

1. **Stage 1 — Transcription**: Extracts audio with ffmpeg, transcribes with `faster-whisper-large-v3-turbo`
2. **Stage 2 — Editing**: Sends raw transcript to a local `llama-server` (Qwen model) to fix ASR errors
3. Results are saved to `raw_text/` (raw transcript) and `final_text/` (corrected)

## Requirements

- WSL2 with Ubuntu 22.04
- NVIDIA GPU (RTX 30xx / 40xx / 50xx)
- Windows NVIDIA driver installed (no driver needed inside WSL)

## Setup

### 1. Download release assets

From the [Releases page](https://github.com/toperus/VoiceForge/releases), download:
- `llama-server` → place in `bin/llama-server`
- `Qwen-AzE.i1-Q6_K.gguf` → place in `models/Qwen-AzE.i1-Q6_K.gguf`

### 2. Run setup script

```bash
bash setup_wsl.sh
```

This installs system dependencies, CUDA runtime libraries, sets up the Python virtual environment, and downloads the Whisper model automatically.

## Usage

```bash
source venv/bin/activate

# Single file
python3 main.py Recording.m4a

# Entire folder
python3 main.py recordings/
```

Supported formats: `.mp4`, `.mkv`, `.avi`, `.mov`, `.mp3`, `.wav`, `.flac`, `.m4a`

## Project structure

```
bin/
  llama-server          # Pre-built binary (download from Releases)
models/
  Qwen-AzE.i1-Q6_K.gguf                        # Qwen GGUF model (download from Releases)
  models--faster-whisper-large-v3-turbo-ct2/    # Whisper model (download_whisper.py)
raw_text/               # Raw ASR transcripts
final_text/             # Corrected transcripts
```
