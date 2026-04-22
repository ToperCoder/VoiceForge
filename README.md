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


From the [Releases page](https://github.com/ToperCoder/VoiceForge/releases), download:
- `llama-server` → place in `bin/llama-server`
- `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` → place in `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`


### 2. Install dependencies

- Create and activate a Python virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Install Python requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Download Whisper model (see `models/` for structure)

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
  Qwen3-4B-Instruct-2507-Q4_K_M.gguf           # Qwen GGUF model (download from Releases)
  models--azerbaijani-whisper-small-ct2/       # Whisper model
raw_text/               # Raw ASR transcripts
final_text/             # Corrected transcripts
```
