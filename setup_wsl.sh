#!/bin/bash

# VoiceForge Setup Script for WSL2 (Ubuntu 22.04)
# Pre-built llama-server included — no build step required.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--- VoiceForge Setup for WSL2 ---"

# 1. System dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg wget curl gnupg ca-certificates

# 2. CUDA runtime libraries
echo "[2/4] Setting up CUDA runtime libraries..."
if ! dpkg -l libcublas12-cuda-12 &>/dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y libcublas12-cuda-12 cuda-cudart-12-8
else
    echo "  CUDA runtime already installed."
fi

RELEASE_URL="https://github.com/ToperCoder/VoiceForge/releases"

# 3. Check pre-built llama-server
echo "[3/5] Checking llama-server binary..."
if [ ! -f "$SCRIPT_DIR/bin/llama-server" ]; then
    echo ""
    echo "ERROR: bin/llama-server not found."
    echo "  Download it from: $RELEASE_URL"
    echo "  Place it at: $SCRIPT_DIR/bin/llama-server"
    echo ""
    exit 1
fi
chmod +x "$SCRIPT_DIR/bin/llama-server"
echo "  llama-server found."

# 4. Check model file
echo "[4/5] Checking model file..."
if [ ! -f "$SCRIPT_DIR/models/Qwen-AzE.i1-Q6_K.gguf" ]; then
    echo ""
    echo "ERROR: models/Qwen-AzE.i1-Q6_K.gguf not found."
    echo "  Download it from: $RELEASE_URL"
    echo "  Place it at: $SCRIPT_DIR/models/Qwen-AzE.i1-Q6_K.gguf"
    echo ""
    exit 1
fi
echo "  Model file found."

# 5. Python virtual environment
echo "[5/6] Creating Python virtual environment..."
cd "$SCRIPT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Download Whisper model
echo "[6/6] Downloading Whisper model (~500 MB)..."
python3 download_whisper.py

# Prepare project directories
mkdir -p "$SCRIPT_DIR/models" "$SCRIPT_DIR/raw_text" "$SCRIPT_DIR/final_text"

echo ""
echo "--- Setup Complete ---"
echo "IMPORTANT: The Windows NVIDIA driver must be installed. No driver needed inside WSL."
echo ""
echo "To run VoiceForge:"
echo "  source venv/bin/activate"
echo "  python3 main.py <file_or_directory>"
echo "---"
