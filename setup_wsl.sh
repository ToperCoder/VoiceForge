#!/bin/bash

# VoiceForge Setup Script for Debian on WSL2
# Requires pre-built bin/llama-server (CUDA 12.8, multi-arch).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--- VoiceForge Setup for Debian WSL2 ---"

# 1. System dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg wget curl gnupg ca-certificates

# 2. CUDA 12 runtime libraries (needed by llama-server and faster-whisper)
echo "[2/4] Installing CUDA 12 runtime libraries..."
if ! dpkg -s libcublas12-cuda-12 &> /dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y libcublas12-cuda-12
else
    echo "  CUDA 12 runtime already installed."
fi

# 3. Check pre-built llama-server binary
echo "[3/4] Checking llama-server binary..."
if [ ! -f "$SCRIPT_DIR/bin/llama-server" ]; then
    echo "  ERROR: bin/llama-server not found."
    echo "  Place the pre-built llama-server binary at: $SCRIPT_DIR/bin/llama-server"
    exit 1
fi
echo "  llama-server found."

# 4. Python virtual environment
echo "[4/4] Creating Python virtual environment..."
cd "$SCRIPT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p "$SCRIPT_DIR/models" "$SCRIPT_DIR/raw_text" "$SCRIPT_DIR/final_text"

echo ""
echo "--- Setup Complete ---"
echo "IMPORTANT: The Windows NVIDIA driver must be recent enough for your GPU."
echo "  It exposes CUDA to WSL2 automatically — no driver install needed inside Debian."
echo ""
echo "Place your model file at: models/Qwen-AzE.i1-Q6_K.gguf"
echo ""
echo "To run VoiceForge:"
echo "  source venv/bin/activate"
echo "  python3 main.py <file_or_directory>"
echo "---"
