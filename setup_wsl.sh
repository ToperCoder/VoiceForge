#!/bin/bash

# VoiceForge Setup Script for Debian on WSL2
# Target: RTX 50xx (Blackwell, SM120), CUDA 12, Debian 12 Bookworm
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--- VoiceForge Setup for Debian WSL2 (Blackwell/SM120) ---"

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get update
# gpgv is used by apt for repo signature verification and tolerates NVIDIA's SHA1 key.
# sqv (Sequoia PGP) is stricter and rejects SHA1 since the Feb 2026 policy change.
sudo apt-get install -y python3-venv python3-pip ffmpeg build-essential cmake \
    wget curl git gnupg ca-certificates ninja-build

# 2. CUDA Toolkit 12 via apt (Debian 12 repo)
echo "[2/5] Setting up CUDA Toolkit 12..."
if ! command -v nvcc &> /dev/null; then
    echo "  Adding NVIDIA CUDA repository for Debian 12..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-8
    echo "  Appending CUDA path to ~/.bashrc..."
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
else
    echo "  CUDA already detected: $(nvcc --version | head -1)"
fi

# 3. Build llama.cpp (llama-server) with CUDA SM120 support
# SM120 = Blackwell (RTX 5080 / 5090). Compiling from source guarantees
# the correct architecture flags are set.
echo "[3/5] Building llama-server for SM120 (Blackwell)..."
LLAMA_BUILD_DIR="/tmp/llama-cpp-build"
if [ ! -f "$SCRIPT_DIR/bin/llama-server" ]; then
    rm -rf "$LLAMA_BUILD_DIR"
    git clone --depth=1 https://github.com/ggerganov/llama.cpp "$LLAMA_BUILD_DIR"
    cmake -S "$LLAMA_BUILD_DIR" -B "$LLAMA_BUILD_DIR/build" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja
    cmake --build "$LLAMA_BUILD_DIR/build" --target llama-server -j$(nproc)
    mkdir -p "$SCRIPT_DIR/bin"
    cp "$LLAMA_BUILD_DIR/build/bin/llama-server" "$SCRIPT_DIR/bin/llama-server"
    rm -rf "$LLAMA_BUILD_DIR"
    echo "  llama-server installed to bin/llama-server"
else
    echo "  llama-server already present, skipping build."
fi

# 4. Python virtual environment
echo "[4/5] Creating Python virtual environment..."
cd "$SCRIPT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Prepare project directories
echo "[5/5] Preparing directories..."
mkdir -p "$SCRIPT_DIR/models" "$SCRIPT_DIR/raw_text" "$SCRIPT_DIR/final_text"

echo ""
echo "--- Setup Complete ---"
echo "IMPORTANT: The Windows NVIDIA driver must be recent enough for the 50xx card."
echo "  It exposes CUDA to WSL2 automatically — no driver install needed inside Debian."
echo ""
echo "Place your model file at: models/Qwen-AzE.i1-Q6_K.gguf"
echo ""
echo "To run VoiceForge:"
echo "  source venv/bin/activate"
echo "  python3 main.py <file_or_directory>"
echo "---"
