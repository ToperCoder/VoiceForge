#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# System dependencies
sudo apt-get update
sudo apt-get install -y gpgv python3-venv python3-pip ffmpeg build-essential cmake \
    wget curl git gnupg ca-certificates ninja-build
sudo apt-get remove -y --purge sqv 2>/dev/null || true

# CUDA Toolkit 12.8
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Build llama-server
git clone --depth=1 https://github.com/ggerganov/llama.cpp /tmp/llama-cpp-build
cmake -S /tmp/llama-cpp-build -B /tmp/llama-cpp-build/build \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build /tmp/llama-cpp-build/build --target llama-server -j$(nproc)
mkdir -p "$SCRIPT_DIR/bin"
cp /tmp/llama-cpp-build/build/bin/llama-server "$SCRIPT_DIR/bin/llama-server"
rm -rf /tmp/llama-cpp-build

# Python environment
cd "$SCRIPT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Project directories
mkdir -p "$SCRIPT_DIR/models" "$SCRIPT_DIR/raw_text" "$SCRIPT_DIR/final_text"
