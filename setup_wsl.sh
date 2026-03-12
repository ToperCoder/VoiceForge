#!/bin/bash

# VoiceForge Setup Script for Debian on WSL2
set -e

echo "--- VoiceForge Setup for Debian (WSL2) ---"

# 1. Update and install system dependencies
echo "[1/4] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg build-essential cmake wget gnupg

# 2. Setup NVIDIA CUDA for WSL2 (Debian 12 Bookworm example)
# Note: This follows the official NVIDIA guide for WSL2
echo "[2/4] Setting up CUDA Toolkit for WSL2..."
if ! command -v nvcc &> /dev/null; then
    echo "Adding NVIDIA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    # We install ONLY the toolkit, the driver comes from Windows host
    sudo apt install -y cuda-toolkit-12-3 
    rm cuda-keyring_1.1-1_all.deb
else
    echo "CUDA Toolkit already detected."
fi

# 3. Create virtual environment
echo "[3/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
echo "[4/4] Installing Python packages..."
pip install --upgrade pip

# For WSL2 CUDA support, some packages might need specific versions
# llama-cpp-python needs to be compiled with CUDA support
echo "Building llama-cpp-python with CUDA support..."
setenv CMAKE_ARGS "-DGGML_CUDA=ON"
pip install -r requirements.txt

# 5. Prepare directories
echo "[5/4] Preparing directories..."
mkdir -p models raw_text final_text

echo ""
echo "--- Setup Complete ---"
echo "IMPORTANT: Make sure you have the latest NVIDIA Game Ready / Studio Driver installed on WINDOWS."
echo ""
echo "To run VoiceForge in WSL2:"
echo "1. source venv/bin/activate"
echo "2. python3 main.py"
echo "---"
