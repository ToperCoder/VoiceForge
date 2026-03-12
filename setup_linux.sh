#!/bin/bash

# VoiceForge Setup Script for Debian-based systems
set -e

echo "--- VoiceForge Setup for Debian ---"

# 1. Update and install system dependencies
echo "[1/4] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg build-essential cmake

# Note: NVIDIA drivers and CUDA Toolkit must be installed manually or via:
# sudo apt install -y nvidia-driver nvidia-cuda-toolkit

# 2. Create virtual environment
echo "[2/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
echo "[3/4] Installing Python packages..."
# Faster-whisper and llama-cpp-python usually require some env vars for GPU support on Linux
# We'll try to install them from the requirements file
pip install --upgrade pip
pip install -r requirements.txt

# 4. Prepare directories
echo "[4/4] Preparing directories..."
mkdir -p models raw_text final_text

echo ""
echo "--- Setup Complete ---"
echo "To run VoiceForge:"
echo "1. source venv/bin/activate"
echo "2. python3 main.py"
echo "---"
