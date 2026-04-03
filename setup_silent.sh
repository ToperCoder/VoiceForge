#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# System dependencies
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg wget curl gnupg ca-certificates

# CUDA runtime libraries
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y libcublas12-cuda-12 cuda-cudart-12-8

# Check pre-built llama-server
if [ ! -f "$SCRIPT_DIR/bin/llama-server" ]; then
    echo "ERROR: bin/llama-server not found." >&2
    exit 1
fi
chmod +x "$SCRIPT_DIR/bin/llama-server"

# Python environment
cd "$SCRIPT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Project directories
mkdir -p "$SCRIPT_DIR/models" "$SCRIPT_DIR/raw_text" "$SCRIPT_DIR/final_text"
