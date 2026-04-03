import os
import sys

# Check that the script is running inside a virtual environment
if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    print("❌ ERROR: Please activate the virtual environment first:")
    print("source venv/bin/activate")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub for downloading...")
    os.system(f"{sys.executable} -m pip install huggingface_hub tqdm")
    from huggingface_hub import snapshot_download

import config

def download_whisper():
    os.makedirs("models", exist_ok=True)

    if os.path.exists(config.WHISPER_MODEL):
        print(f"✅ Whisper already found at: {config.WHISPER_MODEL}")
        return

    print("Downloading Whisper Turbo (~500 MB). Please wait...")
    try:
        whisper_dir = snapshot_download(
            repo_id=config.WHISPER_REPO_ID,
            local_dir=config.WHISPER_MODEL,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Whisper downloaded to: {whisper_dir}")
    except Exception as e:
        print(f"❌ Error downloading Whisper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_whisper()
