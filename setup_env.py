import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd, desc):
    print(f"\n[{desc}]")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during: {desc}")
        sys.exit(1)

def main():
    print("=== VoiceForge Project Setup ===")
    
    # 1. Check/Create Virtual Environment
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Define venv python and pip paths
    if os.name == 'nt': # Windows
        python_exe = str(venv_dir / "Scripts" / "python.exe")
        pip_exe = str(venv_dir / "Scripts" / "pip.exe")
    else: # Linux/Mac
        python_exe = str(venv_dir / "bin" / "python")
        pip_exe = str(venv_dir / "bin" / "pip")

    # 2. Upgrade pip
    run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    # 3. Install llama-cpp-python with CUDA support
    # We use a direct URL to a pre-compiled binary for Windows + CUDA 12.1
    print("\n[Installing llama-cpp-python with CUDA support]")
    wheel_url = "https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/v0.3.16-cuda12.1-sm86-py312/llama_cpp_python-0.3.16+cuda12.1.sm86.ampere-cp312-cp312-win_amd64.whl"
    run_cmd([pip_exe, "install", wheel_url], "Installing llama-cpp-python (pre-compiled wheel)")

    # 4. Install other requirements
    if os.path.exists("requirements.txt"):
        run_cmd([pip_exe, "install", "-r", "requirements.txt"], "Installing other requirements")

    print("\n✅ Setup complete! Use 'venv\\Scripts\\activate' to start.")

if __name__ == "__main__":
    main()
