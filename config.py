from pathlib import Path

WHISPER_REPO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Use Path for cross-platform directory handling
WHISPER_MODEL = Path("models") / "azerbaijani-whisper-small-ct2"

WHISPER_COMPUTE = "float16" 
LANGUAGE = "az"
# Use the latest Qwen3-4B-Instruct-2507 checkpoint
QWEN_MODEL_PATH = Path("models").resolve() / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
USE_GPU = True

# llama-server binary (built by setup_wsl.sh)
LLAMA_SERVER_BIN = Path(__file__).parent / "bin" / "llama-server"
LLAMA_SERVER_PORT = 8080

# libcublas.so.12 path for ctranslate2 and llama-server (Linux only)
CUBLAS_LIB = "/usr/lib/x86_64-linux-gnu/libcublas/12/libcublas.so.12"

# Directories for results
RAW_TEXT_DIR = Path("raw_text")
FINAL_TEXT_DIR = Path("final_text")

