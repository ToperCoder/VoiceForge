from pathlib import Path

WHISPER_REPO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Use Path for cross-platform directory handling
WHISPER_MODEL = Path("models") / "models--faster-whisper-large-v3-turbo-ct2"
WHISPER_COMPUTE = "float16" 
LANGUAGE = "az"
QWEN_MODEL_PATH = Path("models").resolve() / "Qwen-AzE.i1-Q6_K.gguf"
USE_GPU = True

# llama-server binary (built by setup_wsl.sh)
LLAMA_SERVER_BIN = Path(__file__).parent / "bin" / "llama-server"
LLAMA_SERVER_PORT = 8080

# Directories for results
RAW_TEXT_DIR = Path("raw_text")
FINAL_TEXT_DIR = Path("final_text")

