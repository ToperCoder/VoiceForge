from pathlib import Path

WHISPER_REPO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Use Path for cross-platform directory handling
WHISPER_MODEL = Path("models") / "azerbaijani-whisper-small-ct2"

WHISPER_COMPUTE = "float16" 
LANGUAGE = "az"
USE_GPU = True

# Ollama settings
OLLAMA_API_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "voiceforge-llm"

# libcublas.so.12 path for ctranslate2 (Linux only)
CUBLAS_LIB = "/usr/lib/x86_64-linux-gnu/libcublas/12/libcublas.so.12"

# Directories for results
RAW_TEXT_DIR = Path("raw_text")
FINAL_TEXT_DIR = Path("final_text")

