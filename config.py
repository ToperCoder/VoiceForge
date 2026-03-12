from pathlib import Path

WHISPER_REPO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Use Path for cross-platform directory handling
WHISPER_MODEL = Path("models") / "models--faster-whisper-large-v3-turbo-ct2"
WHISPER_COMPUTE = "float16" 
LANGUAGE = "az"
QWEN_MODEL_PATH = Path("models").resolve() / "Qwen-AzE.i1-Q6_K.gguf"
USE_GPU = True

# Directories for results
RAW_TEXT_DIR = Path("raw_text")
FINAL_TEXT_DIR = Path("final_text")

