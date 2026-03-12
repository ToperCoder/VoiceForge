import os

WHISPER_REPO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_MODEL = os.path.join("models", "models--faster-whisper-large-v3-turbo-ct2")
WHISPER_COMPUTE = "float16" 
LANGUAGE = "az"
QWEN_MODEL_PATH = os.path.abspath(os.path.join("models", "Qwen-AzE.i1-Q6_K.gguf"))
USE_GPU = True

# Папки для результатов
RAW_TEXT_DIR = "raw_text"
FINAL_TEXT_DIR = "final_text"
