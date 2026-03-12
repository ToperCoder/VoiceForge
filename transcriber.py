import gc
from faster_whisper import WhisperModel
import config

def load_whisper():
    """
    Загружает модель Whisper.
    """
    device = "cuda" if config.USE_GPU else "cpu"
    
    model = WhisperModel(
        config.WHISPER_MODEL, 
        device=device, 
        compute_type=config.WHISPER_COMPUTE,
        local_files_only=True
    )
    return model

def transcribe_audio(model: WhisperModel, audio_path: str) -> str:
    """
    Распознает речь из аудиофайла.
    """
    segments, info = model.transcribe(
        audio_path,
        language=config.LANGUAGE,
        beam_size=5,
        vad_filter=True
    )
    
    text = ""
    for segment in segments:
        text += segment.text + " "
    
    return text.strip()

def unload_whisper(model):
    """
    Полностью выгружает модель Whisper из памяти.
    """
    print("\nUnloading Whisper...\n")
    del model
    gc.collect()
