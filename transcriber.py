import gc
from faster_whisper import WhisperModel
import config

def load_whisper():
    """
    Loads the Whisper model.
    """
    device = "cuda" if config.USE_GPU else "cpu"
    
    model = WhisperModel(
        str(config.WHISPER_MODEL), 
        device=device, 
        compute_type=config.WHISPER_COMPUTE,
        local_files_only=True
    )
    return model

def transcribe_audio(model: WhisperModel, audio_path: str) -> str:
    """
    Transcribes speech from an audio file.
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
    Fully unloads the Whisper model from memory.
    """
    print("\n[DEBUG] Releasing Whisper memory...")
    if model:
        try:
            # Remove references
            del model
        except:
            pass
    gc.collect()
    # gc.collect() should trigger C++ destructors since we use C++ libs
    print("✅ Whisper memory cleared.\n")
