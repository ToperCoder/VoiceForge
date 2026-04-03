import sys
import os
import time
import ctypes
from pathlib import Path

import config
if os.path.exists(config.CUBLAS_LIB):
    ctypes.CDLL(config.CUBLAS_LIB)

import utils
import config
import audio_extractor
import transcriber
import text_polisher

SUPPORTED_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.mp3', '.wav', '.flac', '.m4a'}

def get_files(target_path: str):
    path = Path(target_path)
    if not path.exists():
        print(f"Error: Path {target_path} not found.")
        sys.exit(1)
        
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTS:
            return [path]
        else:
            print(f"Error: Unsupported file {path.name}")
            sys.exit(1)
            
    files = []
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            files.append(f)
            
    return sorted(files)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.mp4 | folder/>")
        sys.exit(1)
        
    target = sys.argv[1]
    files = get_files(target)
    
    if not files:
        print("No supported files found.")
        sys.exit(0)
        
    config.RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    config.FINAL_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    
    processed_files = [] # list of tuples (filename, stem)
    
    # === STAGE 1: ASR ===
    print("\n" + "="*50)
    print("\u25b6 STAGE 1: LOADING WHISPER MODEL (TRANSCRIPTION)")
    print("="*50)
    print("⏳ Initializing...")
    t0 = time.time()
    whisper_model = transcriber.load_whisper()
    print(f"✅ Whisper ready! ({time.time()-t0:.1f}s)")
    
    for f in files:
        print(f"\n--- Processing file: {f.name} ---")
        temp_wav = f.with_name(f.stem + "_temp.wav")
        
        print(f"⏳ Extracting audio (ffmpeg)...")
        audio_extractor.extract_audio(str(f), str(temp_wav))
        
        print(f"⏳ Transcribing speech (this may take a while)...")
        t0 = time.time()
        try:
            text = transcriber.transcribe_audio(whisper_model, str(temp_wav))
            print(f"✅ Transcription done ({time.time()-t0:.1f}s)")
            
            raw_path = Path(config.RAW_TEXT_DIR) / f"{f.stem}.txt"
            with open(raw_path, "w", encoding="utf-8") as out:
                out.write(text)
                
            processed_files.append((f.name, f.stem))
        finally:
            if temp_wav.exists():
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass
                    
    transcriber.unload_whisper(whisper_model)
    
    # === STAGE 2: TEXT POLISHING ===
    if not processed_files:
        print("No files were transcribed.")
        return

    print("\n" + "="*50)
    print("▶ STAGE 2: LOADING QWEN MODEL (EDITING)")
    print("="*50)
    print("⏳ Loading editor model into memory...")
    t0 = time.time()
    qwen_model = text_polisher.load_qwen()
    print(f"✅ Qwen ready! ({time.time()-t0:.1f}s)")
    for original_name, stem in processed_files:
        print(f"\n--- Polishing text for: {original_name} ---")
        
        raw_path = Path(config.RAW_TEXT_DIR) / f"{stem}.txt"
        final_path = Path(config.FINAL_TEXT_DIR) / f"{stem}.txt"
        
        with open(raw_path, "r", encoding="utf-8") as inf:
            text = inf.read()
            
        print(f"⏳ Editing in progress (Qwen is analysing the text)...")
        t0 = time.time()
        if text.strip():
            polished = text_polisher.polish_text(qwen_model, text)
        else:
            polished = text
            
        with open(final_path, "w", encoding="utf-8") as outf:
            outf.write(polished)
            
        print(f"✅ Editing done ({time.time()-t0:.1f}s)")
        
    text_polisher.unload_qwen(qwen_model)
    
    print("\n" + "="*50)
    print("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
    print(f"Check folders '{config.RAW_TEXT_DIR}' and '{config.FINAL_TEXT_DIR}'.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
