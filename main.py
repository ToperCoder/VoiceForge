import sys
import os
import time
from pathlib import Path
import utils
# Setup DLL paths before any other local imports that might depend on them
utils.setup_cuda_path()

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

    
    processed_files = [] # Список кортежей (имя_файла, имя_без_расширения)
    
    # === STAGE 1: ASR ===
    print("\n" + "="*50)
    print("▶ ЭТАП 1: ЗАГРУЗКА МОДЕЛИ WHISPER (РАСПОЗНАВАНИЕ)")
    print("="*50)
    print("⏳ Инициализация (если запускается впервые, может начаться скачивание ~500 МБ)...")
    t0 = time.time()
    whisper_model = transcriber.load_whisper()
    print(f"✅ Whisper готов! (заняло {time.time()-t0:.1f} сек)")
    
    for f in files:
        print(f"\n--- Обработка файла: {f.name} ---")
        temp_wav = f.with_name(f.stem + "_temp.wav")
        
        print(f"⏳ Разделение аудио (ffmpeg)...")
        audio_extractor.extract_audio(str(f), str(temp_wav))
        
        print(f"⏳ Распознавание речи (это может занять время)...")
        t0 = time.time()
        try:
            text = transcriber.transcribe_audio(whisper_model, str(temp_wav))
            print(f"✅ Распознавание завершено (заняло {time.time()-t0:.1f} сек)")
            
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
        print("Ни одного файла не было распознано.")
        return
        
    print("\n" + "="*50)
    print("▶ ЭТАП 2: ЗАГРУЗКА МОДЕЛИ QWEN (РЕДАКТИРОВАНИЕ)")
    print("="*50)
    print("⏳ Подгружаем нейросеть-редактора в память...")
    t0 = time.time()
    qwen_model = text_polisher.load_qwen()
    print(f"✅ Qwen готов! (заняло {time.time()-t0:.1f} сек)")
    
    for original_name, stem in processed_files:
        print(f"\n--- Полировка текста для: {original_name} ---")
        
        raw_path = Path(config.RAW_TEXT_DIR) / f"{stem}.txt"
        final_path = Path(config.FINAL_TEXT_DIR) / f"{stem}.txt"
        
        with open(raw_path, "r", encoding="utf-8") as inf:
            text = inf.read()
            
        print(f"⏳ Идет редактура (модель Qwen анализирует текст)...")
        t0 = time.time()
        if text.strip():
            polished = text_polisher.polish_text(qwen_model, text)
        else:
            polished = text
            
        with open(final_path, "w", encoding="utf-8") as outf:
            outf.write(polished)
            
        print(f"✅ Редактура завершена (заняло {time.time()-t0:.1f} сек)")
        
    text_polisher.unload_qwen(qwen_model)
    
    print("\n" + "="*50)
    print("🎉 ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print(f"Проверьте папки '{config.RAW_TEXT_DIR}' и '{config.FINAL_TEXT_DIR}'.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
