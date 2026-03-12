import os
import sys

# Проверка, что скрипт запущен в виртуальном окружении
if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    print("❌ ОШИБКА: Пожалуйста, активируйте виртуальное окружение:")
    print("venv\\Scripts\\activate")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Устанавливаем huggingface_hub для скачивания...")
    os.system(f"{sys.executable} -m pip install huggingface_hub tqdm")
    from huggingface_hub import snapshot_download, hf_hub_download

import config

def download_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Скачивание Whisper Turbo (для CTranslate2 / faster-whisper)
    print("\n=== [1/2] Скачивание модели распознавания речи (Whisper Turbo) ===")
    
    if os.path.exists(config.WHISPER_MODEL):
        print(f"✅ Whisper уже найден в: {config.WHISPER_MODEL}")
    else:
        print("Эта модель намного точнее для азербайджанского. Пожалуйста, подождите...")
        try:
            whisper_dir = snapshot_download(
                repo_id=config.WHISPER_REPO_ID,
                local_dir=config.WHISPER_MODEL,
                local_dir_use_symlinks=False, # Важно для Windows
                resume_download=True
            )
            print(f"✅ Whisper успешно скачан/проверен в: {whisper_dir}")
        except Exception as e:
            print(f"❌ Ошибка скачивания Whisper: {e}")

    # 2. Скачивание Qwen 2.5 (Языковая модель)
    print("\n=== [2/2] Скачивание модели редактора (Qwen 2.5 3B) ===")
    
    import glob
    existing_gguf = glob.glob(os.path.join(models_dir, "*.gguf"))
    
    if existing_gguf:
        print(f"✅ GGUF модель уже найдена: {os.path.basename(existing_gguf[0])}")
        print("Пропускаем скачивание Qwen.")
    else:
        qwen_filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
        qwen_path = os.path.join(models_dir, qwen_filename)
        print("Эта модель весит около ~2.5 ГБ. Заваривайте чай...")
        try:
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename=qwen_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ Qwen успешно скачан в: {qwen_path}")
        except Exception as e:
            print(f"❌ Ошибка скачивания Qwen: {e}")

    print("\n🎉 Все модели готовы! Теперь можно запускать:")
    print("python main.py Recording.m4a")

if __name__ == "__main__":
    download_models()
