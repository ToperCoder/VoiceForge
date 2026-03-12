import subprocess
import imageio_ffmpeg

def extract_audio(input_file: str, output_file: str):
    """
    Извлекает аудио из видео или конвертирует аудио файл в формат:
    - mono
    - 16kHz
    - wav
    """
    print("Extracting audio...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, 
        "-y",               # Перезаписать файл если он существует
        "-i", input_file,   # Входной файл
        "-ac", "1",         # Одноканальный звук (mono)
        "-ar", "16000",     # Частота дискретизации 16kHz
        output_file         # Выходной файл
    ]
    
    # Запускаем ffmpeg без вывода в консоль
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_file
