import os
import sys
import subprocess
from pathlib import Path

def is_venv():
    """Проверяет, запущен ли скрипт внутри виртуального окружения (venv)."""
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def run_cmd(cmd, desc):
    print(f"\n[{desc}]")
    print(f"Выполняется: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при выполнении: {desc}")
        sys.exit(1)

def main():
    print("=== Установка зависимостей VoiceForge ===")
    print("Все пакеты устанавливаются строго в текущее виртуальное окружение.\n")

    if not is_venv():
        print("❌ ОШИБКА: Вы не находитесь в виртуальном окружении!")
        print("Пожалуйста, сначала создайте и активируйте venv:")
        print("  1. python -m venv venv")
        print("  2. venv\\Scripts\\activate")
        print("  3. python setup_env.py")
        sys.exit(1)

    print("✅ Виртуальное окружение активно.")
    
    python_exe = sys.executable

    # 1. Обновляем pip
    run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "Обновление pip")

    # 2. Устанавливаем llama-cpp-python
    print("\n--- [2/3] Установка llama-cpp-python (Компиляция с CUDA) ---")
    print("Настраиваем окружение для Visual Studio 18 (2026) и CUDA 13.2...")
    
    # Прямые пути, которые мы нашли в системе
    vcvars_path = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
    
    env = os.environ.copy()
    env["FORCE_CMAKE"] = "1"
    
    # Путь должен заканчиваться обратным слэшем для MSBuild в некоторых случаях
    cuda_path_win = cuda_path.rstrip('\\') + '\\'
    env["CudaToolkitDir"] = cuda_path_win
    env["CUDA_PATH"] = cuda_path
    
    env["CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"] = "1"
    
    # Явно указываем компилятор и путь в CMAKE_ARGS
    # Используем двойные кавычки для защиты от пробелов в путях
    # Включаем /Zc:preprocessor для совместимости CUDA 13.2 и MSVC
    nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
    env["CMAKE_ARGS"] = (
        f'-DGGML_CUDA=on '
        f'-DCUDAToolkit_ROOT="{cuda_path}" '
        f'-DCMAKE_CUDA_COMPILER="{nvcc_path}" '
        f'-DCMAKE_CXX_FLAGS="/Zc:preprocessor" '
        f'-DCMAKE_C_FLAGS="/Zc:preprocessor" '
        f'-DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"'
    )
    
    if os.path.exists(vcvars_path):
        print(f"✅ Активируем компилятор: {vcvars_path}")
        # Добавляем путь к CUDA/bin в PATH для нахождения nvcc
        env["PATH"] = os.path.join(cuda_path, "bin") + os.pathsep + env["PATH"]
        
        # Для CMake также явно укажем путь к nvcc через переменную окружения
        env["CUDACXX"] = nvcc_path
        
        # Для CMake также явно укажем путь к nvcc через переменную окружения
        env["CUDACXX"] = os.path.join(cuda_path, "bin", "nvcc.exe")
        
        cmd = f'"{vcvars_path}" x64 && set'
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('cp1251', errors='ignore')
            for line in output.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    env[key] = value
        except Exception as e:
            print(f"⚠️ Ошибка при активации окружения MSVC: {e}")
    else:
        print(f"❌ Ошибка: Не найден {vcvars_path}")
        sys.exit(1)

    # Установка
    print("--- Установка llama-cpp-python (это может занять 20-30 минут) ---")
    
    # Чтобы избежать ошибок с длинными путями в Windows (MAX_PATH),
    # будем собирать в отдельной папке с коротким именем
    build_dir = "C:\\build_llama"
    try:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir, exist_ok=True)
        
        print(f"Используем короткий путь для сборки: {build_dir}")
        
        # Очищаем папку от предыдущих попыток
        import shutil
        for item in os.listdir(build_dir):
            item_path = os.path.join(build_dir, item)
            try:
                if os.path.isfile(item_path): os.unlink(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path)
            except Exception: pass

        # Скачиваем исходники
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", 
            "--no-binary", ":all:", "llama-cpp-python>=0.3.16", 
            "-d", build_dir
        ], env=env)
        
        # Находим скачанный архив
        archives = [f for f in os.listdir(build_dir) if f.startswith("llama-cpp-python") and f.endswith(".tar.gz")]
        if not archives:
            raise Exception("Не удалось скачать исходники")
        
        import tarfile
        archive_path = os.path.join(build_dir, archives[0])
        print(f"Распаковка {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=build_dir)
            
        # Находим папку после распаковки
        extracted_dirs = [d for d in os.listdir(build_dir) if os.path.isdir(os.path.join(build_dir, d)) and "llama-cpp-python" in d]
        if not extracted_dirs:
            raise Exception("Папка не найдена после распаковки")
            
        extracted_dir = os.path.join(build_dir, extracted_dirs[0])
        
        print(f"Запуск сборки из {extracted_dir}...")
        # Устанавливаем из локальной папки
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            ".", "--no-cache-dir", "--force-reinstall", "--upgrade"
        ], cwd=extracted_dir, env=env)
        
        print("✅ llama-cpp-python успешно установлена!")
    except Exception as e:
        print(f"❌ Ошибка компиляции: {e}")
        print("Попробуйте выполнить: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
        sys.exit(1)

    # 3. Устанавливаем остальные мелкие зависимости из requirements
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, "r") as f:
            reqs = [line.strip() for line in f if line.strip() and 
                    "llama-cpp-python" not in line.lower()]
        
        if reqs:
            run_cmd([python_exe, "-m", "pip", "install"] + reqs, "Установка остальных библиотек")
    
    print("\n✅ Установка успешно завершена!")
    print("\nТеперь запустите: python main.py <ваш_аудио_файл.m4a>")

if __name__ == "__main__":
    main()
