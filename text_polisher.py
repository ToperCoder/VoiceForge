import gc
import sys
import time
import subprocess
import requests
import utils
import config


def load_qwen() -> subprocess.Popen:
    """
    Запускает llama-server как подпроцесс и ждёт готовности.
    """
    if not config.QWEN_MODEL_PATH.exists():
        print(f"❌ ОШИБКА: Файл модели не найден: {config.QWEN_MODEL_PATH}")
        sys.exit(1)

    if not config.LLAMA_SERVER_BIN.exists():
        print(f"❌ ОШИБКА: llama-server не найден: {config.LLAMA_SERVER_BIN}")
        print("Запустите setup_wsl.sh для сборки llama.cpp.")
        sys.exit(1)

    n_gpu = -1 if config.USE_GPU else 0

    cmd = [
        str(config.LLAMA_SERVER_BIN),
        "--model", str(config.QWEN_MODEL_PATH),
        "--ctx-size", "4096",
        "--n-gpu-layers", str(n_gpu),
        "--port", str(config.LLAMA_SERVER_PORT),
        "--host", "127.0.0.1",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"\n❌ ОШИБКА запуска llama-server: {e}")
        sys.exit(1)

    health_url = f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/health"
    for _ in range(60):
        if proc.poll() is not None:
            stderr_out = proc.stderr.read().decode(errors="replace").strip()
            print(f"❌ ОШИБКА: llama-server завершился с кодом {proc.returncode}")
            if stderr_out:
                print("--- llama-server stderr ---")
                print(stderr_out[-2000:])  # last 2000 chars is enough for diagnosis
                print("---------------------------")
            sys.exit(1)
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.terminate()
    print("❌ ОШИБКА: llama-server не запустился за 60 секунд.")
    sys.exit(1)


def polish_text(proc: subprocess.Popen, text: str) -> str:
    """
    Полирует текст через HTTP API llama-server (OpenAI-compatible).
    """
    if len(text.strip()) < 5:
        return text

    messages = [
        {
            "role": "system",
            "content": (
                "Sən Azərbaycan dili üzrə peşəkar redaktorsan. Tapşırığın: mətndəki nitq tanıma (ASR) səhvlərini düzəltməkdir.\n"
                "MƏSULİYYƏT:\n"
                "1. Mətnin mənasını və BÜTÜN cümlələri olduğu kimi saxla. Heç bir məlumatı silmə və ya ümumiləşdirmə.\n"
                "2. Xüsusi adları, sayt adlarını (məs: 'Modernaz'), rəqəmləri və şəxsləri dəyişmə.\n"
                "3. Yalnız aşkar fonetik səhvləri düzəlt (məs: 'yarım çıx' -> 'yarımçıq', 'ərzasını' -> 'ərizəsini', 'isteyfa' -> 'istefa').\n"
                "4. Əlavə şərh yazma, yalnız təmizlənmiş mətni qaytar.\n\n"
                "NÜMUNƏ:\n"
                "Giriş: 'Modernazın xəbərinə görə ərzasını bugün rəsmən təqdim edəcək edilir. Yarım çıx kəsildi.'\n"
                "Çıxış: 'Modernazın xəbərinə görə ərizəsini bu gün rəsmən təqdim edir. Yarımçıq kəsildi.'"
            )
        },
        {"role": "user", "content": f"Aşağıdakı mətni peşəkar şəkildə redaktə et (heç nəyi silmə):\n\n{text}"}
    ]

    max_tokens = min(2048, len(text) * 2 + 100)

    try:
        response = requests.post(
            f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return utils.clean_llm_output(result)
    except Exception as e:
        print(f"⚠️ Ошибка при полировке: {e}")
        return text


def unload_qwen(proc: subprocess.Popen):
    """
    Останавливает llama-server и освобождает ресурсы.
    """
    print("\n[DEBUG] Остановка llama-server...")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    gc.collect()
    print("✅ llama-server остановлен.\n")
