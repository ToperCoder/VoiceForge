import gc
import os
import re
import sys
import time
import subprocess
import requests
import utils
import config

# Qwen3 system prompt: strict ASR corrector, non-thinking mode
_SYSTEM_PROMPT = (
    "Sən yalnız ASR (speech-to-text) korrektorusən.\n"
    "SƏRT QAYDALAR:\n"
    "1. Heç bir cümləni yenidən yazma.\n"
    "2. Cümlə strukturunu dəyişmə.\n"
    "3. Heç bir sözü silmə.\n"
    "4. Heç bir söz əlavə etmə.\n"
    "5. Yalnız səhv yazılmış sözləri düzəlt.\n"
    "6. Əgər əmin deyilsənsə sözü olduğu kimi saxla.\n"
    "7. Sinonim istifadə ETMƏ.\n"
    "8. Parafraz ETMƏ.\n"
    "9. Yalnız typo fix.\n"
    "10. Output input ilə maksimum oxşar olmalıdır.\n"
    "11. Dəyişiklik sayı mümkün qədər az olmalıdır.\n"
    "12. Output 95% input ilə eyni olmalıdır.\n\n"
    "İCAZƏ VERİLƏN:\n"
    "məcarat → macəra\n"
    "təbiyyət → təbiət\n"
    "lüqət → lüğət\n\n"
    "QADAĞANDIR:\n"
    "cümlə dəyişmək\n"
    "qrammatika yaxşılaşdırmaq\n"
    "stil dəyişmək\n"
    "məzmun dəyişmək\n\n"
    "--- NÜMUNƏ ---\n"
    "GİRİŞ: Onun məcarat səyahəti haqqında təbiyyəti lüqətdə belə izah olunur.\n"
    "ÇIXIŞ: Onun macəra səyahəti haqqında təbiəti lüğətdə belə izah olunur.\n"
    "--- NÜMUNƏ SONU ---\n\n"
    "Only correct characters, not sentences.\n"
    "Yalnız düzəldilmiş mətni qaytar."
)


def load_qwen() -> subprocess.Popen:
    """
    Launches llama-server as a subprocess and waits for it to be ready.
    """
    if not config.QWEN_MODEL_PATH.exists():
        print(f"❌ ERROR: Model file not found: {config.QWEN_MODEL_PATH}")
        sys.exit(1)

    if not config.LLAMA_SERVER_BIN.exists():
        print(f"❌ ERROR: llama-server not found: {config.LLAMA_SERVER_BIN}")
        print("Run setup_wsl.sh to set up the binary.")
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

    env = os.environ.copy()
    if os.path.exists(config.CUBLAS_LIB):
        env["LD_PRELOAD"] = config.CUBLAS_LIB

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    except Exception as e:
        print(f"\n❌ ERROR launching llama-server: {e}")
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
    print("❌ ERROR: llama-server did not start within 60 seconds.")
    sys.exit(1)


def _correct_chunk(chunk: str) -> str:
    """
    Sends a single sentence chunk to Qwen3 for ASR correction.
    /no_think disables Qwen3's reasoning mode — faster, no -----------
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Bu ASR outputudur. Minimum dəyişiklik et.\n"
                "Only fix obvious speech recognition errors.\n"
                "Do NOT add any content not present in the input.\n"
                "Do NOT write more text than the input.\n"
                "/no_think\n\n"
                + chunk
            )
        }
    ]

    # ~3 chars/token for Azerbaijani (diacritics tokenize as multi-byte subwords)
    # 1.5x multiplier to ensure output is never truncated mid-word
    max_tokens = min(1024, int(len(chunk) / 3 * 1.5) + 100)

    response = requests.post(
        f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "repeat_penalty": 1.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()
    corrected = utils.clean_llm_output(result)

    # Sanity check: if output is less than 70% of input length, model likely
    # truncated or hallucinated. Keep the original chunk instead.
    if len(corrected) < len(chunk) * 0.7:
        return chunk

    return corrected


def polish_text(proc: subprocess.Popen, text: str) -> str:
    """
    Hybrid: single call for short text, paragraph chunking for long text.
    Paragraph = separated by two or more newlines.
    """
    if len(text.strip()) < 5:
        return text

    # If short, do a single call (faster, better context)
    if len(text) < 2000:
        try:
            return _correct_chunk(text)
        except Exception as e:
            print(f"⚠️ Correction failed, keeping original: {e}")
            return text

    # For long text, split by paragraph (two or more newlines)
    paragraphs = re.split(r'\n{2,}', text.strip())
    paragraphs = [p for p in paragraphs if p.strip()]

    corrected = []
    for para in paragraphs:
        try:
            fixed = _correct_chunk(para)
            corrected.append(fixed)
        except Exception as e:
            print(f"⚠️ Paragraph correction failed, keeping original: {e}")
            corrected.append(para)

    return "\n\n".join(corrected)


def unload_qwen(proc: subprocess.Popen):
    """
    Stops llama-server and releases resources.
    """
    print("\n[DEBUG] Stopping llama-server...")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    gc.collect()
    print("✅ llama-server stopped.\n")
