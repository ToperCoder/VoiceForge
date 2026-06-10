import gc
import os
import re
import sys
import time
import subprocess
import requests
import utils
import config

# LLM system prompt: strict ASR corrector, non-thinking mode
_SYSTEM_PROMPT = (
    "Sən Azərbaycan dili ASR (speech-to-text) korrektorusən.\n"
    "Vəzifən: Avtomatik nitq tanıma sisteminin səhvlərini düzəltmək.\n\n"
    "ASR səhvləri necə olur?\n"
    "- Fonetik oxşar sözlər qarışır: 'turistlərin' → 'istilərin', 'subyekt' → 'süqlət'\n"
    "- Hərflər düşür və ya əlavə olunur: 'qoyuluşunun' → 'qoruşunun', 'dəqiq' → 'təqiq'\n"
    "- Şəkilçilər səhv tanınır: 'qarşılıqlı əlaqə' → 'qarşılıqla əlaqələ'\n"
    "- Mürəkkəb terminlər səhv yazılır: 'subyektlərinin' → 'süqletlərinin'\n\n"
    "QAYDALAR:\n"
    "1. Cümlə strukturunu, söz sırasını və məzmunu dəyişmə.\n"
    "2. Söz əlavə etmə və ya silmə.\n"
    "3. Sinonim və parafraz istifadə etmə.\n"
    "4. Yalnız səhv tanınan sözləri kontekstə əsasən düzəlt.\n"
    "5. Əgər söz düzgündürsə toxunma.\n\n"
    "NÜMUNƏLƏR:\n"
    "GİRİŞ: İstilərin xüsusi coğrafi xüsusiyyətlərə malik ərazilərdə fiziki aktivlik müşayiət olunan turizm.\n"
    "ÇIXIŞ: Turistlərin xüsusi coğrafi xüsusiyyətlərə malik ərazilərdə fiziki aktivlik müşayiət olunan turizm.\n\n"
    "GİRİŞ: Sahibkarlıq süqletlərinin inkişafına kömək edən sərhədləri təqiq müəyyən edilmiş ərazilər.\n"
    "ÇIXIŞ: Sahibkarlıq subyektlərinin inkişafına kömək edən sərhədləri dəqiq müəyyən edilmiş ərazilər.\n\n"
    "GİRİŞ: İnvestisiya qoruşunun təşviqi məqsədilə ayrılan ərazilər.\n"
    "ÇIXIŞ: İnvestisiya qoyuluşunun təşviqi məqsədilə ayrılan ərazilər.\n\n"
    "GİRİŞ: Məzbunu ifadə etmir.\n"
    "ÇIXIŞ: Məzmunu ifadə etmir.\n\n"
    "Yalnız düzəldilmiş mətni qaytar. Heç bir izahat yazma."
)


def load_llm():
    """
    Checks if Ollama is running and accessible.
    """
    print("⏳ Checking Ollama connection...")
    try:
        response = requests.get(f"{config.OLLAMA_API_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            return True
        else:
            print(f"❌ ERROR: Ollama returned status code {response.status_code}")
            sys.exit(1)
    except requests.ConnectionError:
        print("❌ ERROR: Ollama is not running. Please start Ollama and try again.")
        print("   If you haven't imported the model yet, run: ollama create voiceforge-llm -f Modelfile")
        sys.exit(1)


def _correct_chunk(chunk: str) -> str:
    """
    Sends a single sentence chunk to the LLM for ASR correction.
    /no_think disables the reasoning mode — faster, no -----------
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Bu ASR çıxışıdır. Kontekstə əsasən səhv tanınan sözləri düzəlt.\n"
                "Cümlə strukturunu və məzmunu dəyişmə.\n"
                "/no_think\n\n"
                + chunk
            )
        }
    ]

    # ~3 chars/token for Azerbaijani (diacritics tokenize as multi-byte subwords)
    # 1.5x multiplier to ensure output is never truncated mid-word
    max_tokens = min(1024, int(len(chunk) / 3 * 1.5) + 100)

    response = requests.post(
        f"{config.OLLAMA_API_URL}/v1/chat/completions",
        json={
            "model": config.OLLAMA_MODEL_NAME,
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


def polish_text(text: str) -> str:
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


def unload_llm():
    """
    Unloads the model from Ollama's VRAM.
    """
    print("\n[DEBUG] Unloading model from VRAM...")
    try:
        requests.post(
            f"{config.OLLAMA_API_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL_NAME, "keep_alive": 0},
            timeout=5
        )
        print("✅ Model unloaded.\n")
    except Exception as e:
        print(f"⚠️ Failed to unload model: {e}\n")
    gc.collect()
