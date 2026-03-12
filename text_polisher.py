import gc
import os
import sys
import utils
# Setup DLL paths BEFORE importing llama_cpp
utils.setup_cuda_path()

from llama_cpp import Llama
import config

def load_qwen():
    """
    Загружает модель Qwen через llama_cpp.
    """
    if not config.QWEN_MODEL_PATH.exists():
        print(f"❌ ОШИБКА: Файл не найден!")
        sys.exit(1)

        
    n_gpu = -1 if config.USE_GPU else 0
    
    try:
        # Загрузка модели (verbose=False для чистых логов)
        llm = Llama(
            model_path=config.QWEN_MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=4096,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"\n❌ ОШИБКА Llama: {e}")
        print("Посмотрите логи выше для поиска 'unknown model architecture'")
        sys.exit(1)

def polish_text(llm: Llama, text: str) -> str:
    """
    Полирует сохраненный текст, исправляя ошибки распознавания, используя Chat-режим Qwen.
    """
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

    # Если текст слишком короткий, полировка может только навредить или занять лишнее время
    if len(text.strip()) < 5:
        return text

    # Уменьшаем max_tokens для скорости и предотвращения галлюцинаций
    # Для полировки достаточно длины исходного текста + запас
    max_tokens_polish = min(2048, len(text) * 2 + 100)

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens_polish,
            temperature=0.1, # Немного повышаем с 0.0 для стабильности, но оставляем низким
            top_p=0.9,
            repeat_penalty=1.1
        )
        
        result = response['choices'][0]['message']['content'].strip()
        
        # Очистка от возможных остатков промпта или шаблона
        return utils.clean_llm_output(result)
    except Exception as e:
        print(f"⚠️ Ошибка при полировке: {e}")
        return text

def unload_qwen(llm):
    """
    Полностью выгружает модель LLM из памяти.
    """
    print("\n[DEBUG] Освобождение памяти Qwen...")
    if llm:
        try:
            # llama-cpp-python автоматически освобождает память при удалении объекта
            del llm
        except:
            pass
    gc.collect()
    print("✅ Память Qwen очищена.\n")
