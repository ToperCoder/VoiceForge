import gc
import os
import sys
from llama_cpp import Llama
import config

def load_qwen():
    """
    Загружает модель Qwen через llama_cpp.
    """
    print(f"⏳ Проверка пути: {config.QWEN_MODEL_PATH}")
    if not os.path.exists(config.QWEN_MODEL_PATH):
        print(f"❌ ОШИБКА: Файл не найден!")
        sys.exit(1)
        
    n_gpu = -1 if config.USE_GPU else 0
    
    try:
        # Загрузка модели с подробными логами
        llm = Llama(
            model_path=config.QWEN_MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=4096,
            verbose=True # Снова включаем для диагностики
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
        {"role": "system", "content": "Sən Azərbaycan dili üzrə peşəkar redaktorsan. Tapşırığın: mətndəki nitq tanıma (ASR) səhvlərini düzəltməkdir (məsələn: 'Sünni intellekt' -> 'Süni intellekt', 'gözərdir' -> 'gözəldir', 'büyük dəfə' -> 'böyük töhfə'). Durğu işarələrini düzgün qoy və mətni tam təbii hala gətir. MƏSULİYYƏT: Mənanı qoru, adları saxlamağa çalış, lakin aşkar fonetik səhvləri mütləq düzəlt."},
        {"role": "user", "content": f"Aşağıdakı mətni peşəkar şəkildə redaktə et:\n\n{text}"}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=4096,
        temperature=0.0
    )
    
    result = response['choices'][0]['message']['content'].strip()
    return result

def unload_qwen(llm):
    """
    Полностью выгружает модель LLM из памяти.
    """
    print("\nUnloading Qwen...\n")
    del llm
    gc.collect()
