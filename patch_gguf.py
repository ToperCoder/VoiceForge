import os

# Путь к модели
model_path = r'C:\Users\user\Documents\GitHub\VoiceForge\models\Qwen-AzE.i1-Q6_K.gguf'
backup_path = model_path + '.bak'

if not os.path.exists(model_path):
    print(f"Файл не найден: {model_path}")
    exit(1)

# Делаем бекап 
if not os.path.exists(backup_path):
    print("Создаем резервную копию...")
    import shutil
    shutil.copy2(model_path, backup_path)

print("Начинаем патчинг метаданных...")

# Читаем только начало файла (метаданные обычно в первых нескольких мегабайтах)
# Для безопасности пропатчим первый 1 ГБ, так как GGUF может иметь много тензоров
# Но на самом деле метаданные всегда в самом начале.
with open(model_path, 'rb') as f:
    header = f.read(10 * 1024 * 1024) # 10 MB должно хватить с головой для любых метаданных

# Ищем 'qwen3' (в кодировке utf-8)
old_bytes = b'qwen3'
new_bytes = b'qwen2'

count = header.count(old_bytes)
if count == 0:
    print("Строка 'qwen3' не найдена. Возможно, файл уже пропатчен или имеет другую структуру.")
else:
    print(f"Найдено вхождений: {count}. Заменяем...")
    new_header = header.replace(old_bytes, new_bytes)
    
    # Записываем обратно только измененную часть
    with open(model_path, 'r+b') as f:
        f.write(new_header)
    print("Патчинг завершен успешно!")
