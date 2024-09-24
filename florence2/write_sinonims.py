import os
import zipfile
import json
import urllib.parse
from googletrans import Translator
from tqdm import tqdm





# тут просто все категории из моих jsonl сохранятся в файл





# Путь к исходной папке с архивами
source_dir = "/home/imran-nasyrov/cvat/"

# Путь к папке, куда будут распаковываться архивы
destination_dir = "/home/imran-nasyrov/cvat_unzip/"

# Файл для записи уже обработанных папок
processed_dirs_file = "/home/imran-nasyrov/SCT/florence2/processed_dirs.txt"

# Файл для записи категорий и их архивов
categories_file = "/home/imran-nasyrov/SCT/florence2/categories_with_archives.txt"

# Инициализация переводчика
translator = Translator()
translator.raise_exception = True  # Исправляем с 'raise_Exception'

# Проверяем, существует ли целевая директория, если нет — создаем её
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Загружаем список уже обработанных папок
if os.path.exists(processed_dirs_file):
    with open(processed_dirs_file, "r") as f:
        processed_dirs = set(line.strip() for line in f)
else:
    processed_dirs = set()

# Словарь для хранения категорий и связанных с ними архивов
categories_dict = {}

# Функция для декодирования и сбора категорий на русском языке
def decode_and_collect_categories(categories, archive_name):
    for category in categories:
        # Декодирование имени категории с UTF-8
        decoded_name = urllib.parse.unquote(category["name"])

        # Проверка, что категория уже на русском (если есть кириллица, то это русский)
        if any("а" <= char <= "я" for char in decoded_name.lower()):
            russian_name = decoded_name
        else:
            russian_name = translator.translate(decoded_name, src="en", dest="ru").text

        # Добавляем категорию в словарь, если ее нет, или обновляем список архивов
        if russian_name in categories_dict:
            categories_dict[russian_name].append(archive_name)
        else:
            categories_dict[russian_name] = [archive_name]


# Проходимся по всем файлам в исходной папке
with tqdm(total=len(os.listdir(source_dir)), desc="Processing directories") as pbar_dirs:
    for filename in os.listdir(source_dir):
        if filename.endswith(".zip"):
            folder_name = filename[:-4]  # Убираем '.zip' из имени файла

            # Пропускаем папки, которые уже были обработаны
            if folder_name in processed_dirs:
                print(f"Папка {folder_name} уже обработана, пропускаем.")
                pbar_dirs.update(1)
                continue

            file_path = os.path.join(source_dir, filename)
            folder_path = os.path.join(destination_dir, folder_name)

            # Создаем папку для распакованных файлов
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Распаковываем архив в созданную папку
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(folder_path)

            print(f"Архив {filename} успешно распакован в {folder_path}")

            # Путь к файлу instances_default.json
            json_file_path = os.path.join(
                folder_path, "annotations", "instances_default.json"
            )

            # Проверяем, существует ли файл
            if os.path.exists(json_file_path):
                # Открываем и читаем JSON файл
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)

                # Декодируем и собираем категории
                try:
                    decode_and_collect_categories(data["categories"], filename)

                    # Добавляем папку в список обработанных
                    with open(processed_dirs_file, "a") as f:
                        f.write(folder_name + "\n")
                    processed_dirs.add(folder_name)

                except Exception as e:
                    print(f"Ошибка при обработке файла {json_file_path}: {e}")
                    break  # Прерываем выполнение при ошибке

            pbar_dirs.update(1)

# Запись категорий и архивов в файл
with open(categories_file, "w", encoding="utf-8") as f:
    for category, archives in categories_dict.items():
        # Записываем архивы через запятую
        # f.write(f"{', '.join(archives)}\n")
        # Затем записываем категорию
        f.write(f"{category}\n")

print("Процесс завершен.")
