import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time

# Путь к исполняемому файлу tesseract - обновите при необходимости
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Раскомментируйте и настройте для Windows

def capture_full_screen():
    """
    Захват всего экрана
    """
    screenshot = pyautogui.screenshot()
    return np.array(screenshot)

def capture_screen_region(region=None):
    """
    Захват определенной области экрана
    region - это кортеж (left, top, width, height)
    Если region равен None, захватывает весь экран
    """
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)

def preprocess_image_methods(image):
    """
    Применяет различные методы предобработки изображения и сохраняет их все
    Возвращает список обработанных изображений
    """
    processed_images = []
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Метод 1: Простое преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_images.append(("gray", gray))
    
    # Метод 2: Адаптивное пороговое значение
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    processed_images.append(("adaptive_thresh", adaptive_thresh))
    
    # Метод 3: Обычное пороговое значение с инверсией (белый текст на черном фоне)
    _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(("binary_inv", binary_inv))
    
    # Метод 4: Обычное пороговое значение (черный текст на белом фоне)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    processed_images.append(("binary", binary))
    
    # Метод 5: Отсо пороговое значение (автоматически выбирает оптимальное значение)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(("otsu", otsu))
    
    # Метод 6: Увеличение контрастности
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    _, contrast_thresh = cv2.threshold(contrast, 150, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(("contrast", contrast_thresh))
    
    # Метод 7: Морфологические операции для очистки шума
    kernel = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    processed_images.append(("opening", opening))
    
    # Метод 8: Масштабирование изображения (увеличение в 2 раза)
    height, width = gray.shape
    scaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    _, scaled_thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(("scaled", scaled_thresh))
    
    return processed_images

def read_text_from_region(region=None, save_debug_image=True):
    """
    Захват области экрана и извлечение текста из нее,
    используя различные методы предобработки
    """
    # Захват всего экрана для визуализации
    full_screen = capture_full_screen()
    
    # Рисование прямоугольника на полном экране для отображения области
    if region and save_debug_image:
        left, top, width, height = region
        cv2.rectangle(full_screen, (left, top), (left + width, top + height), (0, 255, 0), 2)
        # Сохранение полного экрана с прямоугольником
        cv2.imwrite('full_screen_with_region.png', cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR))
    
    # Захват только указанной области для OCR
    if region:
        img = capture_screen_region(region)
    else:
        img = full_screen
    
    # Сохранение оригинального вырезанного региона
    cv2.imwrite('original_region.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Применение различных методов предобработки
    processed_images = preprocess_image_methods(img)
    
    results = []
    
    for name, processed_img in processed_images:
        # Сохранение обработанного изображения
        cv2.imwrite(f'processed_{name}.png', processed_img)
        
        # Распознавание текста с разными настройками tesseract
        configs = [
            '--psm 6',  # Предполагаем один блок текста
            '--psm 7',  # Предполагаем один строка текста
            '--psm 8',  # Предполагаем одно слово
            '--psm 4',  # Предполагаем одностолбцовый текст с переносами строк
            '--psm 3 -l eng'  # Предполагаем произвольный текст, указываем английский язык
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(processed_img, config=config)
                cleaned_text = text.strip()
                if cleaned_text:  # Если текст не пустой
                    results.append({
                        "method": name,
                        "config": config,
                        "text": cleaned_text
                    })
            except Exception as e:
                print(f"Ошибка при обработке {name} с {config}: {e}")
    
    return results

def find_best_result(results, keywords=["Red", "Red", "Green", "Green", "Blue"]):
    """
    Пытается найти лучший результат из списка результатов,
    основываясь на наличии ключевых слов
    """
    if not results:
        return None
    
    # Сначала смотрим, есть ли результаты, содержащие все ключевые слова
    best_matches = []
    for result in results:
        text = result["text"]
        score = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        result["score"] = score
        best_matches.append(result)
    
    # Сортируем по оценке (сколько ключевых слов содержится)
    best_matches.sort(key=lambda x: x["score"], reverse=True)
    
    return best_matches[0] if best_matches else None

if __name__ == "__main__":
    time.sleep(2)
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Определение области с указанными вами значениями
    left = int(screen_width * 0.61)
    top = int(screen_height * 0.654)
    width = int(screen_width * 0.0725)
    height = int(screen_height * 0.145)
    
    region = (left, top, width, height)
    
    print("Начало обнаружения текста...")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Отслеживаемая область: {region}")
    
    # Получение всех результатов с разными методами обработки
    results = read_text_from_region(region)
    
    # Вывод всех результатов
    print(f"Всего получено {len(results)} вариантов распознавания")
    
    for i, result in enumerate(results):
        print(f"\nВариант #{i+1}:")
        print(f"Метод обработки: {result['method']}")
        print(f"Конфигурация Tesseract: {result['config']}")
        print(f"Распознанный текст:\n{result['text']}")
    
    # Поиск лучшего результата
    best_result = find_best_result(results)
    if best_result:
        print("\n" + "="*50)
        print(f"Лучший результат (содержит больше всего ключевых слов):")
        print(f"Метод обработки: {best_result['method']}")
        print(f"Конфигурация Tesseract: {best_result['config']}")
        print(f"Распознанный текст:\n{best_result['text']}")
        print(f"Оценка (количество найденных ключевых слов): {best_result['score']}")
    else:
        print("\nНе удалось найти подходящий результат")