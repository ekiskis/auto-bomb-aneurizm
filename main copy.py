import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time
import re

# Путь к исполняемому файлу tesseract - обновите при необходимости
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Раскомментируйте и настройте для Windowsn

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

def extract_colors(text):
    """
    Извлекает названия цветов из текста
    Поддерживаемые цвета: Red, Green, Black, Blue, White
    Возвращает список найденных цветов в порядке их появления
    """
    # Преобразуем текст к нижнему регистру для упрощения поиска
    text_lower = text.lower()
    
    # Список поддерживаемых цветов
    colors = ["red", "green", "black", "blue", "white"]
    
    # Найдем все вхождения цветов в тексте
    found_colors = []
    
    # Используем регулярные выражения для поиска цветов
    for color in colors:
        # Ищем все вхождения данного цвета
        matches = re.finditer(r'\b' + color + r'\b', text_lower)
        
        # Добавляем все найденные вхождения с их позициями
        for match in matches:
            found_colors.append({
                "color": color,
                "position": match.start(),
                "original": text[match.start():match.end()]  # Оригинальный регистр из текста
            })
    
    # Сортируем найденные цвета по их позиции в тексте
    found_colors.sort(key=lambda x: x["position"])
    
    # Возвращаем только названия цветов в правильном порядке
    return [color_info["original"] for color_info in found_colors]

def read_text_from_region(region=None, save_debug_image=True):
    """
    Захват области экрана и извлечение текста с цветами из нее
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
    # cv2.imwrite('original_region.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Преобразование в оттенки серого (метод, который показал лучшие результаты)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('processed_gray.png', gray)
    
    # Распознавание текста с Tesseract в режиме --psm 6
    text = pytesseract.image_to_string(gray, config='--psm 6')
    
    # Извлечение цветов из распознанного текста
    colors = extract_colors(text)
    
    return {
        "raw_text": text.strip(),
        "colors": colors
    }

def monitor_color_sequence(region, interval=1.0, max_attempts=3):
    """
    Непрерывный мониторинг области для отслеживания изменений в последовательности цветов
    """
    last_colors = []
    consecutive_failures = 0
    
    try:
        while True:
            result = read_text_from_region(region)
            current_colors = result["colors"]
            
            # Если нашли хотя бы один цвет, сбрасываем счетчик неудач
            if current_colors:
                consecutive_failures = 0
                
                if current_colors != last_colors:
                    print("\nОбнаружена новая последовательность цветов:")
                    print(f"Сырой текст:\n{result['raw_text']}")
                    print(f"Распознанные цвета: {current_colors}")
                    
                    # Проверка на наличие ровно 5 цветов
                    if len(current_colors) == 5:
                        print("✓ Последовательность корректна (5 цветов)")
                    else:
                        print(f"⚠ Внимание: обнаружено {len(current_colors)} цветов вместо 5")
                    
                    last_colors = current_colors
            else:
                consecutive_failures += 1
                print(f"Не удалось обнаружить цвета (попытка {consecutive_failures}/{max_attempts})")
                
                # Если несколько попыток подряд неудачны, сделаем паузу подольше
                if consecutive_failures >= max_attempts:
                    print("Слишком много неудачных попыток подряд. Увеличиваем интервал...")
                    time.sleep(interval * 3)  # Увеличенный интервал для восстановления
                    consecutive_failures = 0
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Мониторинг остановлен")

if __name__ == "__main__":
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Определение области с указанными вами значениями
    left = int(screen_width * 0.61)
    top = int(screen_height * 0.654)
    width = int(screen_width * 0.0725)
    height = int(screen_height * 0.145)
    
    region = (left, top, width, height)
    
    print("Начало обнаружения цветовой последовательности...")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Отслеживаемая область: {region}")
    print("Поддерживаемые цвета: Red, Green, Black, Blue, White")
    print("Нажмите Ctrl+C для остановки")
    
    # Для одноразового чтения:
    result = read_text_from_region(region)
    print(f"\nРаспознанный текст:\n{result['raw_text']}")
    print(f"Обнаруженные цвета: {result['colors']}")
    
    # Спрашиваем пользователя, хочет ли он запустить непрерывный мониторинг
    choice = input("\nЗапустить непрерывный мониторинг последовательности цветов? (y/n): ")
    if choice.lower() == 'y':
        print("Запуск непрерывного мониторинга...")
        monitor_color_sequence(region)
    else:
        print("Программа завершена")