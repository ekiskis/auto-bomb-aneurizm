import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time
import re
import json
import os

# Функция для создания конфигурационного файла по умолчанию
def create_default_config():
    default_config = {
        # Путь к исполняемому файлу tesseract
        "tesseract_path": r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        
        # Интервал между проверками в секундах
        "check_interval": 1.0,
        
        # Задержка после нажатия клавиши E в миллисекундах
        "press_interval": 500,
        
        # Включение/выключение вывода логов в консоль
        "enable_logs": True,
        
        # Включение/выключение сохранения отладочных изображений
        "enable_debug_images": False,
        
        # Настройки диапазонов цветов
        "color_ranges": {
            "black": {
                "lower": [0, 0, 0],
                "upper": [0, 0, 0],
                "name": "Black"
            },
            "white": {
                "lower": [150, 150, 170],
                "upper": [180, 180, 201],
                "name": "White"
            },
            "green": {
                "lower": [0, 130, 0],
                "upper": [0, 255, 0],
                "name": "Green"
            },
            "blue": {
                "lower": [0, 0, 150],
                "upper": [0, 0, 255],
                "name": "Blue"
            },
            "red": {
                "lower": [130, 0, 0],
                "upper": [255, 0, 0],
                "name": "Red"
            }
        },
        
        # Прямоугольники для проводов
        "wire_rectangles": [
            [75, 154, 235, 22],
            [74, 205, 237, 26],
            [74, 256, 235, 22],
            [74, 311, 237, 17],
            [74, 363, 235, 14]
        ],
        
        # Область для OCR (текст с названиями цветов)
        "text_region": {
            "left_factor": 0.61,
            "top_factor": 0.654,
            "width_factor": 0.0725,
            "height_factor": 0.145
        },
        
        # Область для определения цветов проводов
        "wire_region": {
            "left_factor": 0.2,
            "top_factor": 0.3,
            "width_factor": 0.2,
            "height_factor": 0.5
        },
        
        # Координаты для клика при совпадении цветов
        "click_position": {
            "x_factor": 0.7,
            "y_factor": 0.73
        },
    }
    
    # Добавляем комментарии к конфигурационному файлу
    config_with_comments = {
        "_comment_general": "Конфигурационный файл для программы проверки цветов проводов",
        "_comment_tesseract": "Путь к исполняемому файлу Tesseract OCR",
        "tesseract_path": default_config["tesseract_path"],
        
        "_comment_colors": "Диапазоны RGB цветов для распознавания цветов проводов",
        "color_ranges": default_config["color_ranges"],
        
        "_comment_wire_rectangles": "Координаты прямоугольников для проводов в формате [x, y, ширина, высота]",
        "wire_rectangles": default_config["wire_rectangles"],
        
        "_comment_text_region": "Область экрана для распознавания текста (коэффициенты от размеров экрана)",
        "text_region": default_config["text_region"],
        
        "_comment_wire_region": "Область экрана для определения цветов проводов (коэффициенты от размеров экрана)",
        "wire_region": default_config["wire_region"],
        
        "_comment_click": "Координаты для клика при совпадении цветов (коэффициенты от размеров экрана)",
        "click_position": default_config["click_position"],
        
        "_comment_intervals": "Временные интервалы работы программы",
        "check_interval": default_config["check_interval"],
        "press_interval": default_config["press_interval"],
        
        "_comment_debugging": "Настройки логирования и отладки",
        "enable_logs": default_config["enable_logs"],
        "enable_debug_images": default_config["enable_debug_images"]
    }
    
    # Записываем конфигурацию в файл
    with open("config.json", "w", encoding="utf-8") as config_file:
        json.dump(config_with_comments, config_file, indent=4, ensure_ascii=False)
    
    print("Создан файл конфигурации config.json с параметрами по умолчанию.")
    return default_config

def load_config():
    """
    Загружает конфигурацию из файла или создает файл с конфигурацией по умолчанию
    """
    config_path = "config.json"
    
    # Проверяем существование файла конфигурации
    if not os.path.exists(config_path):
        return create_default_config()
    
    # Читаем файл конфигурации
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        
        # Удаляем комментарии из конфигурации
        clean_config = {k: v for k, v in config.items() if not k.startswith("_comment")}
        return clean_config
        
    except Exception as e:
        print(f"Ошибка при чтении файла конфигурации: {e}")
        print("Создаю конфигурацию по умолчанию...")
        return create_default_config()

def log_message(message, config):
    """
    Выводит сообщение в консоль, если включен режим логирования
    """
    if config.get("enable_logs", True):
        print(message)

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

def detect_dominant_color(roi, color_ranges):
    """
    Определяет доминирующий цвет в области изображения (ROI)
    """
    # Усредним все пиксели в ROI для получения доминирующего цвета
    average_color = np.mean(roi, axis=(0, 1)).astype(int)
    
    # Определяем, к какому цвету ближе всего средний цвет
    detected_color = "unknown"
    min_distance = float('inf')
    
    for color, ranges in color_ranges.items():
        lower = np.array(ranges["lower"])
        upper = np.array(ranges["upper"])
        
        # Проверяем, попадает ли средний цвет в диапазон
        if np.all(average_color >= lower) and np.all(average_color <= upper):
            detected_color = ranges["name"]
            break
        
        # Если не попадает в диапазон, вычисляем "расстояние" до центра диапазона
        center = (lower + upper) / 2
        distance = np.sum(np.abs(average_color - center))
        
        if distance < min_distance:
            min_distance = distance
            detected_color = ranges["name"]
    
    return detected_color, average_color.tolist()

def detect_wire_colors_in_rectangles(img, rectangles, color_ranges, config):
    """
    Определяет цвета проводов в заданных прямоугольных областях
    rectangles - список списков [x, y, width, height] для каждого провода
    """
    detected_colors = []
    debug_img = img.copy()  # Копия для отображения прямоугольников
    
    # Проверяем цвет в каждом прямоугольнике
    for i, (x, y, w, h) in enumerate(rectangles):
        # Проверяем, что координаты в пределах изображения
        if (x >= 0 and y >= 0 and 
            x + w <= img.shape[1] and 
            y + h <= img.shape[0]):
            
            # Извлекаем область изображения (ROI)
            roi = img[y:y+h, x:x+w]
            
            # Определяем доминирующий цвет в ROI
            color_name, avg_color = detect_dominant_color(roi, color_ranges)
            
            # Отмечаем прямоугольник на отладочном изображении
            color_bgr = (0, 255, 0)  # Зеленый цвет для рамки по умолчанию
            
            # Если цвет распознан правильно, используем его для рамки
            if color_name.lower() in color_ranges:
                color_data = color_ranges[color_name.lower()]
                color_bgr = (int(color_data["lower"][2]), int(color_data["lower"][1]), int(color_data["lower"][0]))
            
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(debug_img, f"{i+1}: {color_name}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
            
            detected_colors.append({
                "position": i + 1,
                "color": color_name,
                "avg_rgb": avg_color
            })
        else:
            detected_colors.append({
                "position": i + 1,
                "color": "invalid position",
                "avg_rgb": None
            })
    
    # Сохраняем отладочное изображение, если включен режим отладки
    if config.get("enable_debug_images", False):
        timestamp = int(time.time())
        cv2.imwrite(f'wire_detection_{timestamp}.png', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    # Возвращаем только названия цветов в порядке позиций
    return [color_info["color"] for color_info in detected_colors], debug_img

def read_text_from_region(region=None, wire_region=None, config=None):
    """
    Захват области экрана для OCR и области с проводами для определения цветов
    """
    if config is None:
        config = {}
    
    # Загружаем настройки цветов из конфигурации
    color_ranges = {}
    for color, data in config.get("color_ranges", {}).items():
        color_ranges[color] = {
            "lower": np.array(data["lower"]),
            "upper": np.array(data["upper"]),
            "name": data["name"]
        }
    
    # Загружаем настройки прямоугольников из конфигурации
    wire_rectangles = config.get("wire_rectangles", [])
    
    # Захват всего экрана для визуализации
    full_screen = capture_full_screen()
    
    # Рисование прямоугольника на полном экране для отображения области OCR
    if region and config.get("enable_debug_images", False):
        left, top, width, height = region
        cv2.rectangle(full_screen, (left, top), (left + width, top + height), (0, 255, 0), 2)
    
    # Рисование прямоугольника для области проводов
    if wire_region and config.get("enable_debug_images", False):
        left, top, width, height = wire_region
        cv2.rectangle(full_screen, (left, top), (left + width, top + height), (0, 0, 255), 2)
    
    # Сохранение полного экрана с прямоугольниками
    if config.get("enable_debug_images", False):
        timestamp = int(time.time())
        cv2.imwrite(f'full_screen_with_regions_{timestamp}.png', cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR))
    
    # Захват области текста для OCR
    if region:
        text_img = capture_screen_region(region)
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
        
        # Сохраняем изображение для OCR, если включен режим отладки
        if config.get("enable_debug_images", False):
            timestamp = int(time.time())
            cv2.imwrite(f'ocr_region_{timestamp}.png', gray)
        
        # Распознавание текста
        text = pytesseract.image_to_string(gray, config='--psm 6')
        # Извлечение цветов из текста
        text_colors = extract_colors(text)
    else:
        text = ""
        text_colors = []
    
    # Захват области с проводами для определения цветов
    if wire_region:
        wire_img = capture_screen_region(wire_region)
        # Определение цветов проводов используя прямоугольники
        wire_colors, debug_img = detect_wire_colors_in_rectangles(wire_img, wire_rectangles, color_ranges, config)
    else:
        wire_img = None
        wire_colors = []
        debug_img = None
    
    return {
        "raw_text": text.strip(),
        "text_colors": text_colors,
        "wire_colors": wire_colors,
        "debug_image": debug_img
    }

def compare_color_sequences(text_colors, wire_colors):
    """
    Сравнивает две последовательности цветов и выдает результат
    """
    # Нормализуем названия цветов для сравнения
    normalized_text_colors = [color.lower() for color in text_colors]
    normalized_wire_colors = [color.lower() for color in wire_colors]
    
    # Проверяем длину
    if len(normalized_text_colors) != len(normalized_wire_colors):
        return {
            "match": False,
            "reason": f"Разная длина последовательностей: текст ({len(normalized_text_colors)}) vs провода ({len(normalized_wire_colors)})"
        }
    
    # Проверяем совпадение цветов
    differences = []
    for i, (text_color, wire_color) in enumerate(zip(normalized_text_colors, normalized_wire_colors)):
        if text_color != wire_color:
            differences.append({
                "position": i + 1,
                "text_color": text_color,
                "wire_color": wire_color
            })
    
    if differences:
        return {
            "match": False,
            "differences": differences,
            "reason": f"Найдены несоответствия в {len(differences)} позициях"
        }
    else:
        return {
            "match": True,
            "reason": "Последовательности цветов полностью совпадают"
        }

def automated_color_check(config):
    """
    Автоматизированная проверка цветов с автоматическими нажатиями клавиш
    
    Arguments:
        config: Конфигурация программы
    """
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Параметры из конфигурации
    text_region_config = config.get("text_region", {})
    wire_region_config = config.get("wire_region", {})
    click_position_config = config.get("click_position", {})
    
    # Расчет конкретных координат на основе коэффициентов
    text_left = int(screen_width * text_region_config.get("left_factor", 0.61))
    text_top = int(screen_height * text_region_config.get("top_factor", 0.654))
    text_width = int(screen_width * text_region_config.get("width_factor", 0.0725))
    text_height = int(screen_height * text_region_config.get("height_factor", 0.145))
    
    text_region = (text_left, text_top, text_width, text_height)
    
    wire_left = int(screen_width * wire_region_config.get("left_factor", 0.2))
    wire_top = int(screen_height * wire_region_config.get("top_factor", 0.3))
    wire_width = int(screen_width * wire_region_config.get("width_factor", 0.2))
    wire_height = int(screen_height * wire_region_config.get("height_factor", 0.5))
    
    wire_region = (wire_left, wire_top, wire_width, wire_height)
    
    click_x = int(screen_width * click_position_config.get("x_factor", 0.7))
    click_y = int(screen_height * click_position_config.get("y_factor", 0.73))
    
    interval = config.get("check_interval", 1.0)
    press_interval = config.get("press_interval", 500)
    
    try:
        print("Запуск автоматизированной проверки цветов...")
        print("Для остановки нажмите Ctrl+C")
        
        while True:
            # Нажимаем клавишу E
            log_message("Нажатие клавиши E...", config)
            pyautogui.press('e')
            
            # Небольшая задержка для обновления экрана
            time.sleep(press_interval / 1000)
            
            # Захват и анализ экрана
            result = read_text_from_region(text_region, wire_region, config)
            current_text_colors = result["text_colors"]
            current_wire_colors = result["wire_colors"]
            
            log_message(f"Сырой текст:\n{result['raw_text']}", config)
            log_message(f"Цвета из текста: {current_text_colors}", config)
            log_message(f"Цвета проводов: {current_wire_colors}", config)
            
            # Если нашли последовательности цветов, анализируем их
            if current_text_colors and current_wire_colors:
                # Сравниваем последовательности
                comparison = compare_color_sequences(current_text_colors, current_wire_colors)
                
                if comparison["match"]:
                    log_message(f"✓ СОВПАДЕНИЕ: {comparison['reason']}", config)
                    log_message(f"Производим клик в позицию ({click_x}, {click_y})...", config)
                    # Клик мышью в указанную позицию при совпадении
                    pyautogui.click(click_x, click_y)
                else:
                    log_message(f"✗ НЕСООТВЕТСТВИЕ: {comparison['reason']}", config)
                    if "differences" in comparison:
                        for diff in comparison["differences"]:
                            log_message(f"  Позиция {diff['position']}: текст '{diff['text_color']}', провод '{diff['wire_color']}'", config)
                    log_message("Нажатие клавиши ESC...", config)
                    # Нажатие ESC при несоответствии
                    pyautogui.press('esc')
            else:
                log_message("Не удалось обнаружить полные последовательности цветов либо расшифровать текст", config)
                log_message("Нажатие клавиши ESC...", config)
                # Нажатие ESC при ошибке распознавания
                pyautogui.press('esc')
            
            # Ждем перед следующей проверкой
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("Автоматизированная проверка остановлена")

if __name__ == "__main__":
    # Загрузка конфигурации
    config = load_config()
    
    # Настройка пути к Tesseract
    if "tesseract_path" in config:
        pytesseract.pytesseract.tesseract_cmd = config["tesseract_path"]
    
    # Запуск автоматизированной проверки
    automated_color_check(config)