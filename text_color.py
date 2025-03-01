import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time
import re
import os
import configparser
import ast

def create_default_config():
    """
    Создает конфигурационный файл по умолчанию в формате INI
    """
    # Создаем ConfigParser с сохранением регистра
    config = configparser.ConfigParser(empty_lines_in_values=False)
    # Добавляем настройку для сохранения регистра
    config.optionxform = str
    
    # Основные настройки
    config['General'] = {}
    # Добавляем комментарии как отдельные элементы
    config['General']['# ВАЖНЫЕ НАСТРОЙКИ'] = '#'
    config['General']['# Путь к исполняемому файлу Tesseract OCR'] = '#'
    config['General']['tesseract_path'] = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    config['General']['# Интервал между проверками в секундах'] = '#'
    config['General']['check_interval'] = '1.0'
    config['General']['# Задержка после нажатия клавиши E в миллисекундах'] = '#'
    config['General']['press_interval'] = '500'
    config['General']['# Включение/выключение вывода логов в консоль (True/False)'] = '#'
    config['General']['enable_logs'] = 'True'
    config['General']['# Включение/выключение сохранения отладочных изображений (True/False)'] = ''
    config['General']['enable_debug_images'] = 'False'
    config['General']['# НИЖЕ ИДУТ СКОРЕЕ ВСЕГО ВАМ НЕ НУЖНЫЕ НАСТРОЙКИ'] = '#'
    
    # Координаты областей (коэффициенты от размеров экрана)
    config['Regions'] = {}
    config['Regions']['# Область текста с названиями цветов'] = '#'
    config['Regions']['text_left_factor'] = '0.61'
    config['Regions']['text_top_factor'] = '0.654'
    config['Regions']['text_width_factor'] = '0.0725'
    config['Regions']['text_height_factor'] = '0.145'
    config['Regions']['# Область с проводами'] = '#'
    config['Regions']['wire_left_factor'] = '0.2'
    config['Regions']['wire_top_factor'] = '0.3'
    config['Regions']['wire_width_factor'] = '0.2'
    config['Regions']['wire_height_factor'] = '0.5'
    config['Regions']['# Координаты для клика при совпадении'] = '#'
    config['Regions']['click_x_factor'] = '0.7'
    config['Regions']['click_y_factor'] = '0.73'
    
    # Прямоугольники для проводов
    config['WireRectangles'] = {}
    config['WireRectangles']['# Прямоугольники для проводов в формате [x, y, ширина, высота]'] = '#'
    config['WireRectangles']['rect1'] = '[75, 154, 235, 22]'
    config['WireRectangles']['rect2'] = '[74, 205, 237, 26]'
    config['WireRectangles']['rect3'] = '[74, 256, 235, 22]'
    config['WireRectangles']['rect4'] = '[74, 311, 237, 17]'
    config['WireRectangles']['rect5'] = '[74, 363, 235, 14]'
    
    # Диапазоны цветов
    config['ColorRanges'] = {}
    config['ColorRanges']['# Диапазоны RGB для цветов в формате [R, G, B]'] = '#'
    
    config['ColorRanges']['# Black'] = '#'
    config['ColorRanges']['black_lower'] = '[0, 0, 0]'
    config['ColorRanges']['black_upper'] = '[0, 0, 0]'
    config['ColorRanges']['black_name'] = 'Black'
    
    config['ColorRanges']['# White'] = '#'
    config['ColorRanges']['white_lower'] = '[150, 150, 170]'
    config['ColorRanges']['white_upper'] = '[180, 180, 201]'
    config['ColorRanges']['white_name'] = 'White'
    
    config['ColorRanges']['# Green'] = '#'
    config['ColorRanges']['green_lower'] = '[0, 130, 0]'
    config['ColorRanges']['green_upper'] = '[0, 255, 0]'
    config['ColorRanges']['green_name'] = 'Green'
    
    config['ColorRanges']['# Blue'] = '#'
    config['ColorRanges']['blue_lower'] = '[0, 0, 150]'
    config['ColorRanges']['blue_upper'] = '[0, 0, 255]'
    config['ColorRanges']['blue_name'] = 'Blue'
    
    config['ColorRanges']['# Red'] = '#'
    config['ColorRanges']['red_lower'] = '[130, 0, 0]'
    config['ColorRanges']['red_upper'] = '[255, 0, 0]'
    config['ColorRanges']['red_name'] = 'Red'
    
    # Записываем конфигурацию в файл
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)
    
    print("Создан файл конфигурации config.ini с параметрами по умолчанию.")
    
    # Преобразуем в словарь для использования в программе
    return config_to_dict(config)

def config_to_dict(config):
    """
    Преобразует объект configparser.ConfigParser в словарь с правильными типами данных
    """
    result = {}
    
    # Обработка основных настроек
    if 'General' in config:
        result['tesseract_path'] = config['General'].get('tesseract_path', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        result['check_interval'] = config['General'].getfloat('check_interval', 1.0)
        result['press_interval'] = config['General'].getint('press_interval', 500)
        result['enable_logs'] = config['General'].getboolean('enable_logs', True)
        result['enable_debug_images'] = config['General'].getboolean('enable_debug_images', False)
    
    # Обработка областей экрана
    if 'Regions' in config:
        # Область текста
        result['text_region'] = {
            'left_factor': config['Regions'].getfloat('text_left_factor', 0.61),
            'top_factor': config['Regions'].getfloat('text_top_factor', 0.654),
            'width_factor': config['Regions'].getfloat('text_width_factor', 0.0725),
            'height_factor': config['Regions'].getfloat('text_height_factor', 0.145)
        }
        
        # Область проводов
        result['wire_region'] = {
            'left_factor': config['Regions'].getfloat('wire_left_factor', 0.2),
            'top_factor': config['Regions'].getfloat('wire_top_factor', 0.3),
            'width_factor': config['Regions'].getfloat('wire_width_factor', 0.2),
            'height_factor': config['Regions'].getfloat('wire_height_factor', 0.5)
        }
        
        # Координаты для клика
        result['click_position'] = {
            'x_factor': config['Regions'].getfloat('click_x_factor', 0.7),
            'y_factor': config['Regions'].getfloat('click_y_factor', 0.73)
        }
    
    # Обработка прямоугольников проводов
    if 'WireRectangles' in config:
        wire_rectangles = []
        for i in range(1, 10):  # 10 проводов максимум
            rect_key = f'rect{i}'
            if rect_key in config['WireRectangles']:
                try:
                    rect = ast.literal_eval(config['WireRectangles'][rect_key])
                    if isinstance(rect, list) and len(rect) == 4:
                        wire_rectangles.append(rect)
                except (SyntaxError, ValueError):
                    pass  # Пропускаем неправильно форматированные значения
        
        result['wire_rectangles'] = wire_rectangles
    
    # Обработка диапазонов цветов
    if 'ColorRanges' in config:
        color_ranges = {}
        for color in ['black', 'white', 'green', 'blue', 'red']:
            lower_key = f'{color}_lower'
            upper_key = f'{color}_upper'
            name_key = f'{color}_name'
            
            if lower_key in config['ColorRanges'] and upper_key in config['ColorRanges'] and name_key in config['ColorRanges']:
                try:
                    lower = ast.literal_eval(config['ColorRanges'][lower_key])
                    upper = ast.literal_eval(config['ColorRanges'][upper_key])
                    name = config['ColorRanges'][name_key]
                    
                    if isinstance(lower, list) and isinstance(upper, list) and len(lower) == 3 and len(upper) == 3:
                        color_ranges[color] = {
                            'lower': lower,
                            'upper': upper,
                            'name': name
                        }
                except (SyntaxError, ValueError):
                    pass  # Пропускаем неправильно форматированные значения
        
        result['color_ranges'] = color_ranges
    
    return result

def load_config():
    """
    Загружает конфигурацию из INI-файла или создает файл с конфигурацией по умолчанию
    """
    config_path = 'config.ini'
    
    # Проверяем существование файла конфигурации
    if not os.path.exists(config_path):
        return create_default_config()
    
    # Читаем файл конфигурации
    try:
        config = configparser.ConfigParser()
        # Добавляем настройку для сохранения регистра
        config.optionxform = str
        config.read(config_path, encoding='utf-8')
        
        # Преобразуем конфигурацию в словарь
        return config_to_dict(config)
        
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
        input("Нажмите любую клавишу что бы запустить")
        print("Запуск авто-работы на заводе")
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
        print("\nАвтоматизированная проверка остановлена")

if __name__ == "__main__":
    # Загрузка конфигурации
    config = load_config()
    
    # Настройка пути к Tesseract
    if "tesseract_path" in config:
        pytesseract.pytesseract.tesseract_cmd = config["tesseract_path"]
    
    # Запуск автоматизированной проверки
    automated_color_check(config)