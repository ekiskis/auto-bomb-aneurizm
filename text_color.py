import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time
import re

# Путь к исполняемому файлу tesseract - обновите при необходимости
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Раскомментируйте и настройте для Windows

# Определим RGB-коды для обнаружения цветов проводов
COLOR_RANGES = {
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([0, 0, 0]),
        "name": "Black"
    },
    "white": {
        "lower": np.array([150, 150, 170]),
        "upper": np.array([180, 180, 201]),
        "name": "White"
    },
    "green": {
        "lower": np.array([0, 130, 0]),
        "upper": np.array([0, 255, 0]),
        "name": "Green"
    },
    "blue": {
        "lower": np.array([0, 0, 150]),
        "upper": np.array([0, 0, 255]),
        "name": "Blue"
    },
    "red": {
        "lower": np.array([130, 0, 0]),
        "upper": np.array([255, 0, 0]),
        "name": "Red"
    }
}

# Прямоугольники для проводов
WIRE_RECTANGLES = [
    (75, 154, 235, 22),
    (74, 205, 237, 26),
    (74, 256, 235, 22),
    (74, 311, 237, 17),
    (74, 363, 235, 14)
]

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

def detect_dominant_color(roi):
    """
    Определяет доминирующий цвет в области изображения (ROI)
    """
    # Усредним все пиксели в ROI для получения доминирующего цвета
    average_color = np.mean(roi, axis=(0, 1)).astype(int)
    
    # Определяем, к какому цвету ближе всего средний цвет
    detected_color = "unknown"
    min_distance = float('inf')
    
    for color, ranges in COLOR_RANGES.items():
        lower = ranges["lower"]
        upper = ranges["upper"]
        
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

def detect_wire_colors_in_rectangles(img, rectangles):
    """
    Определяет цвета проводов в заданных прямоугольных областях
    rectangles - список кортежей (x, y, width, height) для каждого провода
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
            color_name, avg_color = detect_dominant_color(roi)
            
            # Отмечаем прямоугольник на отладочном изображении
            color_bgr = (0, 255, 0)  # Зеленый цвет для рамки по умолчанию
            
            # Если цвет распознан правильно, используем его для рамки
            if color_name.lower() in COLOR_RANGES:
                color_data = COLOR_RANGES[color_name.lower()]
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
    
    # Сохраняем отладочное изображение
    # cv2.imwrite('wire_detection_debug.png', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    # Возвращаем только названия цветов в порядке позиций
    return [color_info["color"] for color_info in detected_colors], debug_img

def read_text_from_region(region=None, wire_region=None, save_debug_image=False):
    """
    Захват области экрана для OCR и области с проводами для определения цветов
    """
    # Захват всего экрана для визуализации
    full_screen = capture_full_screen()
    
    # Рисование прямоугольника на полном экране для отображения области OCR
    if region and save_debug_image:
        left, top, width, height = region
        cv2.rectangle(full_screen, (left, top), (left + width, top + height), (0, 255, 0), 2)
    
    # Рисование прямоугольника для области проводов
    if wire_region and save_debug_image:
        left, top, width, height = wire_region
        cv2.rectangle(full_screen, (left, top), (left + width, top + height), (0, 0, 255), 2)
    
    # Сохранение полного экрана с прямоугольниками
    # if save_debug_image:
    #     cv2.imwrite('full_screen_with_regions.png', cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR))
    
    # Захват области текста для OCR
    if region:
        text_img = capture_screen_region(region)
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
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
        wire_colors, debug_img = detect_wire_colors_in_rectangles(wire_img, WIRE_RECTANGLES)
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

def automated_color_check(text_region, wire_region, click_position, interval=1.0, press_interval=500):
    """
    Автоматизированная проверка цветов с автоматическими нажатиями клавиш
    
    Arguments:
        text_region: Область экрана для распознавания текста
        wire_region: Область экрана с проводами
        click_position: Кортеж (x, y) - координаты куда кликать при совпадении
        interval: Интервал между проверками в секундах
        press_interval: Задержка после нажатия клавиши E в миллисекундах
    """
    try:
        print("Запуск автоматизированной проверки цветов...")
        print("Для остановки нажмите Ctrl+C")
        
        while True:
            # Нажимаем клавишу E
            print("Нажатие клавиши E...")
            pyautogui.press('e')
            
            # Небольшая задержка для обновления экрана
            time.sleep(press_interval / 1000)
            
            # Захват и анализ экрана
            result = read_text_from_region(text_region, wire_region)
            current_text_colors = result["text_colors"]
            current_wire_colors = result["wire_colors"]
            
            print(f"Сырой текст:\n{result['raw_text']}")
            print(f"Цвета из текста: {current_text_colors}")
            print(f"Цвета проводов: {current_wire_colors}")
            
            # Если нашли последовательности цветов, анализируем их
            if current_text_colors and current_wire_colors:
                # Сравниваем последовательности
                comparison = compare_color_sequences(current_text_colors, current_wire_colors)
                
                if comparison["match"]:
                    print(f"✓ СОВПАДЕНИЕ: {comparison['reason']}")
                    print(f"Производим клик в позицию {click_position}...")
                    # Клик мышью в указанную позицию при совпадении
                    pyautogui.click(click_position[0], click_position[1])
                else:
                    print(f"✗ НЕСООТВЕТСТВИЕ: {comparison['reason']}")
                    if "differences" in comparison:
                        for diff in comparison["differences"]:
                            print(f"  Позиция {diff['position']}: текст '{diff['text_color']}', провод '{diff['wire_color']}'")
                    print("Нажатие клавиши ESC...")
                    # Нажатие ESC при несоответствии
                    pyautogui.press('esc')
            else:
                print("Не удалось обнаружить полные последовательности цветов")
                print("Нажатие клавиши ESC...")
                # Нажатие ESC при ошибке распознавания
                pyautogui.press('esc')
            
            # Ждем перед следующей проверкой
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("Автоматизированная проверка остановлена")

if __name__ == "__main__":
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Область для OCR (текст с названиями цветов)
    text_left = int(screen_width * 0.61)
    text_top = int(screen_height * 0.654)
    text_width = int(screen_width * 0.0725)
    text_height = int(screen_height * 0.145)
    
    text_region = (text_left, text_top, text_width, text_height)
    
    # Область для определения цветов проводов
    wire_left = int(screen_width * 0.2)
    wire_top = int(screen_height * 0.3)
    wire_width = int(screen_width * 0.2)
    wire_height = int(screen_height * 0.5)
    
    wire_region = (wire_left, wire_top, wire_width, wire_height)
    
    # Координаты для клика при совпадении цветов
    click_x = int(screen_width * 0.5)  # По умолчанию середина экрана
    click_y = int(screen_height * 0.7)  # Пример положения
    
    # Запрос у пользователя координат для клика
    print("=== Программа автоматизированной проверки цветов проводов ===")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Область текста: {text_region}")
    print(f"Область проводов: {wire_region}")
    print("Поддерживаемые цвета: Red, Green, Black, Blue, White")
    
    click_input = input("\nВведите координаты для клика в формате 'x,y' (или нажмите Enter для значений по умолчанию): ")
    if click_input.strip():
        try:
            click_x, click_y = map(int, click_input.split(','))
            print(f"Координаты клика установлены: ({click_x}, {click_y})")
        except:
            print(f"Неверный формат. Используются координаты по умолчанию: ({click_x}, {click_y})")
    
    # Интервал между проверками
    interval_input = input("\nВведите интервал между проверками в секундах (или нажмите Enter для значения по умолчанию 1.0): ")
    interval = 1.0
    if interval_input.strip():
        try:
            interval = float(interval_input)
            print(f"Интервал установлен: {interval} сек")
        except:
            print(f"Неверный формат. Используется интервал по умолчанию: {interval} сек")
    
    # Задержка после нажатия E
    press_interval_input = input("\nВведите задержку после нажатия клавиши E в миллисекундах (или нажмите Enter для значения по умолчанию 500): ")
    press_interval = 500
    if press_interval_input.strip():
        try:
            press_interval = int(press_interval_input)
            print(f"Задержка установлена: {press_interval} мс")
        except:
            print(f"Неверный формат. Используется задержка по умолчанию: {press_interval} мс")
    
    # Пауза перед началом автоматизации
    print("\nДля запуска автоматизированной проверки нажмите Enter...")
    print("У вас будет 5 секунд, чтобы переключиться в нужное окно.")
    input()
    
    print("Начало через 5 секунд...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # Запуск автоматизированной проверки
    automated_color_check(text_region, wire_region, (click_x, click_y), interval, press_interval)