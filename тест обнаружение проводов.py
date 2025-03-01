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
# Это примерные значения, вам нужно будет заменить их на точные RGB-коды ваших проводов
COLOR_RANGES = {
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([50, 50, 50]),
        "name": "Black"
    },
    "white": {
        "lower": np.array([170, 150, 150]),
        "upper": np.array([201, 180, 180]),
        "name": "White"
    },
    "green": {
        "lower": np.array([0, 130, 0]),
        "upper": np.array([0, 255, 0]),
        "name": "Green"
    },
    "blue": {
        "lower": np.array([180, 0, 0]),
        "upper": np.array([200, 0, 0]),
        "name": "Blue"
    },
    "red": {
        "lower": np.array([100, 0, 0]),
        "upper": np.array([255, 100, 100]),
        "name": "Red"
    }
}

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

def detect_wire_colors(img, wire_positions=None):
    """
    Определяет цвета проводов на изображении
    wire_positions - список кортежей (y, x) с координатами точек для проверки цвета
    Если wire_positions не указан, пытается автоматически определить позиции проводов
    """
    # Если позиции проводов не указаны, попробуем определить их автоматически
    if wire_positions is None:
        # Определим примерное расположение проводов (это надо настраивать под конкретное изображение)
        height, width = img.shape[:2]
        wire_count = 5  # Предполагаем 5 проводов
        
        # Вычисляем равномерно распределенные точки для проверки цветов проводов
        # Предполагается, что провода расположены вертикально по центру изображения
        y_pos = height // 2
        x_step = width // (wire_count + 1)
        
        wire_positions = []
        for i in range(1, wire_count + 1):
            wire_positions.append((y_pos, i * x_step))
    
    detected_colors = []
    debug_img = img.copy()  # Копия для отображения точек проверки
    
    # Проверяем цвет в каждой указанной позиции
    for i, (y, x) in enumerate(wire_positions):
        # Получаем RGB значение в указанной точке
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            rgb = img[y, x]
            
            # Отмечаем точку проверки на отладочном изображении
            cv2.circle(debug_img, (x, y), 5, (255, 0, 255), -1)
            cv2.putText(debug_img, f"Point {i+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Определяем цвет по RGB значению
            color_name = "unknown"
            for color, ranges in COLOR_RANGES.items():
                lower = ranges["lower"]
                upper = ranges["upper"]
                
                # Проверяем, попадает ли RGB в диапазон цвета
                if np.all(rgb >= lower) and np.all(rgb <= upper):
                    color_name = ranges["name"]
                    break
            
            detected_colors.append({
                "position": i + 1,
                "color": color_name,
                "rgb": rgb.tolist()
            })
        else:
            detected_colors.append({
                "position": i + 1,
                "color": "invalid position",
                "rgb": None
            })
    
    # Сохраняем отладочное изображение
    cv2.imwrite('wire_detection_debug.png', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    # Возвращаем только названия цветов в порядке позиций
    return [color_info["color"] for color_info in detected_colors], debug_img

def read_text_from_region(region=None, wire_region=None, save_debug_image=True):
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
    if save_debug_image:
        cv2.imwrite('full_screen_with_regions.png', cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR))
    
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
        # Определение цветов проводов
        wire_colors, debug_img = detect_wire_colors(wire_img)
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

def monitor_color_sequence(text_region, wire_region, interval=1.0, max_attempts=3):
    """
    Непрерывный мониторинг областей для отслеживания и сравнения последовательностей цветов
    """
    last_text_colors = []
    last_wire_colors = []
    consecutive_failures = 0
    
    try:
        while True:
            result = read_text_from_region(text_region, wire_region)
            current_text_colors = result["text_colors"]
            current_wire_colors = result["wire_colors"]
            
            # Если нашли последовательности цветов, анализируем их
            if current_text_colors and current_wire_colors:
                consecutive_failures = 0
                
                # Если обнаружены изменения в последовательностях
                if current_text_colors != last_text_colors or current_wire_colors != last_wire_colors:
                    print("\n--- Новые данные обнаружены ---")
                    print(f"Сырой текст:\n{result['raw_text']}")
                    print(f"Цвета из текста: {current_text_colors}")
                    print(f"Цвета проводов: {current_wire_colors}")
                    
                    # Сравниваем последовательности
                    comparison = compare_color_sequences(current_text_colors, current_wire_colors)
                    
                    if comparison["match"]:
                        print(f"✓ СОВПАДЕНИЕ: {comparison['reason']}")
                    else:
                        print(f"✗ НЕСООТВЕТСТВИЕ: {comparison['reason']}")
                        if "differences" in comparison:
                            for diff in comparison["differences"]:
                                print(f"  Позиция {diff['position']}: текст '{diff['text_color']}', провод '{diff['wire_color']}'")
                    
                    last_text_colors = current_text_colors
                    last_wire_colors = current_wire_colors
            else:
                consecutive_failures += 1
                print(f"Не удалось обнаружить полные последовательности цветов (попытка {consecutive_failures}/{max_attempts})")
                
                # Если несколько попыток подряд неудачны, делаем паузу подольше
                if consecutive_failures >= max_attempts:
                    print("Слишком много неудачных попыток подряд. Увеличиваем интервал...")
                    time.sleep(interval * 3)
                    consecutive_failures = 0
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Мониторинг остановлен")

def calibrate_color_detection(wire_region):
    """
    Инструмент для калибровки обнаружения цветов
    Позволяет кликнуть на изображении для определения RGB-кодов
    """
    print("Запуск калибровки цветов...")
    
    # Захват области с проводами
    wire_img = capture_screen_region(wire_region)
    img_copy = wire_img.copy()
    
    # Функция обработки кликов мыши
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Получение цвета пикселя
            rgb = wire_img[y, x]
            
            # Отображение цвета и координат
            print(f"Координаты: ({x}, {y}), RGB: {rgb}")
            
            # Отметка точки на изображении
            cv2.circle(img_copy, (x, y), 5, (255, 0, 255), -1)
            cv2.putText(img_copy, f"RGB: {rgb}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Обновление отображения
            cv2.imshow("Калибровка цветов", img_copy)
    
    # Создание окна и настройка обработчика событий мыши
    cv2.imshow("Калибровка цветов", wire_img)
    cv2.setMouseCallback("Калибровка цветов", mouse_callback)
    
    print("Кликните на провода для определения их RGB-кодов. Нажмите 'q' для выхода.")
    
    # Ожидание нажатия клавиши 'q' для выхода
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cv2.imwrite('color_calibration.png', cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    print("Калибровка завершена. Результаты сохранены в 'color_calibration.png'.")

if __name__ == "__main__":
    
    time.sleep(3)
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Область для OCR (текст с названиями цветов)
    text_left = int(screen_width * 0.61)
    text_top = int(screen_height * 0.654)
    text_width = int(screen_width * 0.0725)
    text_height = int(screen_height * 0.145)
    
    text_region = (text_left, text_top, text_width, text_height)
    
    # Область для определения цветов проводов
    # Это примерная область, вам нужно будет настроить её под ваше изображение
    wire_left = int(screen_width * 0.2)
    wire_top = int(screen_height * 0.3)
    wire_width = int(screen_width * 0.2)
    wire_height = int(screen_height * 0.2)
    
    wire_region = (wire_left, wire_top, wire_width, wire_height)
    
    print("=== Программа распознавания и сравнения цветов проводов ===")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Область текста: {text_region}")
    print(f"Область проводов: {wire_region}")
    print("Поддерживаемые цвета: Red, Green, Black, Blue, White")
    
    # Предложение калибровки
    # calibrate = input("Запустить калибровку цветов проводов? (y/n): ")
    # if calibrate.lower() == 'y':
    #     time.sleep(4)
    #     calibrate_color_detection(wire_region)
        
    #     # После калибровки предложение обновить цветовые диапазоны
    #     print("\nПосле калибровки обновите цветовые диапазоны в коде (COLOR_RANGES)")
    #     print("Пример:")
    #     print('    "green": {')
    #     print('        "lower": np.array([R-20, G-20, B-20]),')
    #     print('        "upper": np.array([R+20, G+20, B+20]),')
    #     print('        "name": "Green"')
    #     print('    },')
    
    # time.sleep(4)
    
    
    # Одноразовое тестирование
    print("\nТестовое чтение...")
    result = read_text_from_region(text_region, wire_region)
    
    print(f"\nРаспознанный текст:\n{result['raw_text']}")
    print(f"Цвета из текста: {result['text_colors']}")
    print(f"Цвета проводов: {result['wire_colors']}")
    
    # Сравнение последовательностей
    if result['text_colors'] and result['wire_colors']:
        comparison = compare_color_sequences(result['text_colors'], result['wire_colors'])
        
        if comparison["match"]:
            print(f"✓ СОВПАДЕНИЕ: {comparison['reason']}")
        else:
            print(f"✗ НЕСООТВЕТСТВИЕ: {comparison['reason']}")
            if "differences" in comparison:
                for diff in comparison["differences"]:
                    print(f"  Позиция {diff['position']}: текст '{diff['text_color']}', провод '{diff['wire_color']}'")
    
    
    # Спрашиваем пользователя, хочет ли он запустить непрерывный мониторинг
    # choice = input("\nЗапустить непрерывный мониторинг и сравнение последовательностей цветов? (y/n): ")
    # if choice.lower() == 'y':
    #     print("Запуск непрерывного мониторинга... (Нажмите Ctrl+C для остановки)")
    #     monitor_color_sequence(text_region, wire_region)
    # else:
    #     print("Программа завершена")