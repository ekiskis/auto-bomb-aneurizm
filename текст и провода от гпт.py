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
        "upper": np.array([50, 50, 50]),
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


# Прямоугольники для проводов (будет заполнено в main)
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
    """Определяет цвет, который встречается в прямоугольнике хотя бы один раз."""
    for pixel in roi.reshape(-1, 3):
        b, g, r = pixel  # BGR -> RGB
        rgb = np.array([r, g, b])
        
        for color, ranges in COLOR_RANGES.items():
            if np.all(rgb >= ranges["lower"]) and np.all(rgb <= ranges["upper"]):
                return ranges["name"], rgb.tolist()
    
    return "unknown", None

def detect_wire_colors(img, rectangles):
    """Обнаружение цветов проводов в заданных прямоугольниках."""
    detected_colors = []
    debug_img = img.copy()
    
    for i, (x, y, w, h) in enumerate(rectangles):
        roi = img[y:y+h, x:x+w]
        color_name, _ = detect_dominant_color(roi)
        
        detected_colors.append({"position": i + 1, "color": color_name})
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
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
        # Определение цветов проводов используя прямоугольники
        wire_colors, debug_img = detect_wire_colors(wire_img, WIRE_RECTANGLES)
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

def setup_wire_rectangles(wire_region):
    """
    Инструмент для настройки прямоугольников проводов
    Позволяет пользователю нарисовать прямоугольники на изображении
    """
    print("Запуск настройки прямоугольников для проводов...")
    
    # Захват области с проводами
    wire_img = capture_screen_region(wire_region)
    img_copy = wire_img.copy()
    
    # Глобальные переменные для отслеживания рисования прямоугольников
    drawing = False
    start_x, start_y = -1, -1
    rectangles = []
    
    # Функция обработки событий мыши
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_x, start_y, img_copy, rectangles
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Начало рисования прямоугольника
            drawing = True
            start_x, start_y = x, y
            img_copy = wire_img.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Обновление изображения при движении мыши
            if drawing:
                temp_img = img_copy.copy()
                cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow("Настройка прямоугольников", temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Завершение рисования прямоугольника
            drawing = False
            
            # Убедимся, что координаты правильные
            rx = min(start_x, x)
            ry = min(start_y, y)
            rw = abs(x - start_x)
            rh = abs(y - start_y)
            
            # Добавляем прямоугольник в список
            rect = (rx, ry, rw, rh)
            rectangles.append(rect)
            
            # Рисуем финальный прямоугольник и номер
            cv2.rectangle(img_copy, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(img_copy, f"{len(rectangles)}", (rx, ry-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Настройка прямоугольников", img_copy)
            print(f"Прямоугольник {len(rectangles)} добавлен: {rect}")
    
    # Создание окна и настройка обработчика событий мыши
    cv2.imshow("Настройка прямоугольников", wire_img)
    cv2.setMouseCallback("Настройка прямоугольников", mouse_callback)
    
    print("Нарисуйте прямоугольники вокруг проводов. Нажмите 'r' для сброса, 'c' для очистки последнего, 's' для сохранения, 'q' для выхода.")
    
    # Основной цикл обработки клавиш
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Выход без сохранения
            print("Выход без сохранения")
            rectangles = []
            break
            
        elif key == ord('r'):
            # Сброс всех прямоугольников
            rectangles = []
            img_copy = wire_img.copy()
            cv2.imshow("Настройка прямоугольников", img_copy)
            print("Все прямоугольники удалены")
            
        elif key == ord('c'):
            # Удаление последнего прямоугольника
            if rectangles:
                rectangles.pop()
                
                # Перерисовка всех оставшихся прямоугольников
                img_copy = wire_img.copy()
                for i, (rx, ry, rw, rh) in enumerate(rectangles):
                    cv2.rectangle(img_copy, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"{i+1}", (rx, ry-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("Настройка прямоугольников", img_copy)
                print(f"Последний прямоугольник удален. Осталось: {len(rectangles)}")
        
        elif key == ord('s'):
            # Сохранение и выход
            if rectangles:
                print("Прямоугольники сохранены:")
                for i, rect in enumerate(rectangles):
                    print(f"  Провод {i+1}: {rect}")
                break
            else:
                print("Нет прямоугольников для сохранения")
    
    cv2.destroyAllWindows()
    
    # Сохранение изображения с прямоугольниками
    if rectangles:
        cv2.imwrite('wire_rectangles.png', cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        print("Настройка завершена. Результаты сохранены в 'wire_rectangles.png'.")
    
    return rectangles

def get_color_calibration_from_rectangles(wire_region, rectangles):
    """
    Калибровка цветовых диапазонов на основе определенных прямоугольников
    """
    print("Калибровка цветовых диапазонов...")
    
    # Захват области с проводами
    wire_img = capture_screen_region(wire_region)
    color_samples = {}
    
    # Получаем средний цвет для каждого прямоугольника
    for i, (x, y, w, h) in enumerate(rectangles):
        # Извлекаем область изображения (ROI)
        roi = wire_img[y:y+h, x:x+w]
        
        # Вычисляем средний цвет
        avg_color = np.mean(roi, axis=(0, 1)).astype(int)
        
        # Предлагаем пользователю назвать цвет
        print(f"Прямоугольник {i+1} - Средний RGB: {avg_color}")
        color_name = input(f"Введите название цвета для прямоугольника {i+1} (red/green/blue/black/white): ").lower()
        
        # Сохраняем образец цвета
        if color_name in ["red", "green", "blue", "black", "white"]:
            if color_name not in color_samples:
                color_samples[color_name] = []
            
            color_samples[color_name].append(avg_color)
    
    # Генерируем новые цветовые диапазоны
    new_color_ranges = {}
    for color, samples in color_samples.items():
        if samples:
            # Преобразуем в массив numpy
            samples_array = np.array(samples)
            
            # Вычисляем минимум и максимум для каждого канала
            min_values = np.min(samples_array, axis=0)
            max_values = np.max(samples_array, axis=0)
            
            # Добавляем запас для учета вариаций
            lower = np.maximum(min_values - 30, 0)
            upper = np.minimum(max_values + 30, 255)
            
            new_color_ranges[color] = {
                "lower": lower,
                "upper": upper,
                "name": color.capitalize()
            }
    
    # Выводим код для новых диапазонов
    print("\nПредлагаемые цветовые диапазоны:")
    print("COLOR_RANGES = {")
    for color, data in new_color_ranges.items():
        print(f"    \"{color}\": {{")
        print(f"        \"lower\": np.array({data['lower'].tolist()}),")
        print(f"        \"upper\": np.array({data['upper'].tolist()}),")
        print(f"        \"name\": \"{data['name']}\"")
        print("    },")
    print("}")
    
    return new_color_ranges

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
    # Это примерная область, вам нужно будет настроить её под ваше изображение
    wire_left = int(screen_width * 0.2)
    wire_top = int(screen_height * 0.3)
    wire_width = int(screen_width * 0.2)
    wire_height = int(screen_height * 0.5)
    
    wire_region = (wire_left, wire_top, wire_width, wire_height)
    
    print("=== Программа распознавания и сравнения цветов проводов ===")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Область текста: {text_region}")
    print(f"Область проводов: {wire_region}")
    print("Поддерживаемые цвета: Red, Green, Black, Blue, White")
    
    # Настройка прямоугольников для проводов
    # setup_rects = input("\nЗапустить настройку прямоугольников для проводов? (y/n): ")
    # if setup_rects.lower() == 'y':
    #     time.sleep(3)
    #     rectangles = setup_wire_rectangles(wire_region)
    #     WIRE_RECTANGLES = rectangles
        
    #     # Предложение калибровки цветовых диапазонов
    #     calibrate = input("\nЗапустить калибровку цветовых диапазонов? (y/n): ")
    #     if calibrate.lower() == 'y' and rectangles:
    #         new_color_ranges = get_color_calibration_from_rectangles(wire_region, rectangles)
            
    #         # Спрашиваем, хочет ли пользователь использовать новые диапазоны
    #         use_new_ranges = input("\nИспользовать новые цветовые диапазоны? (y/n): ")
    #         if use_new_ranges.lower() == 'y':
    #             time.sleep(3)
    #             COLOR_RANGES.update(new_color_ranges)
    #             print("Цветовые диапазоны обновлены")
    # else:
    #     # Если пользователь не хочет настраивать прямоугольники, предложим ввести их вручную
    #     print("\nВведите координаты прямоугольников для проводов (x, y, width, height):")
    #     print("Например: 100,150,30,20")
    #     print("Введите 'готово' когда закончите")
               
    #     rectangles = []
    #     i = 1
    #     while True:
    #         rect_input = input(f"Провод {i}: ")
    #         if rect_input.lower() == 'готово':
    #             break
            
    #         try:
    #             x, y, w, h = map(int, rect_input.split(','))
    #             rectangles.append((x, y, w, h))
    #             i += 1
    #         except:
    #             print("Неверный формат. Используйте: x,y,width,height")
                
    #     print(WIRE_RECTANGLES)
    rectangles = WIRE_RECTANGLES
        
    print(WIRE_RECTANGLES)
    
    # Выводим настроенные прямоугольники
    if WIRE_RECTANGLES:
        print("\nНастроенные прямоугольники:")
        for i, rect in enumerate(WIRE_RECTANGLES):
            print(f"  Провод {i+1}: {rect}")
    else:
        print("\nВНИМАНИЕ: Прямоугольники для проводов не настроены!")
        print("Настройте прямоугольники перед использованием программы")
    
    # Одноразовое тестирование
    if WIRE_RECTANGLES:
        print("\nТестовое чтение...")
        time.sleep(3)
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
        
    else:
        print("Невозможно запустить программу без настроенных прямоугольников для проводов")