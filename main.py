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

def preprocess_image(image):
    """
    Предварительная обработка изображения для улучшения результатов OCR
    """
    # Конвертация в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение порогового значения для получения черного текста на белом фоне
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Применение расширения для утолщения текста
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def read_text_from_region(region=None, preprocess=True, save_debug_image=True):
    """
    Захват области экрана и извлечение текста из нее
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
    
    # Предварительная обработка изображения области, если требуется
    if preprocess:
        preprocessed_img = preprocess_image(img)
        
        # Сохранение предварительно обработанной области
        cv2.imwrite('preprocessed_region.png', preprocessed_img)
    else:
        preprocessed_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Конвертация в PIL Image для pytesseract
    pil_img = Image.fromarray(preprocessed_img)
    
    # Извлечение текста
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    
    return text.strip()

def monitor_text_changes(region, interval=1.0):
    """
    Непрерывный мониторинг области для отслеживания изменений текста
    """
    last_text = ""
    
    try:
        while True:
            current_text = read_text_from_region(region)
            
            if current_text != last_text:
                print(f"Обнаруженный текст: {current_text}")
                last_text = current_text
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Мониторинг остановлен")

if __name__ == "__main__":
    
    # Получение размеров экрана
    screen_width, screen_height = pyautogui.size()
    
    # Определение области с указанными вами значениями
    left = int(screen_width * 0.6)
    top = int(screen_height * 0.65)
    width = int(screen_width * 0.0725)
    height = int(screen_height * 0.145)
    
    region = (left, top, width, height)
    
    print("Начало обнаружения текста. Нажмите Ctrl+C для остановки.")
    print(f"Размеры экрана: {screen_width}x{screen_height}")
    print(f"Отслеживаемая область: {region}")
    time.sleep(1)
    
    # Для одноразового чтения:
    text = read_text_from_region(region)
    print(f"Обнаруженный текст: {text}")
    
    # Для непрерывного мониторинга:
    # monitor_text_changes(region)