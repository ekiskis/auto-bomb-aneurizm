import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time

# Path to tesseract executable - update this if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust for Windows

def capture_screen_region(region=None):
    """
    Захватить определенную область экрана
    область -это кортеж (слева, верхняя, ширина, высота)
    Если регион нет, захватывает весь экран
    """
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)

def preprocess_image(image):
    """
    Предварительно обрабатывать изображение для лучших результатов OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get black text on white background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Apply dilation to make text thicker
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def read_text_from_region(region=None, preprocess=True):
    """
    Захватить область экрана и извлечь из него текст
    """
    # Capture the region
    img = capture_screen_region(region)
    
    # Предварительно обработать изображение, если это необходимо
    if preprocess:
        img = preprocess_image(img)
        
        # Отладка: Сохраните предварительно обработанное изображение
        cv2.imwrite('preprocessed.png', img)
    
    # Convert back to PIL Image for pytesseract
    pil_img = Image.fromarray(img)
    
    # Extract text
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    
    return text.strip()

def monitor_text_changes(region, interval=1.0):
    """
    Непрерывно отслеживать регион для изменений текста
    """
    last_text = ""
    
    try:
        while True:
            current_text = read_text_from_region(region)
            
            if current_text != last_text:
                print(f"Detected text: {current_text}")
                last_text = current_text
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped")

if __name__ == "__main__":
    # Определите область для правого нижнего (отрегулируйте эти значения, чтобы соответствовать вашему экрану)
    # Формат: (слева, вверху, ширина, высота)

    # Примеры значений -отрегулируйте их для вашего конкретного экрана
    screen_width, screen_height = pyautogui.size()
    
    # Это нацелен на приблизительно правящую нижнюю область, где показана последовательность проводов
    # Вам нужно настроить эти значения на основе разрешения экрана
    left = int(screen_width * 0.7)
    top = int(screen_height * 0.6) 
    width = int(screen_width * 0.3)
    height = int(screen_height * 0.4)
    
    region = (left, top, width, height)
    
    print("Starting text detection. Press Ctrl+C to stop.")
    print(f"Monitoring region: {region}")
    
    # Для единовременного чтения:
    text = read_text_from_region(region)
    print(f"Detected text: {text}")
    
    # Для непрерывного мониторинга:
    # monitor_text_changes (область)