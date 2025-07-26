from ultralytics import YOLO
import cv2
import numpy as np
import os

class KidneyDetector:
    def __init__(self, weights_path='C:/Phyton/Проект/best.pt', conf_threshold=0.3):
        try:
            self.model = YOLO(weights_path)
            self.conf_threshold = conf_threshold
            print(f"Модель успешно загружена из {weights_path}")
        except Exception as e:
            raise Exception(f"Ошибка загрузки модели: {str(e)}")
    
    def preprocess_image(self, image):
        """Предобработка изображения для улучшения детекции"""
        # Конвертация в RGB если нужно
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Улучшение контраста (CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect(self, image):
        """Детектирует почки на изображении"""
        try:
            # Предобработка
            processed_image = self.preprocess_image(image)
            
            # Детекция
            results = self.model(processed_image, conf=self.conf_threshold)
            
            # Обработка результатов
            detections = []
            for result in results:
                for box in result.boxes:
                    if box.conf >= self.conf_threshold:
                        class_id = int(box.cls)
                        # Меняем метки классов местами
                        if result.names[class_id] == 'right_kidney':
                            class_name = 'left_kidney'
                        elif result.names[class_id] == 'left_kidney':
                            class_name = 'right_kidney'
                        else:
                            class_name = result.names[class_id]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': float(box.conf),
                            'bbox': [int(x) for x in box.xyxy[0].tolist()]  # x1, y1, x2, y2
                        })
            
            print(f"Найдено детекций: {len(detections)}")
            for det in detections:
                print(f"Обнаружен {det['class']} с уверенностью {det['confidence']:.2f}")
                
            return detections
        except Exception as e:
            print(f"Ошибка детекции: {str(e)}")
            return []