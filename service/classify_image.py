import tensorflow as tf
import numpy as np
import pydicom
import warnings

MODEL_PATHS = {
    'VGG16_v1': 'C:/Phyton/Проект/VGG16_v1.h5',
    'CNN_v1': 'C:/Phyton/Проект/CNN_v1.h5',
    #'ResNet50': 'C:/Phyton/Проект/ResNet50.h5'
}

classes = ['Normal', 'Cyst', 'Stone', 'Tumor']
current_model_name = 'CNN_v1'  # модель по умолчанию
model = tf.keras.models.load_model(MODEL_PATHS[current_model_name])

def load_dicom(dicom_path):
    """Загрузка DICOM изображения без нормализации (только оконные настройки)"""
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)
    
    # Применяем оконные настройки (если есть)
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        if image.min() < 0 or image.max() > 1:
            center = ds.WindowCenter
            width = ds.WindowWidth

            # Обработка случаев, когда center/width заданы как последовательность
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]

            image = np.clip(image, center - width / 2, center + width / 2)

    return image

def set_model(model_name):
    """Установка текущей модели"""
    global current_model_name, model
    if model_name in MODEL_PATHS:
        current_model_name = model_name
        model = tf.keras.models.load_model(MODEL_PATHS[model_name])
        return True
    return False

def get_current_model_info():
    """Информация о текущей модели"""
    return {
        "model_name": current_model_name,
        #"model_path": MODEL_PATHS[current_model_name]
    }

def classify_dicom(image: np.ndarray) -> dict:
    """
    Классифицирует изображение (NumPy array) и возвращает результат
    Предполагается, что модель обучалась на изображениях 512x512 без нормализации
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray")
    
    # Приведение к 3 каналам (если нужно)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)  # grayscale -> RGB
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)  # (H, W, 1) -> (H, W, 3)

    # Ресайз до 512x512 (если нужно)
    if image.shape != (512, 512, 3):
        image = tf.image.resize(image, (512, 512)).numpy()

    # Добавляем batch-размерность
    img = np.expand_dims(image.astype('float32'), axis=0)

    # Предсказание (игнорируем warning о метриках)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = model.predict(img)[0]
    
    # Формируем результат
    return {
        "model_info": get_current_model_info(),
        "prediction": classes[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)}
    }