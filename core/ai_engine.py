import cv2
import os
import urllib.request
import numpy as np


class AIEnhancer:
    def __init__(self):
        # Используем модель EDSR (она одна из лучших для восстановления текстур)
        self.model_filename = "EDSR_x2.pb"
        # x2 означает, что она увеличит картинку в 2 раза (как раз обратное действие вейвлета Haar)
        self.model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"

        self._check_and_download_model()

        # Инициализация Super Resolution в OpenCV
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.modчel_filename)
        self.sr.setModel("edsr", 2)  # Указываем апскейл x2

    def _check_and_download_model(self):
        # Путь к папке core, чтобы модель лежала рядом со скриптом
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, self.model_filename)
        self.model_filename = model_path  # Обновляем путь на полный

        if not os.path.exists(model_path):
            print(f"[AI] Скачиваю веса модели {self.model_filename}...")
            try:
                urllib.request.urlretrieve(self.model_url, model_path)
                print("[AI] Скачивание завершено.")
            except Exception as e:
                print(f"[AI] Ошибка скачивания: {e}")

    def restore_image(self, compressed_img, target_shape):
        """
        Принимает: сжатую (маленькую) картинку.
        Возвращает: восстановленную большую картинку.
        """
        # 1. Прогоняем через нейросеть
        # Она сама увеличит разрешение в 2 раза и дорисует детали
        result = self.sr.upsample(compressed_img)

        # 2. Страховка размеров (иногда бывают нестыковки на 1-2 пикселя)
        h, w = target_shape[:2]
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h))

        return result