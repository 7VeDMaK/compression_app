import cv2
import os
import urllib.request
import numpy as np


class AIEnhancer:
    def __init__(self):
        # Загружаем обе модели для динамического выбора
        self.models = {
            "espcn": {
                "file": "ESPCN_x2.pb",
                "url": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
                "obj": None
            },
            "fsrcnn": {
                "file": "FSRCNN_x2.pb",
                "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
                "obj": None
            }
        }
        self._init_models()

    def _init_models(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        for name, data in self.models.items():
            model_path = os.path.join(current_dir, data["file"])
            if not os.path.exists(model_path):
                print(f"[AI] Скачиваю веса {name.upper()}...")
                urllib.request.urlretrieve(data["url"], model_path)

            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel(name, 2)

            # Включаем внутреннюю оптимизацию OpenCV для процессора
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.models[name]["obj"] = sr

    def analyze_complexity(self, image):
        """Оценивает количество деталей на картинке (Дисперсия Лапласиана)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def restore_image(self, compressed_img, target_shape, steps=1):
        """Динамически выбирает модель и восстанавливает изображение ТАЙЛАМИ (Последовательно)"""

        complexity = self.analyze_complexity(compressed_img)

        if complexity > 150:
            selected_model = "fsrcnn"
            reason = f"Много деталей (Лапл: {complexity:.0f}). Выбрана FSRCNN."
        else:
            selected_model = "espcn"
            reason = f"Гладкие текстуры (Лапл: {complexity:.0f}). Выбрана ESPCN."

        print(f"[AI] {reason} Начинаю кэш-оптимизированную тайловую сборку...")
        sr_engine = self.models[selected_model]["obj"]

        # Оптимальный размер для кэша процессора (спасает RAM на гигантских фото)
        TILE_SIZE = 256
        SCALE = 2 ** steps

        h_comp, w_comp = compressed_img.shape[:2]
        h_target, w_target = target_shape[:2]

        # Создаем пустой холст нужного размера
        result_img = np.zeros((h_comp * SCALE, w_comp * SCALE, 3), dtype=np.uint8)

        total_tiles = ((h_comp // TILE_SIZE) + 1) * ((w_comp // TILE_SIZE) + 1)
        current_tile = 0

        # Обрабатываем по одному квадрату (OpenCV сам раскидает вычисления на все ядра)
        for y in range(0, h_comp, TILE_SIZE):
            for x in range(0, w_comp, TILE_SIZE):
                current_tile += 1

                y_end = min(y + TILE_SIZE, h_comp)
                x_end = min(x + TILE_SIZE, w_comp)

                # Вырезаем кусок
                tile = compressed_img[y:y_end, x:x_end]

                # Апскейлим кусок
                for _ in range(steps):
                    tile = sr_engine.upsample(tile)

                # Вставляем на холст
                y_out, x_out = y * SCALE, x * SCALE
                y_end_out, x_end_out = y_end * SCALE, x_end * SCALE

                result_img[y_out:y_end_out, x_out:x_end_out] = tile

                if current_tile % 5 == 0 or current_tile == total_tiles:
                    print(f"[AI] Обработано {current_tile}/{total_tiles} фрагментов...")

        # Подгонка геометрии под точный размер оригинала
        if result_img.shape[:2] != (h_target, w_target):
            result_img = cv2.resize(result_img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)

        return result_img, selected_model.upper(), reason