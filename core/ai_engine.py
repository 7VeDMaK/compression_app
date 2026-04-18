import cv2
import os
import urllib.request
import numpy as np


class AIEnhancer:
    def __init__(self):
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
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.models[name]["obj"] = sr

    def analyze_complexity(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def restore_image(self, compressed_img, target_shape, steps=1):
        """Интеллектуальное восстановление с динамическим выбором тайлов и нахлестом"""

        complexity = self.analyze_complexity(compressed_img)
        if complexity > 150:
            selected_model = "fsrcnn"
            reason = f"Высокая детализация. Модель FSRCNN."
        else:
            selected_model = "espcn"
            reason = f"Гладкие текстуры. Модель ESPCN."

        sr_engine = self.models[selected_model]["obj"]

        h_comp, w_comp = compressed_img.shape[:2]
        h_target, w_target = target_shape[:2]
        SCALE = 2 ** steps

        # 1. ДИНАМИЧЕСКИЙ ВЫБОР: Если картинка не огромная (< 1 Мегапикселя), делаем целиком
        if h_comp * w_comp <= 1000 * 1000:
            print(f"[AI] {reason} Размер оптимальный. Обрабатываю целиком...")
            current_img = compressed_img
            for _ in range(steps):
                current_img = sr_engine.upsample(current_img)
            result_img = current_img

        # 2. ТАЙЛЫ С НАХЛЕСТОМ (Overlapping) для гигантских изображений
        else:
            print(f"[AI] {reason} Изображение гигантское. Запускаю тайлинг с нахлестом...")
            TILE_SIZE = 256
            OVERLAP = 16  # Захватываем лишние пиксели вокруг тайла, чтобы не было "швов"

            result_img = np.zeros((h_comp * SCALE, w_comp * SCALE, 3), dtype=np.uint8)

            for y in range(0, h_comp, TILE_SIZE):
                for x in range(0, w_comp, TILE_SIZE):
                    # Вычисляем границы с учетом нахлеста
                    y_min = max(0, y - OVERLAP)
                    y_max = min(h_comp, y + TILE_SIZE + OVERLAP)
                    x_min = max(0, x - OVERLAP)
                    x_max = min(w_comp, x + TILE_SIZE + OVERLAP)

                    # Вырезаем тайл с нахлестом
                    tile = compressed_img[y_min:y_max, x_min:x_max]

                    # Апскейлим
                    for _ in range(steps):
                        tile = sr_engine.upsample(tile)

                    # Отрезаем нахлест, чтобы вклеить только идеальную центральную часть
                    crop_top = (y - y_min) * SCALE
                    crop_bottom = tile.shape[0] - ((y_max - min(y + TILE_SIZE, h_comp)) * SCALE)
                    crop_left = (x - x_min) * SCALE
                    crop_right = tile.shape[1] - ((x_max - min(x + TILE_SIZE, w_comp)) * SCALE)

                    tile_cropped = tile[crop_top:crop_bottom, crop_left:crop_right]

                    # Вычисляем куда вклеить
                    y_out = y * SCALE
                    x_out = x * SCALE
                    y_end_out = y_out + tile_cropped.shape[0]
                    x_end_out = x_out + tile_cropped.shape[1]

                    result_img[y_out:y_end_out, x_out:x_end_out] = tile_cropped

        # 3. Подгонка геометрии под точный размер оригинала
        if result_img.shape[:2] != (h_target, w_target):
            result_img = cv2.resize(result_img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)

        return result_img, selected_model.upper(), reason