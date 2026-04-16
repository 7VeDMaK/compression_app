import cv2
import os
import urllib.request

class AIEnhancer:
    def __init__(self):
        # ВАРИАНТ 2: FSRCNN (специально оптимизирована для скорости)
        self.model_filename = "FSRCNN_x2.pb"
        self.model_url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
        self.model_name = "fsrcnn"

        self._check_and_download_model()

        # Инициализация модуля Super Resolution
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.model_filename)
        self.sr.setModel(self.model_name, 2)

        # Выполняем на процессоре (CPU)
        self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _check_and_download_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, self.model_filename)
        self.model_filename = model_path

        if not os.path.exists(model_path):
            print(f"[AI Engine] Скачиваю веса модели {self.model_filename}...")
            try:
                urllib.request.urlretrieve(self.model_url, model_path)
                print("[AI Engine] Скачивание завершено успешно.")
            except Exception as e:
                print(f"[AI Engine] Ошибка скачивания: {e}")

    def restore_image(self, compressed_img, target_shape):
        """
        Апскейлинг сжатого LL-скелета до исходного разрешения
        """
        result = self.sr.upsample(compressed_img)

        # Корректировка геометрии (если были нечетные пиксели при сжатии)
        h, w = target_shape[:2]
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

        return result