import pywt
import cv2
import numpy as np


class WaveletCompressor:
    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    def get_compressed_skeleton(self, image_path):
        """
        Возвращает:
        1. compressed_img - нормализованное изображение (скелет)
        2. original_shape - размеры оригинала
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        # Работаем с float, чтобы не потерять данные при вычислениях
        img = img.astype(np.float32)

        # Разбиваем на каналы
        b, g, r = cv2.split(img)

        def get_ll(channel):
            coeffs = pywt.dwt2(channel, self.wavelet_type)
            LL, (LH, HL, HH) = coeffs

            # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
            # PyWavelets Haar увеличивает значения в 2 раза.
            # Делим на 2, чтобы вернуть оригинальную яркость.
            return LL / 2.0

        ll_b = get_ll(b)
        ll_g = get_ll(g)
        ll_r = get_ll(r)

        # Собираем
        merged_ll = cv2.merge([ll_b, ll_g, ll_r])

        # Теперь безопасно обрезаем (значения уже в пределах нормы)
        # Округляем (round) перед приведением к типу, чтобы уменьшить шум
        compressed_img = np.round(merged_ll)
        compressed_img = np.clip(compressed_img, 0, 255).astype('uint8')

        return compressed_img, img.shape