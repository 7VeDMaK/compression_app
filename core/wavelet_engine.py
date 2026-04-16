import pywt
import cv2
import numpy as np


class WaveletCompressor:
    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    def get_compressed_skeleton(self, image_path):
        """
        Возвращает нормализованное сжатое изображение и его исходный размер
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        # Перевод в YCrCb (согласно дипломной работе)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        # Разбиваем на каналы: Яркость (Y), Цветоразность (Cr, Cb)
        y, cr, cb = cv2.split(img_ycrcb)

        def process_channel(channel):
            # Двумерное дискретное вейвлет-преобразование
            coeffs = pywt.dwt2(channel, self.wavelet_type)
            LL, (LH, HL, HH) = coeffs

            # Нормализация амплитуды фильтра Хаара (возвращаем исходную яркость)
            return LL / 2.0

        ll_y = process_channel(y)
        ll_cr = process_channel(cr)
        ll_cb = process_channel(cb)

        # Объединяем аппроксимирующие матрицы
        merged_ll = cv2.merge([ll_y, ll_cr, ll_cb])

        # Округляем для подавления вычислительного шума и ограничиваем диапазон 8-bit
        merged_ll = np.clip(np.round(merged_ll), 0, 255).astype(np.uint8)

        # Обратная конверсия в BGR для передачи по сети и сохранения
        skeleton_bgr = cv2.cvtColor(merged_ll, cv2.COLOR_YCrCb2BGR)

        return skeleton_bgr, img.shape