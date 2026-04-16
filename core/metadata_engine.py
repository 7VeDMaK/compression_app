import cv2
import numpy as np


class MetadataEngine:
    """Модуль извлечения и применения метаданных (Цвета + Контуры)"""

    @staticmethod
    def extract_color_palette(image, size=(32, 32)):
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def extract_edge_map(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        h, w = edges.shape[:2]
        return cv2.resize(edges, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    @staticmethod
    def apply_color_correction(target_img, palette):
        h, w = target_img.shape[:2]
        color_map = cv2.resize(palette, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.addWeighted(target_img, 0.85, color_map, 0.15, 0)

    @staticmethod
    def apply_edge_sharpening(target_img, edge_map_small):
        h, w = target_img.shape[:2]
        edges = cv2.resize(edge_map_small, (w, h), interpolation=cv2.INTER_CUBIC)

        gaussian = cv2.GaussianBlur(target_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(target_img, 1.5, gaussian, -0.5, 0)

        mask = edges.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.merge([mask, mask, mask])

        final = target_img * (1 - mask) + sharpened * mask
        return np.clip(final, 0, 255).astype(np.uint8)