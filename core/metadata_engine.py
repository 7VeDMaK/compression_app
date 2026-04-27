import cv2
import numpy as np


class MetadataEngine:
    @staticmethod
    def get_optimal_sizes(h, w):
        palette_dim = max(16, min(150, int(max(h, w) * 0.05)))
        edge_scale = 0.25 if max(h, w) > 1500 else 0.5
        return palette_dim, edge_scale

    @staticmethod
    def extract_color_palette(image, force_size=None):
        if force_size == 0:
            return None

        h, w = image.shape[:2]
        if force_size is not None:
            dim = force_size
        else:
            dim = MetadataEngine.get_optimal_sizes(h, w)[0]
        return cv2.resize(image, (dim, dim),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def extract_edge_map(image, force_scale=None):
        if force_scale == 0:
            return None

        h, w = image.shape[:2]
        if force_scale is not None:
            scale = force_scale
        else:
            scale = MetadataEngine.get_optimal_sizes(h, w)[1]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.resize(edges,
                          (max(32, int(w * scale)),
                           max(32, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def apply_color_correction(target_img, palette):
        if palette is None:
            return target_img
        h, w = target_img.shape[:2]
        color_map = cv2.resize(palette, (w, h),
                               interpolation=cv2.INTER_LANCZOS4)
        return cv2.addWeighted(target_img, 0.85, color_map, 0.15, 0)

    @staticmethod
    def apply_edge_sharpening(target_img, edge_map_small):
        if edge_map_small is None:
            return target_img
        h, w = target_img.shape[:2]
        edges = cv2.resize(edge_map_small, (w, h),
                           interpolation=cv2.INTER_LANCZOS4)

        gaussian = cv2.GaussianBlur(target_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(target_img, 1.5, gaussian, -0.5, 0)

        mask = edges.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.merge([mask, mask, mask])

        final = target_img * (1 - mask) + sharpened * mask
        return np.clip(final, 0, 255).astype(np.uint8)