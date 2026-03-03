import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    # PSNR есть встроенный в OpenCV
    return cv2.PSNR(img1, img2)


def calculate_ssim(img1, img2):
    # Для SSIM картинки должны быть в градациях серого
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # score вернет значение от -1 до 1 (где 1 - идентичные картинки)
    score, _ = ssim(gray1, gray2, full=True)
    return score