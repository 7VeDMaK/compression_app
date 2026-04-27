import pywt
import cv2
import numpy as np

class WaveletCompressor:
    def __init__(self, wavelet_type='haar'):
        self.wavelet_type = wavelet_type

    def cascade_compress(self, image, level=1):
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img_ycrcb = img_ycrcb.astype(np.float32)

        current_data = img_ycrcb
        for _ in range(level):
            y, cr, cb = cv2.split(current_data)

            def get_ll(channel):
                coeffs = pywt.dwt2(channel, self.wavelet_type)
                return coeffs[0] / 2.0

            current_data = cv2.merge([
                get_ll(y), get_ll(cr), get_ll(cb)
            ])

        skeleton = np.clip(np.round(current_data), 0, 255)
        skeleton = skeleton.astype(np.uint8)
        return cv2.cvtColor(skeleton, cv2.COLOR_YCrCb2BGR)