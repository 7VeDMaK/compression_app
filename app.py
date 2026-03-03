import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from core.wavelet_engine import WaveletCompressor
from core.ai_engine import AIEnhancer
from core.metrics import calculate_psnr, calculate_ssim

app = Flask(__name__)

# Папки
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

compressor = WaveletCompressor(wavelet_type='haar')
ai_engine = AIEnhancer()


# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ---
def get_file_info(filepath):
    """Возвращает размер файла в КБ (число) и отформатированную строку"""
    if not os.path.exists(filepath):
        return 0, "0 KB"
    size_bytes = os.path.getsize(filepath)
    size_kb = size_bytes / 1024
    return size_kb, f"{size_kb:.2f} KB"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return "Нет файла"
        file = request.files['file']
        if file.filename == '': return "Файл не выбран"

        # 1. Оригинал
        filename = file.filename
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(original_path)
        original_img = cv2.imread(original_path)

        # 2. СЖАТИЕ (Wavelet)
        compressed_data, original_shape = compressor.get_compressed_skeleton(original_path)

        compressed_filename = f"comp_{filename}"
        compressed_path = os.path.join(RESULTS_FOLDER, compressed_filename)
        # Сохраняем с небольшим сжатием JPG, чтобы увидеть реальную экономию места
        cv2.imwrite(compressed_path, compressed_data, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 3. ВОССТАНОВЛЕНИЕ (AI)
        restored_img = ai_engine.restore_image(compressed_data, original_shape)

        restored_filename = f"restored_{filename}"
        restored_path = os.path.join(RESULTS_FOLDER, restored_filename)
        cv2.imwrite(restored_path, restored_img)

        # --- 4. РАСЧЕТ РАЗМЕРОВ ---
        size_orig_val, size_orig_str = get_file_info(original_path)
        size_comp_val, size_comp_str = get_file_info(compressed_path)
        size_rest_val, size_rest_str = get_file_info(restored_path)

        # Считаем, на сколько процентов сжали
        if size_orig_val > 0:
            compression_ratio = (1 - (size_comp_val / size_orig_val)) * 100
            compression_txt = f"-{compression_ratio:.1f}%"
        else:
            compression_txt = "0%"

        # --- 5. МЕТРИКИ ---
        h, w = original_img.shape[:2]
        restored_img_resized = cv2.resize(restored_img, (w, h))

        try:
            psnr_val = round(calculate_psnr(original_img, restored_img_resized), 2)
            ssim_val = round(calculate_ssim(original_img, restored_img_resized), 3)
        except:
            psnr_val, ssim_val = 0, 0

        return render_template('index.html',
                               original=original_path,
                               compressed=compressed_path,
                               restored=restored_path,
                               # Передаем размеры
                               size_orig=size_orig_str,
                               size_comp=size_comp_str,
                               size_rest=size_rest_str,
                               comp_ratio=compression_txt,
                               # Метрики
                               psnr=psnr_val,
                               ssim=ssim_val,
                               restored_flag=True)

    return render_template('index.html', restored_flag=False)


if __name__ == '__main__':
    app.run(debug=True)