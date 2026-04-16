import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request
from core.wavelet_engine import WaveletCompressor
from core.ai_engine import AIEnhancer
from core.metadata_engine import MetadataEngine
from core.metrics import calculate_psnr, calculate_ssim

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

compressor = WaveletCompressor()
ai_engine = AIEnhancer()
meta_engine = MetadataEngine()


# --- БЕЗОПАСНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С КИРИЛЛИЦЕЙ ---
def cv2_imread_utf8(path):
    """Читает изображение по пути с русскими буквами"""
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def cv2_imwrite_utf8(path, img, params=None):
    """Сохраняет изображение по пути с русскими буквами"""
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img, params or [])
    if result:
        encoded_img.tofile(path)


def get_file_info(filepath):
    if not os.path.exists(filepath): return 0, "0 KB"
    size_bytes = os.path.getsize(filepath)
    return size_bytes / 1024, f"{size_bytes / 1024:.2f} KB"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        comp_level = int(request.form.get('level', 1))

        if not file or file.filename == '': return "Файл не выбран"

        filename = file.filename
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(original_path)

        # Используем безопасное чтение
        original_img = cv2_imread_utf8(original_path)
        if original_img is None:
            return "Ошибка чтения файла. Проверьте формат изображения."

        h_orig, w_orig = original_img.shape[:2]
        size_orig_val, size_orig_str = get_file_info(original_path)

        # --- КОДЕР ---
        start_time = time.time()

        color_palette = meta_engine.extract_color_palette(original_img)
        edge_map = meta_engine.extract_edge_map(original_img)
        skeleton = compressor.cascade_compress(original_img, level=comp_level)

        name_only = filename.rsplit('.', 1)[0]
        compressed_path = os.path.join(RESULTS_FOLDER, f"comp_{name_only}.jpg")

        # Безопасное сохранение
        cv2_imwrite_utf8(compressed_path, skeleton, [cv2.IMWRITE_JPEG_QUALITY, 85])
        size_comp_val, size_comp_str = get_file_info(compressed_path)
        h_comp, w_comp = skeleton.shape[:2]

        # --- ДЕКОДЕР ---
        raw_restored, model_name, ai_reason = ai_engine.restore_image(skeleton, original_img.shape, steps=comp_level)

        if raw_restored.shape[:2] != (h_orig, w_orig):
            raw_restored = cv2.resize(raw_restored, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        color_corrected = meta_engine.apply_color_correction(raw_restored, color_palette)

        # Применяем контуры, БЕЗ искусственного пересвета CLAHE!
        final_img = meta_engine.apply_edge_sharpening(color_corrected, edge_map)

        restored_path = os.path.join(RESULTS_FOLDER, f"rest_{name_only}.jpg")
        cv2_imwrite_utf8(restored_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        size_rest_val, size_rest_str = get_file_info(restored_path)

        exec_time = round(time.time() - start_time, 2)

        psnr_val = round(calculate_psnr(original_img, final_img), 2)
        ssim_val = round(calculate_ssim(original_img, final_img), 3)
        compression_ratio = (1 - (size_comp_val / size_orig_val)) * 100 if size_orig_val > 0 else 0

        stages = [
            {"title": "1. Оригинал", "desc": "Исходное изображение.", "size": size_orig_str,
             "res": f"{w_orig}x{h_orig} px", "path": original_path},
            {"title": f"2. DWT Скелет (Уровень {comp_level})", "desc": "Передаваемый файл.", "size": size_comp_str,
             "res": f"{w_comp}x{h_comp} px", "path": compressed_path,
             "highlight": f"Сжато на: {compression_ratio:.1f}%"},
            {"title": f"3. Восстановление ({model_name})", "desc": "Только AI + Цвета + Контуры.",
             "size": size_rest_str, "res": f"{w_orig}x{h_orig} px", "path": restored_path}
        ]

        return render_template('index.html', stages=stages, psnr=psnr_val, ssim=ssim_val,
                               comp_ratio=f"{compression_ratio:.1f}%", time=exec_time, restored_flag=True)

    return render_template('index.html', restored_flag=False)


if __name__ == '__main__':
    app.run(debug=True)