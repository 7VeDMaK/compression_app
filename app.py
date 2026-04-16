import os
import cv2
from flask import Flask, render_template, request
from core.wavelet_engine import WaveletCompressor
from core.ai_engine import AIEnhancer
from core.metrics import calculate_psnr, calculate_ssim

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

compressor = WaveletCompressor(wavelet_type='haar')
ai_engine = AIEnhancer()

def get_file_info(filepath):
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

        filename = file.filename
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(original_path)
        original_img = cv2.imread(original_path)
        h_orig, w_orig = original_img.shape[:2]

        # --- ЭТАП 1: ОРИГИНАЛ ---
        size_orig_val, size_orig_str = get_file_info(original_path)

        # --- ЭТАП 2: СЖАТИЕ (DWT) ---
        compressed_data, original_shape = compressor.get_compressed_skeleton(original_path)
        name_without_ext = filename.rsplit('.', 1)[0]
        compressed_filename = f"comp_{name_without_ext}.jpg"
        compressed_path = os.path.join(RESULTS_FOLDER, compressed_filename)
        # Имитируем квантование и энтропийное кодирование через JPG 95
        cv2.imwrite(compressed_path, compressed_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        size_comp_val, size_comp_str = get_file_info(compressed_path)
        h_comp, w_comp = compressed_data.shape[:2]

        # --- ЭТАП 3: ВОССТАНОВЛЕНИЕ (AI) ---
        restored_img = ai_engine.restore_image(compressed_data, original_shape)
        restored_filename = f"restored_{name_without_ext}.jpg"
        restored_path = os.path.join(RESULTS_FOLDER, restored_filename)
        cv2.imwrite(restored_path, restored_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        size_rest_val, size_rest_str = get_file_info(restored_path)

        # --- СТАТИСТИКА И МЕТРИКИ ---
        compression_ratio = (1 - (size_comp_val / size_orig_val)) * 100 if size_orig_val > 0 else 0
        psnr_val = round(calculate_psnr(original_img, restored_img), 2)
        ssim_val = round(calculate_ssim(original_img, restored_img), 3)

        # Формируем данные для "Карточек этапов"
        stages = [
            {
                "title": "1. Исходное изображение",
                "desc": "Входные данные в полном разрешении. Используется как эталон для расчета метрик искажения.",
                "size": size_orig_str,
                "res": f"{w_orig}x{h_orig} px",
                "path": original_path
            },
            {
                "title": "2. Вейвлет-скелет (LL-субполоса)",
                "desc": "Результат DWT. Высокие частоты (детали) отброшены, сохранена только аппроксимация. Размер уменьшен в 4 раза по площади.",
                "size": size_comp_str,
                "res": f"{w_comp}x{h_comp} px",
                "path": compressed_path,
                "highlight": f"Сжатие: {compression_ratio:.1f}%"
            },
            {
                "title": "3. Нейросетевая реконструкция",
                "desc": f"Результат работы модели {ai_engine.model_name.upper()}. Нейросеть восстановила детали, ориентируясь на контекст вейвлет-скелета.",
                "size": size_rest_str,
                "res": f"{w_orig}x{h_orig} px",
                "path": restored_path
            }
        ]

        return render_template('index.html',
                               stages=stages,
                               psnr=psnr_val,
                               ssim=ssim_val,
                               comp_ratio=f"{compression_ratio:.1f}%",
                               restored_flag=True)

    return render_template('index.html', restored_flag=False)

if __name__ == '__main__':
    app.run(debug=True)