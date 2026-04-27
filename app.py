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


def cv2_imread_utf8(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def cv2_imwrite_utf8(path, img, params=None):
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img, params or [])
    if result:
        encoded_img.tofile(path)


def get_file_info(filepath):
    if not os.path.exists(filepath):
        return 0, "0 KB"
    size_bytes = os.path.getsize(filepath)
    return size_bytes / 1024, f"{size_bytes / 1024:.2f} KB"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        app_mode = request.form.get('app_mode', 'auto')
        manual_level = int(request.form.get('level', 1))
        meta_color = int(request.form.get('meta_color', -1))
        meta_edge = float(request.form.get('meta_edge', -1.0))

        uploaded_files = request.files.getlist('files')
        cached_files_str = request.form.get('cached_files', '')

        files_to_process = []
        if uploaded_files and uploaded_files[0].filename != '':
            for f in uploaded_files:
                filepath = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(filepath)
                files_to_process.append(filepath)
        elif cached_files_str:
            files_to_process = cached_files_str.split(',')

        if not files_to_process:
            return "Error: No files"

        results_data = []

        for original_path in files_to_process:
            if not os.path.exists(original_path):
                continue
            original_img = cv2_imread_utf8(original_path)
            if original_img is None:
                continue

            h_orig, w_orig = original_img.shape[:2]
            size_orig_val, size_orig_str = get_file_info(original_path)
            filename = os.path.basename(original_path)
            name_only = filename.rsplit('.', 1)[0]

            t0 = time.time()

            if app_mode == 'auto' and size_orig_val < 50:
                compressed_path = os.path.join(RESULTS_FOLDER,
                                               f"comp_{name_only}.png")
                cv2_imwrite_utf8(compressed_path, original_img)
                t1 = time.time()

                restored_path = os.path.join(RESULTS_FOLDER,
                                             f"rest_{name_only}.png")
                cv2_imwrite_utf8(restored_path, original_img)
                t2 = time.time()

                model_name = "Pass"
                comp_level_str = "Off"
                highlight_text = "Small file"
            else:
                if app_mode == 'auto':
                    megapixels = (h_orig * w_orig) / 1_000_000
                    if megapixels >= 4.0:
                        current_comp_level = 3
                    elif megapixels >= 1.0:
                        current_comp_level = 2
                    else:
                        current_comp_level = 1
                    c_size = None
                    e_scale = None
                    auto_reason = " (Auto)"
                else:
                    current_comp_level = manual_level
                    c_size = meta_color if meta_color != -1 else None
                    e_scale = meta_edge if meta_edge != -1.0 else None
                    auto_reason = " (Manual)"

                if current_comp_level == 0:
                    compressed_path = os.path.join(RESULTS_FOLDER,
                                                   f"comp_{name_only}.jpg")
                    cv2_imwrite_utf8(compressed_path, original_img,
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
                    t1 = time.time()

                    final_img = cv2_imread_utf8(compressed_path)
                    model_name = "Baseline JPEG"
                    restored_path = os.path.join(RESULTS_FOLDER,
                                                 f"rest_{name_only}.jpg")
                    cv2_imwrite_utf8(restored_path, final_img,
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
                    t2 = time.time()
                    comp_level_str = f"OFF{auto_reason}"
                else:
                    color_palette = meta_engine.extract_color_palette(
                        original_img, force_size=c_size)
                    edge_map = meta_engine.extract_edge_map(
                        original_img, force_scale=e_scale)
                    skeleton = compressor.cascade_compress(
                        original_img, level=current_comp_level)

                    compressed_path = os.path.join(RESULTS_FOLDER,
                                                   f"comp_{name_only}.jpg")
                    cv2_imwrite_utf8(compressed_path, skeleton,
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
                    t1 = time.time()

                    raw_restored, model_name, _ = ai_engine.restore_image(
                        skeleton, original_img.shape,
                        steps=current_comp_level)

                    color_corrected = meta_engine.apply_color_correction(
                        raw_restored, color_palette)
                    final_img = meta_engine.apply_edge_sharpening(
                        color_corrected, edge_map)

                    restored_path = os.path.join(RESULTS_FOLDER,
                                                 f"rest_{name_only}.jpg")
                    cv2_imwrite_utf8(restored_path, final_img,
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
                    t2 = time.time()

                    comp_level_str = f"L{current_comp_level}{auto_reason}"

            time_enc = round(t1 - t0, 2)
            time_dec = round(t2 - t1, 2)

            size_comp_val, size_comp_str = get_file_info(compressed_path)
            size_rest_val, size_rest_str = get_file_info(restored_path)
            comp_img = cv2_imread_utf8(compressed_path)
            if comp_img is not None:
                h_comp, w_comp = comp_img.shape[:2]
            else:
                h_comp, w_comp = (0, 0)

            if app_mode == 'auto' and size_orig_val < 50:
                psnr_val = 100.0
                ssim_val = 1.000
                compression_ratio = 0.0
            else:
                restored_img = cv2_imread_utf8(restored_path)
                psnr_val = round(calculate_psnr(
                    original_img, restored_img), 2)
                ssim_val = round(calculate_ssim(
                    original_img, restored_img), 3)
                if size_orig_val > 0:
                    compression_ratio = (1 - (size_comp_val / size_orig_val)) * 100
                else:
                    compression_ratio = 0
                highlight_text = f"Comp: {compression_ratio:.1f}%"

            results_data.append({
                "filename": filename,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "comp_ratio": f"{compression_ratio:.1f}%",
                "time_enc": time_enc,
                "time_dec": time_dec,
                "stages": [
                    {"title": "1. Original",
                     "size": size_orig_str,
                     "res": f"{w_orig}x{h_orig} px",
                     "path": original_path},
                    {"title": f"2. Package ({comp_level_str})",
                     "size": size_comp_str,
                     "res": f"{w_comp}x{h_comp} px",
                     "path": compressed_path,
                     "highlight": highlight_text},
                    {"title": f"3. Result ({model_name})",
                     "size": size_rest_str,
                     "res": f"{w_orig}x{h_orig} px",
                     "path": restored_path}
                ]
            })

        cached_files_out = ",".join(files_to_process)
        return render_template('index.html',
                               results=results_data,
                               cached_files=cached_files_out,
                               app_mode=app_mode,
                               current_level=manual_level,
                               meta_color=meta_color,
                               meta_edge=meta_edge)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)