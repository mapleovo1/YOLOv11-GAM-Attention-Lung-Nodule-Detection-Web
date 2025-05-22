import os
import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import csv
import uuid

from detect import YOLODetector


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['IMAGES_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)

# 初始化检测器
detector = YOLODetector(weights='YOLOv11m_GAM_Attention.pt')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '没有选择文件'})

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # 执行检测
        start_time = time.time()
        result_img, detections = detector.detect(file_path)
        detection_time = time.time() - start_time

        # 保存结果图片
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_img.save(result_path)

        # 返回结果
        result = {
            'time': f"{detection_time:.3f}",
            'count': len(detections),
            'detections': detections,
            'image_path': f"/static/results/{result_filename}",
            'original_path': file_path
        }

        return jsonify(result)

    return jsonify({'error': '不支持的文件类型'})


@app.route('/save_results', methods=['POST'])
def save_results():
    data = request.json
    if not data or 'detections' not in data:
        return jsonify({'error': '没有检测结果可保存'})

    # 生成CSV文件名
    csv_filename = f"detection_results_{uuid.uuid4().hex}.csv"
    csv_path = os.path.join(app.config['RESULT_FOLDER'], csv_filename)

    # 写入CSV文件，添加BOM标记使Excel正确识别UTF-8编码
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['序号', '文件路径', '类别', '置信度', '坐标位置'])

        for i, det in enumerate(data['detections']):
            coords = f"[{det['xmin']}, {det['ymin']}, {det['xmax']}, {det['ymax']}]"
            writer.writerow([i + 1, data.get('original_path', ''), det['class'], det['confidence'], coords])

    return jsonify({
        'success': True,
        'csv_path': f"/static/results/{csv_filename}",
        'filename': csv_filename
    })


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)


@app.route('/exit')
def exit_app():
    return render_template('exit.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)