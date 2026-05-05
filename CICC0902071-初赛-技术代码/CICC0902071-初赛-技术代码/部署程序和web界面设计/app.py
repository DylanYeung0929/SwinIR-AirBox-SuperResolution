import os
import threading
import uuid
import time
import subprocess
import re
import sys
import cv2
import numpy as np
import sophon.sail as sail
import queue
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pipeline import Pipeline
from config import get_args

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 初始化推理管道
args = get_args()
args.save_dir = app.config['RESULT_FOLDER']
pipeline = Pipeline(args)
task_queue = pipeline.input_queue
result_dict = {}

# 启动管道处理线程
pipeline_thread = threading.Thread(target=pipeline.start, args=([],))
pipeline_thread.daemon = True
pipeline_thread.start()

def background_processor():
    """后台处理结果监控"""
    while True:
        if not pipeline.output_queue.empty():
            result = pipeline.output_queue.get()
            if result:
                result_dict[result['path']] = result
        time.sleep(0.1)

# 启动后台处理器
processor_thread = threading.Thread(target=background_processor)
processor_thread.daemon = True
processor_thread.start()

@app.route('/', methods=['GET'])
def index():
    """渲染简单上传界面"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """API端点：上传并处理图像"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(filepath)
        
        # 添加处理任务
        task_queue.put(filepath)
        return jsonify({
            'task_id': task_id,
            'filename': filename,
            'message': 'File uploaded and processing started'
        }), 202

@app.route('/api/status/<task_id>', methods=['GET'])
def check_status(task_id):
    """检查任务状态"""
    # 查找匹配的任务结果
    for path, result in result_dict.items():
        if task_id in path:
            if 'save_path' in result:
                return jsonify({
                    'status': 'completed',
                    'result_path': result['save_path'],
                    'processing_time': result['time'],
                    'psnr': result.get('psnr', None)
                })
            return jsonify({'status': 'processing'})
    
    return jsonify({'status': 'not_found'}), 404

@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    """获取处理后的图像"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/batch', methods=['POST'])
def batch_process():
    """批量处理端点（可选）"""
    # 实现类似单文件上传的批处理逻辑
    # 返回批量任务ID列表
    pass

@app.route('/api/tpu_status', methods=['GET'])
def get_tpu_status():
    """获取TPU状态信息"""
    try:
        # 执行bm-smi命令并捕获输出
        result = subprocess.run(['bm-smi'], capture_output=True, text=True)
        output = result.stdout
        
        # 解析bm-smi输出
        parsed_data = parse_bm_smi(output)
        return jsonify(parsed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def parse_bm_smi(output):
    """解析bm-smi输出"""
    # 示例解析逻辑 - 实际实现需要根据具体bm-smi输出格式调整
    data = {
        "devices": [],
        "summary": {
            "tpu_count": 0,
            "memory_total": 0,
            "memory_used": 0,
            "utilization": 0,
            "temperature": 0
        }
    }
    
    # 解析设备信息
    device_lines = re.findall(r'\| (\d+) *\|.*?(\d+)%.*?(\d+)C.*?(\d+)MB.*?(\d+)MB', output)
    for i, match in enumerate(device_lines):
        device = {
            "id": int(match[0]),
            "utilization": int(match[1]),
            "temperature": int(match[2]),
            "mem_used": int(match[3]),
            "mem_total": int(match[4])
        }
        data["devices"].append(device)
    
    # 计算汇总信息
    if data["devices"]:
        data["summary"]["tpu_count"] = len(data["devices"])
        data["summary"]["memory_total"] = sum(d["mem_total"] for d in data["devices"])
        data["summary"]["memory_used"] = sum(d["mem_used"] for d in data["devices"])
        data["summary"]["utilization"] = sum(d["utilization"] for d in data["devices"]) // len(data["devices"])
        data["summary"]["temperature"] = max(d["temperature"] for d in data["devices"])
    
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)