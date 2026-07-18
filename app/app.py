"""
============================================================
  FastDETR 终端风格可视化前端
  Terminal-Style Real-Time Detection Frontend
============================================================

启动方法 (Launch):
  python app/app.py --model outputs/fastdetr_single/best_model.pth

然后浏览器打开 http://localhost:5000
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fast_detr import build_fast_detr
from utils.box_ops import box_cxcywh_to_xyxy

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
#  全局状态
# ============================================================
DETECTOR = None
DEVICE = None
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
    'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush',
]

# 颜色方案 (终端绿 + 高亮)
COLORS = [
    (0, 255, 0), (0, 255, 127), (50, 255, 50),
    (127, 255, 0), (0, 200, 0), (100, 255, 100),
    (0, 255, 200), (50, 200, 50), (200, 255, 0),
    (0, 180, 0), (150, 255, 50), (0, 255, 50),
]
DEFAULT_COLOR = (0, 255, 0)  # 终端绿


def load_model(model_path: str):
    """加载 FastDETR 模型"""
    global DETECTOR, DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SYS] 设备: {DEVICE}")

    DETECTOR = build_fast_detr(
        num_classes=91,
        backbone_name='resnet50',
        use_denoising=False,
        use_mixed_selection=False,
    )
    DETECTOR.to(DEVICE)

    if model_path and os.path.exists(model_path):
        print(f"[SYS] 加载模型: {model_path}")
        ckpt = torch.load(model_path, map_location=DEVICE)
        DETECTOR.load_state_dict(ckpt['model_state_dict'])
        print(f"[SYS] 模型加载完成 (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[SYS] 使用未训练模型 (随机权重)")

    DETECTOR.eval()
    print("[SYS] 模型就绪")


@torch.no_grad()
def detect_frame(image: np.ndarray, score_threshold: float = 0.5):
    """对单帧进行目标检测"""
    if DETECTOR is None:
        return []

    h, w = image.shape[:2]

    # 预处理
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 推理
    outputs = DETECTOR(img_tensor)
    probs = F.softmax(outputs['pred_logits'], dim=-1)
    scores, labels = probs[0, :, :-1].max(dim=-1)
    boxes = outputs['pred_boxes'][0]

    # 过滤
    keep = scores > score_threshold
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    boxes = boxes[keep].cpu().numpy()

    # 转换框坐标 (cxcywh -> xyxy pixel)
    results = []
    for i in range(len(scores)):
        cx, cy, bw, bh = boxes[i]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 裁剪到图像范围
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            results.append({
                'box': [x1, y1, x2, y2],
                'score': float(scores[i]),
                'label': int(labels[i]),
                'class_name': COCO_CLASSES[int(labels[i])]
                if int(labels[i]) < len(COCO_CLASSES) else 'unknown',
            })

    return results


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """在图像上绘制检测结果 — 终端风格"""
    img = image.copy()

    if not detections:
        # 无检测结果也显示状态
        h, w = img.shape[:2]
        cv2.putText(img, '> NO DETECTIONS_', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        return img

    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = COLORS[det['label'] % len(COLORS)]
        score = det['score']
        name = det['class_name']

        # 终端风格框 (像素化直角矩形)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制角落加强标记 (终端风格装饰)
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 4)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 4)

        # 标签背景 (终端风格: 黑底绿字)
        label = f" [{name}] {score:.2f} "
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 0, 0), -1)
        cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw, y1), color, 1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 终端状态栏
    h, w = img.shape[:2]
    status = f"> DETECTED: {len(detections)} objects | FastDETR v1.0"
    cv2.rectangle(img, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.putText(img, status, (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return img


# ============================================================
#  WebSocket 风格的视频流 (MJPEG)
# ============================================================
class VideoProcessor:
    """视频处理器 — 逐帧检测并输出终端风格渲染结果"""
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        self.current_frame += 1

        # 检测
        dets = detect_frame(frame)

        # 渲染终端风格
        rendered = draw_detections(frame, dets)

        # 编码 JPEG
        _, jpeg = cv2.imencode('.jpg', rendered,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def get_progress(self):
        if self.total_frames > 0:
            return self.current_frame / self.total_frames * 100
        return 0


# 当前活跃的视频处理器
ACTIVE_PROCESSOR = None


# ============================================================
#  Flask 路由
# ============================================================
@app.route('/')
def index():
    """终端风格主页"""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """系统状态"""
    gpu_info = 'N/A'
    if torch.cuda.is_available():
        gpu_info = f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem//1024**3}GB)'

    return jsonify({
        'model_loaded': DETECTOR is not None,
        'device': str(DEVICE) if DEVICE else 'N/A',
        'gpu': gpu_info,
        'coco_classes': len(COCO_CLASSES),
        'timestamp': time.time(),
    })


@app.route('/api/detect_image', methods=['POST'])
def api_detect_image():
    """上传图片进行检测"""
    if 'image' not in request.files:
        return jsonify({'error': 'NO FILE UPLOADED'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename or 'upload.jpg')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 读取并检测
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'CANNOT READ IMAGE'}), 400

    detections = detect_frame(image, score_threshold=0.3)
    rendered = draw_detections(image, detections)

    # 保存渲染结果
    output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                f'result_{filename}')
    cv2.imwrite(output_path, rendered)

    return jsonify({
        'detections': detections,
        'count': len(detections),
        'result_url': f'/uploads/result_{filename}',
    })


@app.route('/api/detect_video', methods=['POST'])
def api_detect_video():
    """上传视频进行处理"""
    global ACTIVE_PROCESSOR

    if 'video' not in request.files:
        return jsonify({'error': 'NO VIDEO FILE UPLOADED'}), 400

    file = request.files['video']
    filename = secure_filename(file.filename or 'video.mp4')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 验证视频
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'CANNOT OPEN VIDEO'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ACTIVE_PROCESSOR = VideoProcessor(filepath)

    return jsonify({
        'filename': filename,
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'stream_url': '/api/video_feed',
    })


@app.route('/api/video_feed')
def api_video_feed():
    """MJPEG 视频流 — 实时检测结果"""
    global ACTIVE_PROCESSOR
    if ACTIVE_PROCESSOR is None:
        return "NO ACTIVE VIDEO", 400

    def generate():
        try:
            for jpeg_bytes in ACTIVE_PROCESSOR:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg_bytes + b'\r\n')
                time.sleep(1.0 / max(ACTIVE_PROCESSOR.fps, 1))
        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/video_progress')
def api_video_progress():
    """视频处理进度"""
    global ACTIVE_PROCESSOR
    if ACTIVE_PROCESSOR is None:
        return jsonify({'progress': 0, 'current': 0, 'total': 0})
    return jsonify({
        'progress': ACTIVE_PROCESSOR.get_progress(),
        'current': ACTIVE_PROCESSOR.current_frame,
        'total': ACTIVE_PROCESSOR.total_frames,
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """静态文件服务"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastDETR 终端风格前端')
    parser.add_argument('--model', type=str, default='',
                        help='模型 checkpoint 路径')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web 服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='监听地址')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    args = parser.parse_args()

    # 加载模型
    load_model(args.model)

    print(f"\n[系统] 终端风格前端已就绪")
    print(f"[系统] 打开浏览器访问: http://localhost:{args.port}")
    print(f"[系统] 按 Ctrl+C 停止\n")

    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True)
