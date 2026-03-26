import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, Response, redirect, url_for
from werkzeug.utils import secure_filename
from torchvision import models
from ultralytics import YOLO
from PIL import Image
from preprocess import preprocess_frame_for_inference, get_transform, weight_path, TEMPLATES_DIR

app = Flask(__name__, template_folder=TEMPLATES_DIR)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# 模型與環境初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Flask 伺服器啟動中，推論設備載入: {device}")

try:
    yolo_model = YOLO(weight_path('yolo11l.pt'))
    print("✅ YOLO 模型載入成功")
except Exception as e:
    print(f"❌ YOLO 模型載入失敗: {e}")

class_names = ['background', 'person']
try:
    eff_model = models.efficientnet_b2(weights=None)
    num_ftrs = eff_model.classifier[1].in_features
    eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
    eff_model.load_state_dict(torch.load(weight_path('best_efficientnet_b2_thermal.pth'), map_location=device))
    eff_model = eff_model.to(device)
    eff_model.eval()
    print("✅ EfficientNet 模型載入成功")
except Exception as e:
    print(f"❌ EfficientNet 模型載入失敗: {e}")

transform = get_transform()

# ==========================================
# 影片逐幀推論生成器 (雙流處理 + 儲存 MP4)
# ==========================================
def generate_frames(video_path, filename):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"📼 開始處理影片，結果將同步儲存於: {output_path}")

    while cap.isOpened():
        success, original_frame = cap.read()
        if not success:
            break

        img_height, img_width = original_frame.shape[:2]
        inference_frame = preprocess_frame_for_inference(original_frame)
        results = yolo_model.predict(source=inference_frame, conf=0.05, iou=0.5, classes=[0], verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)

                cropped_img_cv = inference_frame[y1:y2, x1:x2]
                if cropped_img_cv.size == 0:
                    continue

                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB))
                input_tensor = transform(cropped_img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = eff_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, preds = torch.max(probabilities, 0)

                    label = class_names[preds.item()]
                    conf_score = confidence.item()

                if label == 'background' or conf_score < 0.4:
                    continue

                if conf_score >= 0.8:
                    color = (0, 0, 255)
                    thickness = 3
                else:
                    color = (0, 165, 255)
                    thickness = 2

                cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness)

                display_text = f"P {int(conf_score * 100)}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                buffer = 5

                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

                text_bg_y1 = y1 - text_height - buffer * 2
                text_bg_y2 = y1
                text_origin_y = y1 - buffer

                if text_bg_y1 < 0:
                    text_bg_y1 = y1
                    text_bg_y2 = y1 + text_height + buffer * 2
                    text_origin_y = y1 + text_height + buffer + baseline // 2

                text_bg_x1 = max(x1, 0)
                text_bg_x2 = text_bg_x1 + text_width

                cv2.rectangle(original_frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
                cv2.putText(original_frame, display_text, (text_bg_x1, text_origin_y), font, font_scale, (255, 255, 255), font_thickness)

        out.write(original_frame)

        ret, buffer = cv2.imencode('.jpg', original_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
    print(f"✅ 影片辨識與存檔完成！")

# ==========================================
# Flask 路由設定
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return redirect(request.url)

        file = request.files['video_file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('index.html', video_name=filename)

    return render_template('index.html', video_name=None)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath, filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
