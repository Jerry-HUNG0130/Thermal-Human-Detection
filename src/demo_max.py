import cv2
import torch
import numpy as np
import mss
import time
import winsound
import threading
from torchvision import models
from ultralytics import YOLO
from PIL import Image
from preprocess import preprocess_frame_for_inference, get_transform, weight_path

# ==========================================
# 核心設定區
# ==========================================
YOLO_NATIVE_WEIGHTS = weight_path('yolo11n.pt')
YOLO_CUSTOM_WEIGHTS = weight_path('best.pt')
EFFNET_WEIGHTS = weight_path('best_efficientnet_b2_thermal.pth')

MONITOR_REGION = {"top": 50, "left": 10, "width": 950, "height": 1100}

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
WINDOW_NAME_NATIVE = "YOLO"
WINDOW_NAME_DUAL = "YOLO+EfficientNet"

ALERT_CONF_THRESHOLD = 0.85
BEEP_COOLDOWN = 1.0
last_beep_time = 0

def play_alert_sound():
    winsound.Beep(1000, 300)

# ==========================================
# 模型與環境初始化 (啟動 GPU FP16 加速)
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 雙視窗對比系統啟動中，推論設備: {device}")

yolo_native = YOLO(YOLO_NATIVE_WEIGHTS)
print(f"✅ 對照組模型載入: {YOLO_NATIVE_WEIGHTS}")

yolo_custom = YOLO(YOLO_CUSTOM_WEIGHTS)
print(f"✅ 實驗組一階模型載入: {YOLO_CUSTOM_WEIGHTS}")

class_names = ['background', 'person']
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
eff_model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))
eff_model = eff_model.to(device)
eff_model.eval()
eff_model.half()
print(f"✅ 實驗組二階模型載入: {EFFNET_WEIGHTS} (FP16)")

transform = get_transform()

# ==========================================
# AI 暖機 (Warmup)
# ==========================================
print("🔥 正在對所有 AI 模型進行暖機 (Warmup)，請稍候...")
dummy_frame = np.zeros((MONITOR_REGION["height"], MONITOR_REGION["width"], 3), dtype=np.uint8)

_ = yolo_native.predict(source=dummy_frame, conf=0.5, half=True, verbose=False)
_ = yolo_custom.predict(source=dummy_frame, conf=0.5, half=True, verbose=False)

dummy_pil = Image.fromarray(dummy_frame)
dummy_tensor = transform(dummy_pil).unsqueeze(0).to(device).half()
with torch.no_grad():
    _ = eff_model(dummy_tensor)

print("✅ 暖機完成！系統已全速啟動。")

# ==========================================
# 顯示視窗設定
# ==========================================
cv2.namedWindow(WINDOW_NAME_NATIVE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME_NATIVE, WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow(WINDOW_NAME_DUAL, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME_DUAL, WINDOW_WIDTH, WINDOW_HEIGHT)

# ==========================================
# 即時推論與顯示主迴圈
# ==========================================
with mss.mss() as sct:
    while True:
        start_time = time.time()

        screen_shot = sct.grab(MONITOR_REGION)
        original_frame = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
        img_height, img_width = original_frame.shape[:2]

        frame_native = original_frame.copy()
        frame_dual = original_frame.copy()

        inference_frame = preprocess_frame_for_inference(original_frame)

        # 系統 A：對照組 (純原生 YOLO)
        results_native = yolo_native.predict(source=inference_frame, conf=0.5, iou=0.5, half=True, verbose=False)
        for result in results_native:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_native.names[cls_id]
                conf_score = float(box.conf[0].item())

                if label != 'person': continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame_native, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame_native, f"YOLO: {int(conf_score*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 系統 B：實驗組 (自訓 YOLO + EfficientNet)
        results_custom = yolo_custom.predict(source=inference_frame, conf=0.5, iou=0.5, half=True, verbose=False)
        for result in results_custom:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)

                cropped_img_cv = inference_frame[y1:y2, x1:x2]
                if cropped_img_cv.size == 0: continue

                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB))
                input_tensor = transform(cropped_img_pil).unsqueeze(0).to(device).half()

                with torch.no_grad():
                    outputs = eff_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, preds = torch.max(probabilities, 0)
                    label = class_names[preds.item()]
                    conf_score = confidence.item()

                if label == 'background' or conf_score < 0.5: continue

                if conf_score >= ALERT_CONF_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_beep_time > BEEP_COOLDOWN:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        last_beep_time = current_time

                color = (0, 0, 255) if conf_score >= 0.8 else (0, 165, 255)
                cv2.rectangle(frame_dual, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_dual, f"EffNet: {int(conf_score*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame_native, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_dual, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME_NATIVE, frame_native)
        cv2.imshow(WINDOW_NAME_DUAL, frame_dual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")
