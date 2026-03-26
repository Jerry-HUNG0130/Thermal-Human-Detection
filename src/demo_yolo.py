import cv2
import torch
import numpy as np
import mss
import time
import winsound
import threading
from ultralytics import YOLO
from preprocess import preprocess_frame_for_inference, weight_path

# ==========================================
# 警報系統設定區
# ==========================================
ALERT_CONF_THRESHOLD = 0.85
BEEP_COOLDOWN = 1.0
last_beep_time = 0

def play_alert_sound():
    winsound.Beep(1000, 300)

# ==========================================
# 模型與環境初始化 (純 YOLO 原生對照組)
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 面試展示對照組 (純原生 YOLO) 啟動中，推論設備: {device}")

yolo_model = YOLO(weight_path('yolo11n.pt'))
print("✅ 原生 YOLO 模型載入成功")

# ==========================================
# 設定螢幕擷取區域與 AI 暖機 (Warmup)
# ==========================================
monitor_region = {"top": 50, "left": 10, "width": 950, "height": 1100}
print(f"🎥 設定擷取螢幕區域: {monitor_region}")

print("🔥 正在對 AI 模型進行暖機 (Warmup)，請稍候...")
dummy_frame = np.zeros((monitor_region["height"], monitor_region["width"], 3), dtype=np.uint8)

_ = yolo_model.predict(source=dummy_frame, conf=0.5, half=True, verbose=False)

print("✅ 暖機完成！系統已全速啟動。")
print(f"🔔 警報系統已啟動：當偵測信心度 >= {ALERT_CONF_THRESHOLD*100}% 時將發出提示音。")
print("💡 提示：點擊辨識視窗並按下鍵盤 'q' 鍵即可安全退出程式。")

# ==========================================
# 即時推論與顯示主迴圈
# ==========================================
cv2.namedWindow("yolo11l", cv2.WINDOW_NORMAL)
cv2.resizeWindow("yolo11l", 400, 600)

with mss.mss() as sct:
    while True:
        start_time = time.time()

        screen_shot = sct.grab(monitor_region)
        original_frame = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
        img_height, img_width = original_frame.shape[:2]

        inference_frame = preprocess_frame_for_inference(original_frame)
        results = yolo_model.predict(source=inference_frame, conf=0.5, iou=0.5, half=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_model.names[cls_id]
                conf_score = float(box.conf[0].item())

                if label != 'person':
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)

                if conf_score >= ALERT_CONF_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_beep_time > BEEP_COOLDOWN:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        last_beep_time = current_time

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

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(original_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("yolo11l", original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")
