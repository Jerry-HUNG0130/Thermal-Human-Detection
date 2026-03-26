import cv2
import torch
import numpy as np
import mss
import time
import winsound  # ★ 引入 Windows 內建音效套件
import threading # ★ 引入多執行緒，確保發聲時不卡頓
from ultralytics import YOLO

# ==========================================
# [新增] 警報系統設定區
# ==========================================
ALERT_CONF_THRESHOLD = 0.85  # 手動設定：信心度大於等於 85% 時發出警報
BEEP_COOLDOWN = 1.0          # 手動設定：警報冷卻時間(秒)，避免連續狂嗶
last_beep_time = 0           # 紀錄上次嗶聲的時間

def play_alert_sound():
    """在背景發出嗶聲的函式"""
    winsound.Beep(1000, 300) 

# ==========================================
# 1. 影像預處理函式 (與你的雙架構保持一致的預處理)
# ==========================================
def preprocess_frame_for_inference(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img

# ==========================================
# 2. 模型與環境初始化 (純 YOLO 原生對照組)
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 面試展示對照組 (純原生 YOLO) 啟動中，推論設備: {device}")

# 載入原生 YOLO 權重
yolo_model = YOLO('yolo11n.pt') 
print("✅ 原生 YOLO 模型載入成功")

# ==========================================
# 3. 設定螢幕擷取區域與 AI 暖機 (Warmup)
# ==========================================
monitor_region = {"top": 50, "left": 10, "width": 950, "height": 1100}
print(f"🎥 設定擷取螢幕區域: {monitor_region}")

print("🔥 正在對 AI 模型進行暖機 (Warmup)，請稍候...")
# 產生一張全黑的假圖片進行暖機
dummy_frame = np.zeros((monitor_region["height"], monitor_region["width"], 3), dtype=np.uint8)

# 讓 YOLO 先跑一次
_ = yolo_model.predict(source=dummy_frame, conf=0.5, half=True, verbose=False)

print("✅ 暖機完成！系統已全速啟動。")
print(f"🔔 警報系統已啟動：當偵測信心度 >= {ALERT_CONF_THRESHOLD*100}% 時將發出提示音。")
print("💡 提示：點擊辨識視窗並按下鍵盤 'q' 鍵即可安全退出程式。")

# ==========================================
# 4. 即時推論與顯示主迴圈
# ==========================================
cv2.namedWindow("yolo11l", cv2.WINDOW_NORMAL)
cv2.resizeWindow("yolo11l", 400, 600) 

with mss.mss() as sct:
    while True:
        start_time = time.time() # 用於計算 FPS
        
        # 1. 抓取螢幕畫面
        screen_shot = sct.grab(monitor_region)
        original_frame = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
        img_height, img_width = original_frame.shape[:2]

        # 2. 影像預處理
        inference_frame = preprocess_frame_for_inference(original_frame)

        # 3. YOLO 偵測
        results = yolo_model.predict(source=inference_frame, conf=0.5, iou=0.5, half=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 取得 YOLO 原生判斷的類別 (例如：0 代表 person)
                cls_id = int(box.cls[0].item())
                label = yolo_model.names[cls_id]
                
                # 取得信心度
                conf_score = float(box.conf[0].item())
                
                # 為了公平對比，我們只標記出 YOLO 認為是 "person" 的目標
                if label != 'person':
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)

                # ==========================================
                # ★ 觸發音效警報邏輯 ★
                # ==========================================
                if conf_score >= ALERT_CONF_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_beep_time > BEEP_COOLDOWN:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        last_beep_time = current_time

                # 視覺渲染顏色判斷
                if conf_score >= 0.8:
                    color = (0, 0, 255)      # 高確信：紅色
                    thickness = 3
                else:
                    color = (0, 165, 255)    # 疑似：橘色
                    thickness = 2

                # 畫 Bounding Box
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

        # 4. 計算並在左上角顯示 FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(original_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 5. 原生視窗顯示結果
        cv2.imshow("yolo11l", original_frame)

        # 監聽鍵盤事件，按下 'q' 退出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放資源並關閉視窗
cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")