import cv2
import torch
import numpy as np
import mss
import time
import winsound  
import threading 
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# ==========================================
# ⚙️ [核心設定區] 面試前請在此微調所有參數
# ==========================================
# 1. 權重檔案路徑
YOLO_NATIVE_WEIGHTS = 'yolo11n.pt'                       # 對照組：YOLO 原生權重
YOLO_CUSTOM_WEIGHTS = 'best.pt'                          # 實驗組：YOLO 自訓權重 (尋找熱源)
EFFNET_WEIGHTS = 'best_efficientnet_b2_thermal.pth'      # 實驗組：EfficientNet 權重 (二次確認)

# 2. 視窗擷取區域範圍 (對應手機投影畫面)
MONITOR_REGION = {"top": 50, "left": 10, "width": 950, "height": 1100}

# 3. 顯示視窗設定 (畫面縮放大小)
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
WINDOW_NAME_NATIVE = "YOLO"                              # 左視窗名稱
WINDOW_NAME_DUAL = "YOLO+EfficientNet"                   # 右視窗名稱

# 4. 警報系統設定
ALERT_CONF_THRESHOLD = 0.85  # 信心度大於等於 85% 時發出警報 (僅在雙架構視窗觸發)
BEEP_COOLDOWN = 1.0          # 警報冷卻時間(秒)
last_beep_time = 0           

def play_alert_sound():
    """在背景發出嗶聲的函式"""
    winsound.Beep(1000, 300) 

# ==========================================
# 1. 影像預處理函式 (專供 AI 推論使用)
# ==========================================
def preprocess_frame_for_inference(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img

# ==========================================
# 2. 模型與環境初始化 (啟動 GPU FP16 加速)
# ==========================================
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = int((max_wh - w) / 2), int((max_wh - h) / 2)
        return transforms.functional.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), fill=0, padding_mode='constant')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 雙視窗對比系統啟動中，推論設備: {device}")

# 載入原生 YOLO (對照組)
yolo_native = YOLO(YOLO_NATIVE_WEIGHTS) 
print(f"✅ 對照組模型載入: {YOLO_NATIVE_WEIGHTS}")

# 載入自訓 YOLO + EfficientNet (實驗組)
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

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. AI 暖機 (Warmup)
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
# 4. 顯示視窗設定
# ==========================================
cv2.namedWindow(WINDOW_NAME_NATIVE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME_NATIVE, WINDOW_WIDTH, WINDOW_HEIGHT) 

cv2.namedWindow(WINDOW_NAME_DUAL, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME_DUAL, WINDOW_WIDTH, WINDOW_HEIGHT) 

# ==========================================
# 5. 即時推論與顯示主迴圈
# ==========================================
with mss.mss() as sct:
    while True:
        start_time = time.time()
        
        # 1. 抓取螢幕畫面
        screen_shot = sct.grab(MONITOR_REGION)
        original_frame = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
        img_height, img_width = original_frame.shape[:2]

        # 為了讓兩個視窗畫不同的框，我們必須複製兩張獨立的畫布
        frame_native = original_frame.copy()
        frame_dual = original_frame.copy()

        # 2. 共用的 AI 預處理
        inference_frame = preprocess_frame_for_inference(original_frame)

        # ==========================================
        # 🟢 系統 A：對照組 (純原生 YOLO)
        # ==========================================
        results_native = yolo_native.predict(source=inference_frame, conf=0.5, iou=0.5, half=True, verbose=False)
        for result in results_native:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_native.names[cls_id]
                conf_score = float(box.conf[0].item())
                
                # 原生模型只標記 'person'
                if label != 'person': continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame_native, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame_native, f"YOLO: {int(conf_score*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # ==========================================
        # 🔵 系統 B：實驗組 (自訓 YOLO + EfficientNet)
        # ==========================================
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
                    
                # 觸發音效警報 (僅實驗組發出聲音，避免原生組狂叫)
                if conf_score >= ALERT_CONF_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_beep_time > BEEP_COOLDOWN:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        last_beep_time = current_time

                color = (0, 0, 255) if conf_score >= 0.8 else (0, 165, 255)
                cv2.rectangle(frame_dual, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_dual, f"EffNet: {int(conf_score*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ==========================================
        # 6. 計算 FPS 與 顯示畫面
        # ==========================================
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame_native, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_dual, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME_NATIVE, frame_native)
        cv2.imshow(WINDOW_NAME_DUAL, frame_dual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")