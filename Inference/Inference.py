import cv2
import torch
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. 自定義前處理：防形變補邊 (必須與訓練時完全一致)
# ==========================================
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

# ==========================================
# 2. 模型與環境初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"推論設備載入中: {device}")

# 載入 YOLOv11l
yolo_model = YOLO('yolo11l.pt') 

# 載入 EfficientNet-B2
class_names = ['background', 'person'] 
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
# 請確保這裡的權重檔名正確
eff_model.load_state_dict(torch.load('efficientnet_b2_thermal.pth', map_location=device)) 
eff_model = eff_model.to(device)
eff_model.eval()

# 推論前處理通道
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 讀取影像與雙階段推論
# ==========================================
# 替換成你會遇到問題的那張圖片路徑來測試
img_path = 'image_1.jpg' 
image_cv = cv2.imread(img_path)

if image_cv is None:
    print("找不到圖片，請確認路徑！")
    exit()

# 獲取圖片高度，用於邊界檢查
img_height, img_width = image_cv.shape[:2]

# 階段一：YOLO 偵測
results = yolo_model.predict(source=image_cv, conf=0.01, classes=[0], verbose=False)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 確保座標不超出圖片範圍的防呆機制
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        cropped_img_cv = image_cv[y1:y2, x1:x2]
        if cropped_img_cv.size == 0:
            continue
            
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB))
        
        # 階段二：EfficientNet 分類
        input_tensor = transform(cropped_img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = eff_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, preds = torch.max(probabilities, 0)
            
            label = class_names[preds.item()]
            conf_score = confidence.item()
        
        # ==========================================
        # 4. 救難實務 UI/UX 渲染邏輯 (含彈性位置)
        # ==========================================
        
        # 原則 1：過濾背景
        if label == 'background':
            continue 
            
        # 原則 2：直覺暗示 (顏色與粗細)
        if conf_score >= 0.8:
            color = (0, 0, 255)      # 紅色
            thickness = 3
        else:
            color = (0, 165, 255)    # 橘色
            thickness = 2

        # 畫出 Bounding Box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, thickness)
        
        # 原則 3：視覺防護與【彈性位置調整】
        display_text = f"P {int(conf_score * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        buffer = 5 # 文字與框線的間距
        
        # 取得文字的寬高
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
        
        # 計算文字背景框的預設位置 (顯示在上方)
        text_bg_y1 = y1 - text_height - buffer * 2
        text_bg_y2 = y1
        text_origin_y = y1 - buffer

        # 【關鍵修改】：檢查是否超出圖片上緣
        if text_bg_y1 < 0:
            # 如果空間不足，改為顯示在 Bounding Box 內部頂端
            text_bg_y1 = y1
            text_bg_y2 = y1 + text_height + buffer * 2
            text_origin_y = y1 + text_height + buffer + baseline // 2

        # 確保文字背景框的 X 座標也不會超出左邊界
        text_bg_x1 = max(x1, 0)
        text_bg_x2 = text_bg_x1 + text_width

        # 畫一個實心的底色方塊
        cv2.rectangle(image_cv, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        
        # 畫上白色的文字
        text_color = (255, 255, 255)
        cv2.putText(image_cv, display_text, (text_bg_x1, text_origin_y), font, font_scale, text_color, font_thickness)

# 5. 儲存最終結果
output_path = 'ui_smart_result.jpg'
cv2.imwrite(output_path, image_cv)
print(f"推論完成！具備彈性標籤位置的結果已儲存為 {output_path}")