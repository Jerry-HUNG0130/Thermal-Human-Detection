import cv2
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional as F
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 影像處理工具區
# ==========================================
def preprocess_frame_for_inference(frame):
    """將影像進行去噪與 CLAHE 對比度增強"""
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img

class SquarePad:
    """自動補黑邊將長方形影像轉為正方形，防止 Resize 變形"""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, fill=0, padding_mode='constant')

def load_efficientnet(weight_path, device, num_classes=2):
    """載入自定義的 EfficientNet-B2 模型"""
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==========================================
# 主程式區
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用運算裝置: {device}")

    image_path = "test.jpg"
    output_path = "D:/side_project/result_twostage.jpg"
    
    yolo_weight = "best.pt"
    eff_weight = "best_efficientnet_b2_thermal.pth"

    print("載入 YOLO 與 EfficientNet 模型...")
    yolo_model = YOLO(yolo_weight)
    eff_model = load_efficientnet(eff_weight, device)

    # EfficientNet 的影像前處理 (加入 SquarePad 解決特徵變形)
    eff_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((288, 288)), # 依據你提供的設定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. 讀取原始影像 (供最後畫圖與輸出使用)
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("無法讀取圖片")
        return
    img_draw = original_img.copy()

    # 2. 雙流處理：產生供 AI 辨識的預處理影像
    inference_img = preprocess_frame_for_inference(original_img)
    h, w, _ = inference_img.shape

    # 3. 第一階段：YOLO 讀取預處理影像進行物件偵測
    results = yolo_model(inference_img, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])

        if cls_id != 0:
            continue

        # 確保座標不超出邊界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 4. 從「預處理過的高對比影像」中裁切 ROI 交給第二階段
        roi = inference_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)

        # 5. 第二階段：EfficientNet 分類驗證 (自動執行 SquarePad 與 Resize)
        input_tensor = eff_transform(roi_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = eff_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            eff_conf = float(probabilities[1].item()) # 索引 1 為人

        # 6. 依據最終信心度決定顏色
        if eff_conf >= 0.85:
            color = (0, 0, 255)      # 紅色
        elif eff_conf >= 0.70:
            color = (0, 255, 255)    # 黃色
        else:
            color = (0, 255, 0)      # 綠色

        # 7. 繪製於「原始影像」上
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"Two-Stage: {eff_conf:.2f}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 儲存結果
    cv2.imwrite(output_path, img_draw)
    print(f"[Two-Stage] 處理完成，已儲存至: {output_path}")

if __name__ == "__main__":
    main()