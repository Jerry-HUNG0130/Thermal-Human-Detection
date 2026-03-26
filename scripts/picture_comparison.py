import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from preprocess import preprocess_frame_for_inference, SquarePad, get_transform, weight_path, TEST_DATA_DIR, OUTPUT_DIR

# ==========================================
# 核心設定區
# ==========================================
INPUT_DIR = os.path.join(TEST_DATA_DIR, 'images')
COMPARISON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'dual_comparison')

YOLO_NATIVE_WEIGHTS = weight_path('yolo11l.pt')
YOLO_CUSTOM_WEIGHTS = weight_path('best.pt')
EFFNET_WEIGHTS = weight_path('best_efficientnet_b2_thermal.pth')

TITLE_NATIVE = "Native YOLO (Baseline)"
TITLE_DUAL = "Custom YOLO + EfficientNet"

os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 模型初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 靜態對比生成器啟動中，推論設備: {device}")

yolo_native = YOLO(YOLO_NATIVE_WEIGHTS)
yolo_custom = YOLO(YOLO_CUSTOM_WEIGHTS)

class_names = ['background', 'person']
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
eff_model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))
eff_model = eff_model.to(device)
eff_model.eval()
eff_model.half()

transform = get_transform()

print("✅ 所有模型載入完成，開始批次處理照片...\n")

# ==========================================
# 批次讀取與處理圖片
# ==========================================
valid_extensions = ('.jpg', '.jpeg', '.png')
processed_count = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(valid_extensions):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    original_img = cv2.imread(img_path)

    if original_img is None:
        print(f"⚠️ 無法讀取圖片: {filename}")
        continue

    img_height, img_width = original_img.shape[:2]

    img_native = original_img.copy()
    img_dual = original_img.copy()

    inference_img = preprocess_frame_for_inference(original_img)

    # 系統 A：對照組 (純原生 YOLO)
    results_native = yolo_native.predict(source=inference_img, conf=0.5, iou=0.5, half=True, verbose=False)
    for result in results_native:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = yolo_native.names[cls_id]
            conf_score = float(box.conf[0].item())

            if label != 'person': continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img_native, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(img_native, f"YOLO: {int(conf_score*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.rectangle(img_native, (0, 0), (img_width, 60), (0, 0, 0), -1)
    cv2.putText(img_native, TITLE_NATIVE, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # 系統 B：實驗組 (自訓 YOLO + EfficientNet)
    results_custom = yolo_custom.predict(source=inference_img, conf=0.5, iou=0.5, half=True, verbose=False)
    for result in results_custom:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            cropped_img_cv = inference_img[y1:y2, x1:x2]
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

            color = (0, 0, 255) if conf_score >= 0.8 else (0, 165, 255)
            cv2.rectangle(img_dual, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_dual, f"EffNet: {int(conf_score*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.rectangle(img_dual, (0, 0), (img_width, 60), (0, 0, 0), -1)
    cv2.putText(img_dual, TITLE_DUAL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    combined_img = cv2.hconcat([img_native, img_dual])

    output_path = os.path.join(COMPARISON_OUTPUT_DIR, f"compare_{filename}")
    cv2.imwrite(output_path, combined_img)
    processed_count += 1
    print(f"📸 成功生成對比圖: {output_path}")

print(f"\n🎉 處理完成！總共產出了 {processed_count} 張靜態對比圖。請至 '{COMPARISON_OUTPUT_DIR}' 資料夾查看。")
