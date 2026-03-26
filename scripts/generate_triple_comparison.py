import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from preprocess import preprocess_frame_for_inference, SquarePad, get_transform, weight_path, TEST_DATA_DIR, OUTPUT_DIR

# ==========================================
# 核心設定區
# ==========================================
YOLO_NATIVE_WEIGHTS = weight_path('yolo11l.pt')
YOLO_CUSTOM_WEIGHTS = weight_path('best.pt')
EFFNET_WEIGHTS = weight_path('best_efficientnet_b2_thermal.pth')

INPUT_DIR = os.path.join(TEST_DATA_DIR, 'images')
COMPARISON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'triple_comparison')

TITLE_1 = "1. Native YOLO (Baseline)"
TITLE_2 = "2. Custom YOLO (High Recall)"
TITLE_3 = "3. YOLO + EfficientNet (High Precision)"

CONF_THRESHOLD = 0.2

os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 模型初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 三架構對比圖生成器啟動中，推論設備: {device}")

yolo_native = YOLO(YOLO_NATIVE_WEIGHTS)
yolo_custom = YOLO(YOLO_CUSTOM_WEIGHTS)

class_names = ['background', 'person']
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
eff_model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))
eff_model = eff_model.to(device).eval().half()

transform = get_transform()

print("✅ 所有模型載入完成，開始批次處理照片...\n")

def draw_title(img, title, color=(255, 255, 255)):
    img_width = img.shape[1]
    cv2.rectangle(img, (0, 0), (img_width, 60), (0, 0, 0), -1)
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# ==========================================
# 批次讀取與處理圖片
# ==========================================
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

for filename in tqdm(image_files, desc="產生對比圖中"):
    img_path = os.path.join(INPUT_DIR, filename)
    original_img = cv2.imread(img_path)

    if original_img is None: continue
    img_height, img_width = original_img.shape[:2]

    img_1 = original_img.copy()
    img_2 = original_img.copy()
    img_3 = original_img.copy()

    inference_img = preprocess_frame_for_inference(original_img)

    # 狀況 1：原生 YOLO (Baseline)
    results_native = yolo_native.predict(source=inference_img, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results_native[0].boxes:
        cls_id = int(box.cls[0].item())
        label = yolo_native.names[cls_id]
        if label != 'person': continue

        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(img_1, f"YOLO: {int(conf*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    draw_title(img_1, TITLE_1, (200, 200, 200))

    # 狀況 2：自訓 YOLO (高召回率)
    custom_boxes = []
    results_custom = yolo_custom.predict(source=inference_img, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results_custom[0].boxes:
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        custom_boxes.append([x1, y1, x2, y2])

        cv2.rectangle(img_2, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_2, f"YOLO: {int(conf*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    draw_title(img_2, TITLE_2, (0, 255, 255))

    # 狀況 3：YOLO + EfficientNet (終極過濾)
    for pbox in custom_boxes:
        x1, y1, x2, y2 = max(0, pbox[0]), max(0, pbox[1]), min(img_width, pbox[2]), min(img_height, pbox[3])
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

        if label == 'background' or conf_score < CONF_THRESHOLD: continue

        cv2.rectangle(img_3, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img_3, f"EffNet: {int(conf_score*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    draw_title(img_3, TITLE_3, (0, 255, 0))

    separator = np.full((img_height, 10, 3), 255, dtype=np.uint8)
    combined_img = cv2.hconcat([img_1, separator, img_2, separator, img_3])

    output_path = os.path.join(COMPARISON_OUTPUT_DIR, f"triple_compare_{filename}")
    cv2.imwrite(output_path, combined_img)

print(f"\n🎉 處理完成！請至 '{COMPARISON_OUTPUT_DIR}' 資料夾查看超完美的 PPT 對比圖素材。")
