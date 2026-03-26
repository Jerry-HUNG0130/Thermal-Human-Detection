import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

TEST_IMAGES_DIR = os.path.join(TEST_DATA_DIR, 'images')
TEST_LABELS_DIR = os.path.join(TEST_DATA_DIR, 'labels')

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# IoU 計算與指標比對核心邏輯
# ==========================================
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_image(pred_boxes, gt_boxes, iou_thresh=0.5):
    tp, fp = 0, 0
    matched_gt = []
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    for pbox in pred_boxes:
        best_iou, best_gt_idx = 0, -1
        for i, gbox in enumerate(gt_boxes):
            if i in matched_gt: continue
            iou = calculate_iou(pbox[:4], gbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.append(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

# ==========================================
# 模型初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 啟動模型效能評估器，使用設備: {device}")

yolo_native = YOLO(YOLO_NATIVE_WEIGHTS)
yolo_custom = YOLO(YOLO_CUSTOM_WEIGHTS)

class_names = ['background', 'person']
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
eff_model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))
eff_model = eff_model.to(device).eval().half()

transform = get_transform()

def preprocess_for_effnet(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    return cv2.cvtColor(clahe.apply(smoothed), cv2.COLOR_GRAY2BGR)

# ==========================================
# 測試迴圈與數據收集
# ==========================================
metrics = {
    'Native YOLO': {'TP': 0, 'FP': 0, 'FN': 0},
    'Custom YOLO': {'TP': 0, 'FP': 0, 'FN': 0},
    'YOLO + EffNet': {'TP': 0, 'FP': 0, 'FN': 0}
}

image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
print(f"📦 找到 {len(image_files)} 張測試圖片，開始嚴格評測...")

for img_name in tqdm(image_files, desc="Evaluating Dataset"):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    label_path = os.path.join(TEST_LABELS_DIR, img_name.rsplit('.', 1)[0] + '.txt')

    frame = cv2.imread(img_path)
    if frame is None: continue
    img_h, img_w = frame.shape[:2]

    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if int(parts[0]) == 0:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, y1 = int((cx - w/2) * img_w), int((cy - h/2) * img_h)
                    x2, y2 = int((cx + w/2) * img_w), int((cy + h/2) * img_h)
                    gt_boxes.append([x1, y1, x2, y2])

    # 狀況 1: Native YOLO
    preds_native = []
    results = yolo_native.predict(source=frame, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results[0].boxes:
        if yolo_native.names[int(box.cls[0].item())] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            preds_native.append([x1, y1, x2, y2, float(box.conf[0].item())])

    tp, fp, fn = evaluate_image(preds_native, gt_boxes, IOU_THRESHOLD)
    metrics['Native YOLO']['TP'] += tp; metrics['Native YOLO']['FP'] += fp; metrics['Native YOLO']['FN'] += fn

    # 狀況 2: Custom YOLO
    preds_custom = []
    results = yolo_custom.predict(source=frame, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        preds_custom.append([x1, y1, x2, y2, float(box.conf[0].item())])

    tp, fp, fn = evaluate_image(preds_custom, gt_boxes, IOU_THRESHOLD)
    metrics['Custom YOLO']['TP'] += tp; metrics['Custom YOLO']['FP'] += fp; metrics['Custom YOLO']['FN'] += fn

    # 狀況 3: YOLO + EfficientNet
    preds_dual = []
    eff_frame = preprocess_for_effnet(frame)
    for pbox in preds_custom:
        x1, y1, x2, y2 = max(0, pbox[0]), max(0, pbox[1]), min(img_w, pbox[2]), min(img_h, pbox[3])
        crop = eff_frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        input_tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device).half()
        with torch.no_grad():
            outputs = eff_model(input_tensor)
            conf, pred = torch.max(torch.nn.functional.softmax(outputs, dim=1)[0], 0)
            if class_names[pred.item()] == 'person' and conf.item() >= CONF_THRESHOLD:
                preds_dual.append([x1, y1, x2, y2, conf.item()])

    tp, fp, fn = evaluate_image(preds_dual, gt_boxes, IOU_THRESHOLD)
    metrics['YOLO + EffNet']['TP'] += tp; metrics['YOLO + EffNet']['FP'] += fp; metrics['YOLO + EffNet']['FN'] += fn

# ==========================================
# 計算指標與繪製圖表
# ==========================================
results_list = []
for name, m in metrics.items():
    tp, fp, fn = m['TP'], m['FP'], m['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    results_list.append([name, precision*100, recall*100, f1*100, tp, fp, fn])

df = pd.DataFrame(results_list, columns=['Model', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'TP', 'FP', 'FN'])
df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_metrics.csv'), index=False)
print("\n📊 === 測試集評估報告 ===")
print(df.to_string(index=False))

labels = df['Model']
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, df['Precision (%)'], width, label='Precision (精準率/抗誤報)', color='#4CAF50')
rects2 = ax.bar(x, df['Recall (%)'], width, label='Recall (召回率/抗漏報)', color='#2196F3')
rects3 = ax.bar(x + width, df['F1-Score (%)'], width, label='F1-Score (綜合實力)', color='#FF9800')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Thermal Imaging Target Detection: Model Architecture Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(loc='lower right')
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

autolabel(rects1); autolabel(rects2); autolabel(rects3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'architecture_comparison_chart.png')
plt.savefig(chart_path, dpi=300)
print(f"\n🎉 完美！報表與比較圖表已儲存至 '{chart_path}'。可以直接貼入你的 PPT 囉！")
