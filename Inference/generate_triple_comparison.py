import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ==========================================
# ⚙️ [核心設定區] 三架構對比生成器參數
# ==========================================
# 1. 權重檔案路徑
YOLO_NATIVE_WEIGHTS = 'yolo11l.pt'                       # 狀況一：原生 YOLO
YOLO_CUSTOM_WEIGHTS = 'best.pt'                          # 狀況二：自訓 YOLO
EFFNET_WEIGHTS = 'best_efficientnet_b2_thermal.pth'      # 狀況三：自訓 EfficientNet

# 2. 資料夾路徑設定
INPUT_DIR = 'test_images'          # 準備用來測試的原始照片 (請挑選幾張容易誤判的石頭/反光圖)
OUTPUT_DIR = 'triple_comparison'   # 程式自動產出的三併圖存放區

# 3. 顯示文字與標題設定
TITLE_1 = "1. Native YOLO (Baseline)"
TITLE_2 = "2. Custom YOLO (High Recall)"
TITLE_3 = "3. YOLO + EfficientNet (High Precision)"

# 4. 推論超參數
CONF_THRESHOLD = 0.5  # 信心度大於 50% 才顯示

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 影像預處理與模型前置作業
# ==========================================
def preprocess_frame_for_inference(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = int((max_wh - w) / 2), int((max_wh - h) / 2)
        return transforms.functional.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), fill=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 三架構對比圖生成器啟動中，推論設備: {device}")

# 載入所有模型
yolo_native = YOLO(YOLO_NATIVE_WEIGHTS) 
yolo_custom = YOLO(YOLO_CUSTOM_WEIGHTS)

class_names = ['background', 'person'] 
eff_model = models.efficientnet_b2(weights=None)
num_ftrs = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
eff_model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device)) 
eff_model = eff_model.to(device).eval().half() 

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("✅ 所有模型載入完成，開始批次處理照片...\n")

# 定義標題繪製函式
def draw_title(img, title, color=(255, 255, 255)):
    img_width = img.shape[1]
    cv2.rectangle(img, (0, 0), (img_width, 60), (0, 0, 0), -1)
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# ==========================================
# 2. 批次讀取與處理圖片
# ==========================================
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

for filename in tqdm(image_files, desc="產生對比圖中"):
    img_path = os.path.join(INPUT_DIR, filename)
    original_img = cv2.imread(img_path)
    
    if original_img is None: continue
    img_height, img_width = original_img.shape[:2]
    
    # 複製三張乾淨的畫布
    img_1 = original_img.copy()
    img_2 = original_img.copy()
    img_3 = original_img.copy()
    
    # 共用的 AI 預處理 (濾波與對比強化)
    inference_img = preprocess_frame_for_inference(original_img)

    # ---------------------------------------------------------
    # 🟢 狀況 1：原生 YOLO (Baseline)
    # ---------------------------------------------------------
    results_native = yolo_native.predict(source=inference_img, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results_native[0].boxes:
        cls_id = int(box.cls[0].item())
        label = yolo_native.names[cls_id]
        if label != 'person': continue
        
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 165, 255), 3) # 橘色框
        cv2.putText(img_1, f"YOLO: {int(conf*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    draw_title(img_1, TITLE_1, (200, 200, 200)) # 灰色標題

    # ---------------------------------------------------------
    # 🔵 狀況 2：自訓 YOLO (高召回率，但可能有假警報)
    # ---------------------------------------------------------
    custom_boxes = [] # 收集起來給狀況3用
    results_custom = yolo_custom.predict(source=inference_img, conf=CONF_THRESHOLD, half=True, verbose=False)
    for box in results_custom[0].boxes:
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        custom_boxes.append([x1, y1, x2, y2])
        
        cv2.rectangle(img_2, (x1, y1), (x2, y2), (0, 255, 255), 3) # 黃色框
        cv2.putText(img_2, f"YOLO: {int(conf*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    draw_title(img_2, TITLE_2, (0, 255, 255)) # 黃色標題

    # ---------------------------------------------------------
    # 🔴 狀況 3：YOLO + EfficientNet (終極過濾)
    # ---------------------------------------------------------
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

        # 最終確認為人的畫紅色框
        cv2.rectangle(img_3, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img_3, f"EffNet: {int(conf_score*100)}%", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    draw_title(img_3, TITLE_3, (0, 255, 0)) # 綠色標題，代表最佳解

    # ---------------------------------------------------------
    # 3. 圖像水平合併 (Side-by-Side-by-Side) 與存檔
    # ---------------------------------------------------------
    # 為了避免三張圖並排太寬，我們加入白線作為分隔線
    separator = np.full((img_height, 10, 3), 255, dtype=np.uint8)
    combined_img = cv2.hconcat([img_1, separator, img_2, separator, img_3]) 
    
    output_path = os.path.join(OUTPUT_DIR, f"triple_compare_{filename}")
    cv2.imwrite(output_path, combined_img)

print(f"\n🎉 處理完成！請至 '{OUTPUT_DIR}' 資料夾查看超完美的 PPT 對比圖素材。")