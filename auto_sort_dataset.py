import os
import cv2
from ultralytics import YOLO

# ================= 參數設定區 =================
MODEL_PATH = '/home/student/side_project/YOLO/runs/detect/GCP_L4_Rescue/v11l_thermal_full_run/weights/best.pt'
SOURCE_IMAGES = '/home/student/side_project/YOLO/dataset/images/test'  # 要拿來生成 EfficientNet 資料的圖庫

# 業界建議的信心值門檻
CONF_HIGH = 0.70  # >= 70% 自動歸類為明確的人 (person)
CONF_LOW = 0.30   # <= 30% 自動歸類為明確的背景 (background)
# 介於 30% ~ 70% 的會自動進入 review 資料夾供人工審查

# 建立輸出資料夾
OUTPUT_DIRS = ['/home/student/side_project/EfficientNet/dataset/test/person', 
               '/home/student/side_project/EfficientNet/dataset/test/background', 
               '/home/student/side_project/EfficientNet/dataset/test/review']
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)
# ==============================================

# 1. 載入模型
model = YOLO(MODEL_PATH)

# 2. 進行推論 (將 stream=True 開啟，避免記憶體爆滿)
# 這裡的 conf=0.05 設定極低，是為了讓 YOLO 把所有"可能有東西"的框都抓出來給我們分流
results = model.predict(source=SOURCE_IMAGES, conf=0.05, stream=True)

print("開始進行推論、裁切與自動分流...")

crop_count = {'person': 0, 'background': 0, 'review': 0}

for r in results:
    # 取得原始圖片陣列與檔名
    img = r.orig_img
    base_filename = os.path.basename(r.path).split('.')[0]
    
    # 遍歷這張圖片上的每一個預測框
    for i, box in enumerate(r.boxes):
        conf = float(box.conf[0]) # 取得信心分數
        
        # 取得邊界框的座標 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 裁切圖片 (注意 OpenCV 的陣列切片是 [y:y, x:x])
        crop_img = img[y1:y2, x1:x2]
        
        # 避免裁出長寬為 0 的無效圖片
        if crop_img.size == 0:
            continue
            
        # ================= 分流邏輯 =================
        if conf >= CONF_HIGH:
            folder = '/home/student/side_project/EfficientNet/dataset/test/person'
            crop_count['person'] += 1
        elif conf <= CONF_LOW:
            folder = '/home/student/side_project/EfficientNet/dataset/test/background'
            crop_count['background'] += 1
        else:
            folder = '/home/student/side_project/EfficientNet/dataset/test/review'
            crop_count['review'] += 1
            
        # ================= 命名與存檔 =================
        # 檔名格式: 原檔名_crop編號_conf信心分數.jpg
        # 範例: DJI_001_crop0_conf0.85.jpg
        save_name = f"{base_filename}_crop{i}_conf{conf:.2f}.jpg"
        save_path = os.path.join(folder, save_name)
        
        cv2.imwrite(save_path, crop_img)

print("\n✅ 分流作業完成！統計結果：")
print(f"👉 自動標為 Person (>= {CONF_HIGH*100}%): {crop_count['person']} 張")
print(f"👉 自動標為 Background (<= {CONF_LOW*100}%): {crop_count['background']} 張")
print(f"👉 需人工審查 Review ({CONF_LOW*100}% ~ {CONF_HIGH*100}%): {crop_count['review']} 張")