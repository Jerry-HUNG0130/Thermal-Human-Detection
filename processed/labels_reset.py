import os

def convert_polygon_to_bbox(coords):
    """
    將多邊形頂點座標轉換為 YOLO 標準的 (x_center, y_center, width, height)。
    支援任意數量的頂點 (只要是 x, y 成對出現)。
    """
    # 將座標分為 X 群組與 Y 群組
    xs = coords[0::2]
    ys = coords[1::2]
    
    # 找出極值
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # 防呆機制：確保座標不會超出 0.0 ~ 1.0 的範圍
    x_min = max(0.0, min(1.0, x_min))
    x_max = max(0.0, min(1.0, x_max))
    y_min = max(0.0, min(1.0, y_min))
    y_max = max(0.0, min(1.0, y_max))
    
    # 計算 YOLO 所需的數值
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2.0)
    y_center = y_min + (height / 2.0)
    
    return x_center, y_center, width, height

def robust_label_formatter(input_dir, output_dir, target_class_id=3, new_class_id=0):
    """
    批次處理標籤：過濾特定 ID、統一轉換為 0、並將多邊形自動轉為標準矩形框。
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    
    print(f"🔍 找到 {len(txt_files)} 個標籤檔，開始執行終極格式化...")

    processed_count = 0
    polygon_converted_count = 0
    kept_lines_count = 0

    for filename in txt_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            
            if len(parts) > 0:
                class_id = int(parts[0])
                
                # 只處理我們想要的目標類別
                if class_id == target_class_id:
                    # 取得後面的所有座標數值，並轉為浮點數
                    coords = [float(x) for x in parts[1:]]
                    
                    # 判斷格式：如果是標準的 4 個座標 (加 ID 共 5 個)
                    if len(coords) == 4:
                        x_center, y_center, width, height = coords
                    # 如果是多邊形 (大於 4 個座標，例如 8 個)
                    elif len(coords) > 4 and len(coords) % 2 == 0:
                        x_center, y_center, width, height = convert_polygon_to_bbox(coords)
                        polygon_converted_count += 1
                    else:
                        print(f"⚠️ 警告: 檔案 {filename} 中發現無法辨識的座標格式，已跳過該行。")
                        continue
                    
                    # 重新組裝為標準的 YOLO 格式字串 (保留 6 位小數確保精度)
                    new_line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    new_lines.append(new_line)
        
        # 寫入檔案 (若 new_lines 為空，則自然生成符合負樣本需求的空白檔)
        with open(output_path, 'w') as f:
            f.writelines(new_lines)
            
        kept_lines_count += len(new_lines)
        processed_count += 1

    print("-" * 40)
    print("🎉 終極格式化完成！報告如下：")
    print(f"  📂 處理檔案數: {processed_count} 個")
    print(f"  🎯 成功保留的標籤: {kept_lines_count} 行 (統一轉換為 ID {new_class_id})")
    print(f"  🔄 多邊形成功降維轉換: {polygon_converted_count} 行")
    print("-" * 40)

if __name__ == "__main__":
    # ================= 設定區域 =================
    INPUT_TXT_DIR = r'D:\side_project\original_dataset\1\train\labels'
    OUTPUT_TXT_DIR = r'D:\side_project\Processed\Processed\labels'
    
    # 設定你要找的原作者 ID，以及你想轉換成的新 ID
    TARGET_ID = 3
    NEW_ID = 0
    # ===========================================
    
    robust_label_formatter(INPUT_TXT_DIR, OUTPUT_TXT_DIR, target_class_id=TARGET_ID, new_class_id=NEW_ID)