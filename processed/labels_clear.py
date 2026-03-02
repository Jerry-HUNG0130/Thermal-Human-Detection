import os

def filter_and_remap_labels(input_dir, output_dir, target_class_id=3, new_class_id=0):
    """
    批次讀取 txt 檔，清除目標類別以外的標籤，並將目標 ID 重置為單一類別的 0。
    """
    # 1. 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 輸出資料夾已確認: {output_dir}")
    
    # 2. 取得所有 txt 檔案
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    print(f"🔍 找到 {len(txt_files)} 個標籤檔，準備過濾 (目標 ID: {target_class_id} -> 新 ID: {new_class_id})...")

    # 計數器
    processed_count = 0
    kept_lines_count = 0
    empty_files_count = 0 # 紀錄因為過濾而變成空白的負樣本數

    # 3. 開始遍歷處理
    for filename in txt_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 讀取原始內容
        with open(input_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            
            # 確保該行不是完全空白
            if len(parts) > 0:
                class_id = int(parts[0])
                
                # 核心過濾邏輯：只保留我們指定的 target_class_id (3)
                if class_id == target_class_id:
                    # 替換為我們需要的新 ID (0)
                    parts[0] = str(new_class_id)
                    # 重新組合成字串並加入清單
                    new_lines.append(" ".join(parts) + "\n")
        
        # 寫入到目標資料夾
        # 注意：如果 new_lines 是空的，這裡就會寫入一個空白檔案，完美形成負樣本
        with open(output_path, 'w') as f:
            f.writelines(new_lines)
            
        # 統計資訊
        if len(new_lines) == 0 and len(lines) > 0:
            empty_files_count += 1
            
        kept_lines_count += len(new_lines)
        processed_count += 1
        
    # 4. 輸出執行報告
    print("-" * 40)
    print("🎉 標籤過濾與轉換完成！最終報告：")
    print(f"  📂 總處理檔案: {processed_count} 個")
    print(f"  🎯 成功保留的『人』標籤: {kept_lines_count} 行")
    print(f"  🌟 轉換為空白的負樣本檔案: {empty_files_count} 個 (原本可能只有車/狗/腳踏車)")
    print(f"  🔄 所有保留的標籤 ID 已統一修改為: {new_class_id}")
    print("-" * 40)

if __name__ == "__main__":
    # ================= 設定區域 =================
    # 請替換為你實際的路徑
    INPUT_TXT_DIR = r'D:\side_project\original_dataset\4\train\labels'
    OUTPUT_TXT_DIR = r'D:\side_project\Processed\Processed\4\labels'
    
    # 你的 YAML 中 person 是 3，且我們要將其轉為單一類別的 0
    TARGET_ID = 1
    NEW_ID = 0
    # ===========================================
    
    filter_and_remap_labels(INPUT_TXT_DIR, OUTPUT_TXT_DIR, target_class_id=TARGET_ID, new_class_id=NEW_ID)