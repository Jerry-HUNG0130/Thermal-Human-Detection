import os
import shutil

def auto_sort_by_confidence(source_dir, output_base_dir, high_thresh=0.6, low_thresh=0.3):
    """
    讀取檔名中的 YOLO 信心度，將圖片自動分類到三個不同的資料夾。
    """
    # 1. 建立分類後的目標資料夾
    folder_person = os.path.join(output_base_dir, "likely_person")         # 高機率是人
    folder_background = os.path.join(output_base_dir, "likely_background") # 高機率是背景
    folder_unsure = os.path.join(output_base_dir, "unsure")                # 模稜兩可地帶

    for folder in [folder_person, folder_background, folder_unsure]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 2. 取得來源資料夾內所有的圖片
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"在 {source_dir} 中沒有找到圖片，請檢查路徑！")
        return

    print(f"開始分析 {len(image_files)} 張圖片...\n")
    
    # 用來統計各分類數量的計數器
    counts = {"person": 0, "background": 0, "unsure": 0, "error": 0}

    # 3. 逐一解析檔名並分流
    for filename in image_files:
        try:
            # 我們的檔名格式是：原圖名_box編號_類別_信心度.jpg 
            # 例如: test_box0_person_0.85.jpg
            # 透過字串分割取出最後一個 "_" 後面的數字部分
            conf_str = filename.split('_')[-1].replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            conf_score = float(conf_str)

            # 根據信心度門檻進行分流
            if conf_score >= high_thresh:
                target_folder = folder_person
                counts["person"] += 1
            elif conf_score <= low_thresh:
                target_folder = folder_background
                counts["background"] += 1
            else:
                target_folder = folder_unsure
                counts["unsure"] += 1

            # 複製檔案到目標資料夾
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_folder, filename)
            shutil.copy2(src_path, dst_path)

        except ValueError:
            # 防呆機制：如果檔名格式不對（找不到浮點數），就跳過並記錄
            print(f"⚠️ 無法解析檔名中的信心度，已跳過: {filename}")
            counts["error"] += 1

    # 4. 印出整理報告
    print("-" * 30)
    print("🎉 分流任務完成！報告如下：")
    print(f"✅ 高度確信是人 (>= {high_thresh}): {counts['person']} 張 -> 已存入 likely_person")
    print(f"❌ 高度確信是背景 (<= {low_thresh}): {counts['background']} 張 -> 已存入 likely_background")
    print(f"🤔 模稜兩可需人工確認: {counts['unsure']} 張 -> 已存入 unsure")
    if counts["error"] > 0:
        print(f"⚠️ 格式錯誤跳過的檔案數: {counts['error']} 張")
    print("-" * 30)


# ==========================================
# 參數設定區
# ==========================================
if __name__ == "__main__":
    # 你的原始裁切圖片存放的資料夾
    SOURCE_DIR = "./cropped_dataset" 
    
    # 程式自動建立的分類總資料夾
    OUTPUT_BASE_DIR = "./sorted_staging_area" 
    
    # 信心度門檻設定 (可依據你的觀察微調)
    HIGH_THRESHOLD = 0.60
    LOW_THRESHOLD = 0.30

    auto_sort_by_confidence(
        source_dir=SOURCE_DIR, 
        output_base_dir=OUTPUT_BASE_DIR, 
        high_thresh=HIGH_THRESHOLD, 
        low_thresh=LOW_THRESHOLD
    )