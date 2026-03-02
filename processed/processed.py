import cv2
import os
import numpy as np

def preprocess_one_image(image_path):
    """
    讀取並依據特定流程預處理單張熱顯像圖片 (專為 YOLO 與小物件搜救優化)。
    已加入雙邊濾波器 (Bilateral Filter) 壓制背景雜訊。
    """
    # 1. 讀取圖片 (強制以彩色模式讀取，確保後續 shape 判斷一致)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"❌ 無法讀取圖片: {image_path}")
        return None

    # 2. 轉為單通道灰階 (消除不同熱顯像儀的色偏干擾)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 雙邊濾波保邊降噪(Bilateral Filter)
    # d=5: 鄰域直徑，設定較小以確保運算速度。
    # sigmaColor=50, sigmaSpace=50: 抹平相近溫度的平坦區域(如路面、天空)，但保留高溫人體的銳利邊緣。
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. 對比度增強 (CLAHE) - 💡 溫和降壓版
    # clipLimit=1.5 (調降放大倍率，守住不讓背景雜訊爆發的底線)
    # tileGridSize=(16, 16) (放大網格，讓計算更平緩、趨近全局)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)

    # 5. 擴展為 3 通道灰階 (為了後續無縫接軌 EfficientNet-B2)
    # 這會讓圖片形狀從 (H, W) 變成 (H, W, 3)，且 R=G=B
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    # 直接回傳 uint8 格式的圖片，無需轉 Float32
    return final_img

def batch_process_folder(input_folder, output_folder, file_exts=['.jpg', '.jpeg', '.png']):
    """
    批次處理整個資料夾的圖片並存檔。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已建立輸出資料夾: {output_folder}")

    files = os.listdir(input_folder)
    count = 0
    
    for file_name in files:
        if not any(file_name.lower().endswith(ext) for ext in file_exts):
            continue
            
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        # 呼叫處理函式
        processed_img = preprocess_one_image(input_path)
        
        if processed_img is not None:
            # 已經是標準的 3 通道 uint8 圖片，直接無損存檔！
            cv2.imwrite(output_path, processed_img)
            count += 1
            if count % 50 == 0:  # 稍微拉長印出頻率，避免洗畫面
                print(f"⏳ 已處理 {count} 張圖片...")

    print("-" * 30)
    print(f"🎉 處理完成！共處理 {count} 張圖片。")
    print(f"📂 存放於: {output_folder}")
    print("-" * 30)

# --- 使用範例 ---
if __name__ == "__main__":
    # 設定你的資料夾路徑 (請確認路徑正確)
    source_dir = r'D:\side_project\original_dataset\4\train\images'  # 使用 r'' 防止 Windows 路徑反斜線跳脫
    target_dir = r'D:\side_project\Processed\Processed\4\images' 

    batch_process_folder(source_dir, target_dir)