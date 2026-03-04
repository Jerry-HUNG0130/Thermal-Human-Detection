import cv2
import os
from ultralytics import YOLO

def preprocess_frame_for_inference(frame):
    """將影像進行去噪與 CLAHE 對比度增強"""
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img

def draw_boxes_by_confidence(image_path, model, output_path, model_name):
    """讀取影像、進行推論，並依信心度將框繪製於原始圖片上"""
    # 讀取原始圖片
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"無法讀取圖片: {image_path}")
        return

    # 1. 產生預處理影像供 AI 推論使用
    inference_img = preprocess_frame_for_inference(original_img)

    # 2. 執行推論 (使用預處理過的高對比影像)
    results = model(inference_img, verbose=False)[0]
    
    # 3. 準備畫框 (畫在未經處理的 original_img 上，確保使用者體驗)
    img_draw = original_img.copy()

    for box in results.boxes:
        # 因為預處理沒有改變影像長寬，所以座標可以直接對應回原圖
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if cls_id != 0: # 假設 0 為人
            continue

        if conf >= 0.85:
            color = (0, 0, 255)      # 紅色
        elif conf >= 0.70:
            color = (0, 255, 255)    # 黃色
        else:
            color = (0, 255, 0)      # 綠色

        # 繪製矩形框與標籤
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"{model_name}: {conf:.2f}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 儲存結果
    cv2.imwrite(output_path, img_draw)
    print(f"[{model_name}] 處理完成，已儲存至: {output_path}")

def main():
    image_path = "test.jpg"
    output_dir = "D:/side_project"
    os.makedirs(output_dir, exist_ok=True)

    print("載入模型中...")
    model_pretrained = YOLO("yolo11n.pt") 
    model_custom = YOLO("best.pt") 

    draw_boxes_by_confidence(image_path, model_pretrained, os.path.join(output_dir, "result_pretrained.jpg"), "Pretrained")
    draw_boxes_by_confidence(image_path, model_custom, os.path.join(output_dir, "result_custom.jpg"), "CustomYOLO")

if __name__ == "__main__":
    main()