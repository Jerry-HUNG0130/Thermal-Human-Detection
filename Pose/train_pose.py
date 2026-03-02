from ultralytics import YOLO
import torch

def main():
    # 1. 檢查 GPU 狀態
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # 2. 載入模型
    # 使用yolo11m-pose.pt權重
    model = YOLO('yolo11m-pose.pt') 

    # 3. 開始訓練
    results = model.train(
        data='C:/side_project/data.yaml', 
        epochs=100,         
        imgsz=640,          
        batch=-1,            
        device=0,
        freeze=10, #Backbone (骨幹網路)：大約第0-9層、Neck (頸部網路)：大約第10-21層、Head (預測頭)：第 22 層之後
        cos_lr=True,         
        warmup_epochs=3.0,          
        lr0=0.0001,          
        lrf=0.01,          
        patience=50,        
        workers=4,          
        project='C:/side_project/my_results',
        name='my_experiment',
        exist_ok=True,
        cache=True,
        pose=6.0,
    )

    # 4. 驗證模型 (可選)
    metrics = model.val()
    print(metrics.box.map)  

if __name__ == '__main__':
    main()