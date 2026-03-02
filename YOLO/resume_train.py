from ultralytics import YOLO

# 1. 載入最後一次中斷時存檔的權重
model = YOLO('runs/detect/GCP_L4_Rescue/v11l_thermal_full_run/weights/last.pt') 

# 2. 啟動接續訓練 (不要傳入任何其他超參數！)
results = model.train(resume=True)

print("✅ 已經成功從中斷點無縫接續訓練！")