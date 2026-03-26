import torch
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from ultralytics import YOLO

print("⏳ 開始下載 YOLO11l 官方原始權重...")
# 當 YOLO 發現當前目錄沒有 yolo11l.pt 時，就會自動從 GitHub 官方 Release 抓取最新版
yolo_model = YOLO('yolo11l.pt') 
print("✅ YOLO11l 下載完成！檔案已儲存為 yolo11l.pt\n")

print("⏳ 開始下載 EfficientNet-B2 官方原始權重...")
# 透過 weights=EfficientNet_B2_Weights.DEFAULT 指令，載入官方在 ImageNet 上訓練好的最佳權重
eff_model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

# 將權重獨立抽出來，並存成你指定的檔名 (這裡加上 _original 以便跟你自己訓練的做區分)
torch.save(eff_model.state_dict(), 'efficientnet_b2_original.pth')
print("✅ EfficientNet-B2 下載完成！檔案已儲存為 efficientnet_b2_original.pth\n")

print("🎉 所有官方原始權重已準備就緒！")