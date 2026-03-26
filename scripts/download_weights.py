import os
import sys
import torch
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from preprocess import weight_path

print("⏳ 開始下載 YOLO11l 官方原始權重...")
yolo_model = YOLO('yolo11l.pt')
os.rename('yolo11l.pt', weight_path('yolo11l.pt'))
print("✅ YOLO11l 下載完成！檔案已儲存至 weights/yolo11l.pt\n")

print("⏳ 開始下載 EfficientNet-B2 官方原始權重...")
eff_model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
torch.save(eff_model.state_dict(), weight_path('efficientnet_b2_original.pth'))
print("✅ EfficientNet-B2 下載完成！檔案已儲存至 weights/efficientnet_b2_original.pth\n")

print("🎉 所有官方原始權重已準備就緒！")
