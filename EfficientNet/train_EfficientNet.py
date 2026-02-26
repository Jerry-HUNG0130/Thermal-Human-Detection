import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# 1. 自定義前處理：防形變補邊 (Letterboxing)
# ==========================================
class SquarePad:
    """
    自定義的 Transform：將 YOLO 裁切出來長寬不一的圖片，
    等比例補上黑邊變成正方形，完美保留熱顯像的人體比例特徵。
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        # 計算左右、上下需要補多少邊界
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        
        # padding 順序為 (左, 上, 右, 下)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

# ==========================================
# 2. 環境設定與資料前處理
# ==========================================
# 自動偵測並使用 L4 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"目前使用的設備: {device}")

# 將 SquarePad 加入到資料擴增與前處理流程中
data_transforms = {
    'train': transforms.Compose([
        SquarePad(),                      # 第一步：先不失真地補成正方形
        transforms.Resize((288, 288)),    # 第二步：縮放到 EfficientNet-B2 標準大小
        transforms.RandomHorizontalFlip(),# 第三步：隨機水平翻轉 (資料擴增)
        transforms.ToTensor(),            # 第四步：轉成 Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 第五步：常態化
    ]),
    'val': transforms.Compose([
        SquarePad(),                      # 驗證集也要做一樣的補邊處理
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. 載入資料集
# ==========================================
# 假設你的資料夾結構為：
# dataset/
#  ├── train/
#  │   ├── person/
#  │   └── background/
#  └── val/
#      ├── person/
#      └── background/

# num_workers=4 在 GCP 32G RAM 環境下是合理的設定，可加速資料載入
image_datasets = {x: datasets.ImageFolder(f'dataset/{x}', data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
class_names = image_datasets['train'].classes 
print(f"辨識類別: {class_names}")

# ==========================================
# 4. 載入並修改 EfficientNet-B2 模型
# ==========================================
# 載入預訓練權重(遷移學習)
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

# 將最後的全連接層(FC)替換為我們的類別數量 (例如：person 與 background 共 2 類)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names)) 
model = model.to(device)

# ==========================================
# 5. 設定損失函數與優化器
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 6. 訓練迴圈
# ==========================================
num_epochs = 10 # 面試作品通常建議先跑 10~20 個 Epoch 觀察收斂狀況
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 訓練模式
        else:
            model.eval()   # 驗證模式

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # 梯度清零

            # 前向傳播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 反向傳播與參數更新 (僅限訓練階段)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# ==========================================
# 7. 儲存訓練好的模型權重
# ==========================================
torch.save(model.state_dict(), 'efficientnet_b2_thermal.pth')
print("\n🎉 模型訓練完成！權重已儲存為 efficientnet_b2_thermal.pth")