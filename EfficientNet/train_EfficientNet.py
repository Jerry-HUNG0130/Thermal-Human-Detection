import os
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # 【新增】匯入 tqdm 進度條套件

# ==========================================
# 🔧 企業級超參數設定區
# ==========================================
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # 早停機制：連續 5 輪沒進步就停止

# 【新增】分層學習率
LR_BACKBONE = 1e-4  # 預訓練底層用小學習率
LR_HEAD = 1e-3      # 全新分類頭用大學習率

DATA_DIR = '/home/student/side_project/EfficientNet/dataset'
MODEL_SAVE_PATH = 'best_efficientnet_b2_thermal.pth'
CSV_LOG_PATH = 'efficientnet_training_log.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用設備: {device}")

# ==========================================
# 1. 前處理與熱顯像專屬資料增強 (嚴控泛化)
# ==========================================
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = int((max_wh - w) / 2), int((max_wh - h) / 2)
        return transforms.functional.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), fill=0)

data_transforms = {
    'train': transforms.Compose([
        SquarePad(),                      
        transforms.Resize((288, 288)),    
        transforms.RandomHorizontalFlip(p=0.5), 
        # 使用 Affine 處理無人機傾角與位移，不破壞熱源本體
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 微調熱感應曝光
        transforms.ToTensor(),            
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'val': transforms.Compose([
        SquarePad(),                      
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=NUM_WORKERS) for x in ['train', 'val']}

class_names = image_datasets['train'].classes 

# ==========================================
# 2. 模型建構與分層參數設定
# ==========================================
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# 將 Backbone 與 Classifier 分開給予不同的學習率
backbone_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
classifier_params = model.classifier.parameters()

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': classifier_params, 'lr': LR_HEAD}
], weight_decay=WEIGHT_DECAY)

# Warmup + 餘弦退火機制
warmup_epochs = 3
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - warmup_epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# ==========================================
# 3. CSV 記錄器準備
# ==========================================
with open(CSV_LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'LR_Backbone', 'LR_Head'])

# ==========================================
# 4. 訓練迴圈 (含 Early Stopping)
# ==========================================
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
epochs_no_improve = 0  # 早停計數器

for epoch in range(NUM_EPOCHS):
    current_lr_backbone = optimizer.param_groups[0]['lr']
    current_lr_head = optimizer.param_groups[1]['lr']
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} | LR(Backbone): {current_lr_backbone:.6f} | LR(Head): {current_lr_head:.6f}')
    print('-' * 30)

    epoch_metrics = {'train': {'loss': 0, 'acc': 0}, 'val': {'loss': 0, 'acc': 0}}

    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()   
        running_loss, running_corrects = 0.0, 0

        # 【修改】使用 tqdm 包裝 dataloader，並設定 leave=False 讓畫面保持整潔
        batch_iterator = tqdm(dataloaders[phase], desc=f"[{phase.capitalize()}]", leave=False)

        for inputs, labels in batch_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 【新增】動態更新進度條後方的即時 Loss 數值
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        if phase == 'train':
            scheduler.step() # 每過一個 Epoch 更新一次學習率

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        epoch_metrics[phase]['loss'] = epoch_loss
        epoch_metrics[phase]['acc'] = epoch_acc.item()
        print(f'{phase.capitalize():<5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        # ⭐️ Best Checkpoint & Early Stopping 邏輯
        if phase == 'val':
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # 歸零計數器
                print(f'🏆 儲存最好模型！最高驗證準確率更新為: {best_acc:.4f}')
            else:
                epochs_no_improve += 1
                print(f'⚠️ 驗證準確率未提升，早停計數: {epochs_no_improve}/{PATIENCE}')

    # 寫入 CSV 監控檔
    with open(CSV_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, epoch_metrics['train']['loss'], epoch_metrics['train']['acc'], 
                         epoch_metrics['val']['loss'], epoch_metrics['val']['acc'], 
                         current_lr_backbone, current_lr_head])

    # 觸發早停
    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 觸發早停機制！模型已在第 {epoch+1} 輪提早結束訓練。")
        break

# ==========================================
# 5. 結束與儲存
# ==========================================
print('-' * 30)
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 最強權重已儲存: {MODEL_SAVE_PATH} (最高 Acc: {best_acc:.4f})")
print(f"📊 訓練紀錄已匯出至: {CSV_LOG_PATH}")