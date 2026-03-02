from ultralytics import YOLO

# 載入預訓練模型
model = YOLO('yolov11l.pt') 

results = model.train(
    data='thermal_data.yaml',
    
    # --- 硬體與效能極限榨取區 ---
    device=0,
    batch=32,          # 【修正】YOLO11l 模型較大，L4 (24GB VRAM) 開 batch=64 極高機率 OOM，建議先用 32 穩跑。
    imgsz=640,         
    workers=4,         # 【修正】VM 只有 8 個 vCPU，若 workers=8 會導致 CPU 滿載無暇處理系統背景任務而卡死，保留一半給主線程與 OS。
    cache='disk',      # 【關鍵】解決 5 萬張圖片 OOM 問題，利用 GCP SSD 加速 I/O。
    
    # --- 訓練週期與早停策略 ---
    epochs=300,        
    patience=50,       # 100 太長了，通常 50 輪 loss 沒降就代表已經收斂，能幫你省下 GCP 的昂貴運算費。
    save_period=20,    # 每 20 輪強制備份一次權重，避免 VM 意外斷線 (特別是使用 Spot VM 時)。
    
    # --- 學習率與優化器 (動態調整) ---
    optimizer='AdamW', # 針對大型自定義資料集，AdamW 收斂更穩。
    lr0=0.001,         # AdamW 的初始學習率建議用 1e-3 (SGD 才是 1e-2)。
    lrf=0.01,          # 最終學習率降至 lr0 的 1%。
    cos_lr=True,       # 【新增】開啟餘弦退火，讓學習率平滑下降，在訓練後期更容易滑入 Loss 的全域最佳解。
    warmup_epochs=3.0, # 【新增】暖機機制，前 3 輪學習率會從極小值慢慢爬升，避免一開始梯度爆炸破壞預訓練權重。
    warmup_momentum=0.8,
    
    # --- 遷移學習策略 (不凍結) ---
    freeze=None,       # 【關鍵】既然你有高達 5 萬張圖，完全足以從頭微調整個網路！原 COCO 權重是 RGB 學習來的，熱顯像的底層邊緣特徵完全不同，凍結反而會限制模型適應熱顯像灰階特徵的能力。
    
    # --- 驗證與資料結構 ---
    val=True,          # 每輪訓練後進行驗證，監控 mAP。
    rect=False,        # 訓練時維持 False (開啟 Mosaic 時必須為 False)。
    
    # --- 熱顯像專屬增強 (精準控制) ---
    mosaic=1.0,        
    close_mosaic=15,   # 【新增】非常重要！在最後 15 輪關閉 Mosaic，讓模型看「真實完整」的圖片進行最後微調，通常能提升 1~2% 的 mAP。
    mixup=0.0,         # 【關鍵】嚴格保持 0！Mixup 會將兩張圖片半透明疊加，在熱顯像中這會產生「幽靈熱源」，嚴重干擾物理邏輯。
    copy_paste=0.0,    # 保持 0。除非你有做遮罩 (Mask) 標記，否則純 Bounding Box 複製貼上在熱顯像容易產生突兀的背景邊緣。
    hsv_h=0.0,         
    hsv_s=0.0,         
    hsv_v=0.1,         # 允許 10% 亮度干擾，模擬不同溫度下的感測器曝光差異。
    degrees=10.0,      # 【新增】空拍機受風力影響會有傾角，允許正負 10 度的旋轉增強。
    translate=0.1,     # 【新增】允許 10% 平移，增強物件不在畫面正中央時的辨識力。
    
    project='GCP_L4_Rescue',
    name='v11l_thermal_full_run'
)