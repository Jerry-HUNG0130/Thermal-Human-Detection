import os
import cv2
from torchvision import transforms

# 專案根目錄路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 各資料夾路徑
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def preprocess_frame_for_inference(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(smoothed_img)
    final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return final_img


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')


def get_transform():
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def weight_path(filename):
    return os.path.join(WEIGHTS_DIR, filename)
