from ultralytics import YOLO
from dataset import ISICLesionDataset
from modules import calculate_iou
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

CONFIG = {
    "img_size": 640,
    "conf_threshold": 0.5,
    "iou_threshold": 0.8
}

MODEL_PATH = "./runs/detect/train/weights/best.pt"
DATASET_DIR = "./YOLOv8_ISIC"


def visualize_predictions(dataset, model_path, num_samples=9):
    model = YOLO(model_path)
    test_images = os.listdir(f"{dataset.images_dir}/test")

    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        img_name = random.choice(test_images)
        img_path = f"{dataset.images_dir}/test/{img_name}"
        label_path = f"{dataset.labels_dir}/test/{os.path.splitext(img_name)[0]}.txt"

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label = list(map(float, f.readline().split()))
                _, x_center, y_center, bw, bh = label
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        results = model.predict(img_path, conf=CONFIG["conf_threshold"], imgsz=CONFIG["img_size"])
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box[:4].int().tolist()
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.subplot(3, 3, i + 1)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(img_name)

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()


