from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathlib import Path

"""
    Calculate IoU between two boxes [x_center, y_center, width, height].
"""
def calculate_iou(box1, box2):
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def train_model(model_config, dataset_yaml, config):
    """Train YOLOv8 model."""
    model = YOLO(model_config)
    results = model.train(
        data=dataset_yaml,
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        patience=20,
        save=True,
        device=0,              # GPU if available, else CPU
        project='runs/detect',
        name='isic_lesion_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        verbose=True,
        plots=True
    )
    return model