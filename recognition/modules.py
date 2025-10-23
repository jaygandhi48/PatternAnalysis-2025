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
        device="cpu",              # GPU if available, else CPU
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

def evaluate_model(dataset, model_path, config):
    """Evaluate model on test set and compute metrics."""
    model = YOLO(model_path)
    test_images_dir = Path(dataset.images_dir) / 'test'
    test_labels_dir = Path(dataset.labels_dir) / 'test'

    results_data = []
    iou_scores = []
    tp, fp, fn = 0, 0, 0

    image_files = sorted(list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png')))

    for img_path in image_files:
        img_name = img_path.stem
        label_path = test_labels_dir / f"{img_name}.txt"

        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            gt_line = f.readline().strip().split()
            gt_box = [float(x) for x in gt_line[1:]]

        results = model.predict(str(img_path),
                                conf=config['conf_threshold'],
                                iou=config['iou_threshold'],
                                verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            pred_box = results[0].boxes.xywhn[0].cpu().numpy()
            conf = float(results[0].boxes.conf[0].cpu().item())
            iou = calculate_iou(pred_box, gt_box)
            iou_scores.append(iou)

            results_data.append({
                'image': img_name,
                'iou': iou,
                'confidence': conf,
                'detected': True,
                'meets_threshold': iou >= config['iou_threshold']
            })
            if iou >= config['iou_threshold']:
                tp += 1
            else:
                fp += 1
        else:
            results_data.append({
                'image': img_name,
                'iou': 0.0,
                'confidence': 0.0,
                'detected': False,
                'meets_threshold': False
            })
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    df = pd.DataFrame(results_data)
    df.to_csv('detailed_results.csv', index=False)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'results_df': df,
        'iou_scores': iou_scores
    }
    
def plot_metrics(eval_results, config):
    """Plot IoU distribution."""
    df = eval_results['results_df']
    iou_scores = eval_results['iou_scores']

    plt.figure(figsize=(8, 6))
    plt.hist(iou_scores, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(config['iou_threshold'], color='red', linestyle='--', label=f"IoU Threshold = {config['iou_threshold']}")
    plt.title("IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iou_distribution.png")
    plt.show()