# train.py
from dataset import ISICLesionDataset
from modules import train_model, evaluate_model, plot_metrics

CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640,
    "iou_threshold": 0.8,
    "conf_threshold": 0.45,
    "model_name": "yolov8m.yaml"
}

LOCAL_PATHS = {
    "train_images": "./ISIC2018/train/images",
    "train_masks": "./ISIC2018/train/masks",
    "val_images": "./ISIC2018/val/images",
    "val_masks": "./ISIC2018/val/masks",
    "test_images": "./ISIC2018/test/images",
    "test_masks": "./ISIC2018/test/masks"
}

if __name__ == "__main__":
    dataset = ISICLesionDataset(base_dir="./YOLOv8_ISIC")
    dataset_yaml = dataset.prepare_dataset(LOCAL_PATHS)

    print("\n--- Training model ---")
    model = train_model(CONFIG["model_name"], dataset_yaml, CONFIG)

    best_model_path = "./runs/detect/train/weights/best.pt"
    print("\n--- Evaluating model ---")
    ious = evaluate_model(dataset, best_model_path, CONFIG)

    print("\n--- Plotting metrics ---")
    plot_metrics(ious, CONFIG)
