# train.py
from dataset import ISICLesionDataset
from modules import train_model, evaluate_model, plot_metrics

# Training and model configuration parameters
CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640,
    "iou_threshold": 0.8,
    "conf_threshold": 0.45,
    "model_name": "yolov8m.yaml"
}

# Local filesystem paths for ISIC2018 dataset splits
LOCAL_PATHS = {
    "train_images": "./ISIC2018/ISIC2018_Task1-2_Training_Input_x2",
    "train_masks": "./ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2",
    "val_images": "./ISIC2018//ISIC2018_Task1-2_Validation_Input",
    "val_masks": "./ISIC2018/ISIC2018_Task1_Validation_GroundTruth",
    "test_images": "./ISIC2018/ISIC2018_Task1-2_Test_Input",
    "test_masks": "./ISIC2018/ISIC2018_Task1_Test_GroundTruth_2"
}

if __name__ == "__main__":
    # Initialize dataset handler and prepare YOLO-format dataset
    dataset = ISICLesionDataset(base_dir="./YOLOv8_ISIC")
    dataset_yaml = dataset.prepare_dataset(LOCAL_PATHS)
    
    print("\n--- Training model ---")
    model = train_model(CONFIG["model_name"], dataset_yaml, CONFIG)
    
    # Path to best model checkpoint from training
    best_model_path = "./runs/detect/train/weights/best.pt"
    
    print("\n--- Evaluating model ---")
    ious = evaluate_model(dataset, best_model_path, CONFIG)
    
    print("\n--- Plotting metrics ---")
    plot_metrics(ious, CONFIG)