# dataset.py
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

class ISICLesionDataset:
    """
    Handles ISIC dataset preprocessing for YOLOv8 training.
    Converts segmentation masks to bounding box labels and organizes data into train/val/test splits.
    """
    def __init__(self, base_dir="./YOLOv8_ISIC"):
        """
        Initialize dataset handler and create directory structure.
        
        Args:
            base_dir: Root directory for processed dataset
        """
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        
        # Create subdirectories for train/val/test splits
        for split in ['train','val','test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self.stats = {'train':0,'val':0,'test':0}
        
    def mask_to_bbox(self, mask_path):
        """
        Convert binary segmentation mask to YOLO format bounding box.
        
        Args:
            mask_path: Path to segmentation mask image
        
        Returns:
            list: [class_id, x_center, y_center, width, height] in normalized coordinates (0-1)
            None: If mask is invalid or contains no contours
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            return None
        
        # Threshold mask to binary
        _, binary_mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        
        # Find contours and get bounding box of largest contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0: 
            return None
        
        x,y,w,h = cv2.boundingRect(max(contours,key=cv2.contourArea))
        h_img,w_img = mask.shape
        
        # Normalize coordinates to 0-1 range (YOLO format)
        x_center = ( x + w / 2 ) / w_img
        y_center = ( y + h / 2 ) / h_img
        width = w / w_img
        height = h / h_img
        
        return [0,x_center,y_center,width,height]  # Class 0 for single-class lesion detection
    
    def prepare_dataset(self, local_paths):
        """
        Process raw ISIC dataset: convert masks to YOLO labels and copy images.
        Skips splits that already have labels to avoid reprocessing.
        
        Args:
            local_paths: Dictionary containing paths to raw images and masks for each split
        
        Returns:
            str: Path to generated dataset.yaml file
        """
        for split in ['train','val','test']:
            split_img_dir = self.images_dir / split
            split_label_dir = self.labels_dir / split
            
            # Skip if labels already exist
            existing_labels = list(split_label_dir.glob("*.txt"))
            if len(existing_labels) > 0:
                print(f"Skipping {split} — labels already exist ({len(existing_labels)} files).")
                continue
            
            print(f"Creating YOLO labels for {split}...")
            img_dir = Path(local_paths[f"{split}_images"])
            mask_dir = Path(local_paths[f"{split}_masks"])
            processed = 0

            for img_path in tqdm(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')), desc=f"{split}"):
                img_name = img_path.stem
                mask_path = mask_dir / f"{img_name}_segmentation.png"
                
                if not mask_path.exists():
                    continue
                
                bbox = self.mask_to_bbox(mask_path)
                if bbox is None:
                    continue
                
                # Copy image and save corresponding label
                shutil.copy(img_path, split_img_dir / img_path.name)
                with open(split_label_dir / f"{img_name}.txt", 'w') as f:
                    f.write(' '.join(map(str, bbox)))
                processed += 1
            
            self.stats[split] = processed
            print(f"Processed {processed} {split} images.")

        return self.create_yaml()

    def create_yaml(self):
        """
        Generate YOLO dataset configuration file (dataset.yaml).
        Specifies dataset paths, number of classes, and class names.
        
        Returns:
            str: Path to created dataset.yaml file
        """
        yaml_content = f"""path: {self.base_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['lesion']
"""
        yaml_path = self.base_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Dataset YAML saved to: {yaml_path}")
        return str(yaml_path)