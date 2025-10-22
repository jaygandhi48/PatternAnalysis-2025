# dataset.py
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

'''
Class responsible for handling the dataset. 
It converts segmentation masks to labels and 
creating appropriate folders for processed data.
'''
class ISICLesionDataset:
    def __init__(self, base_dir="./YOLOv8_ISIC"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        #Create subfolders for train, test and validation
        for split in ['train','val','test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        self.stats = {'train':0,'val':0,'test':0}