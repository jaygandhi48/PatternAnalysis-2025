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
        
    def mask_to_bbox(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            return None
        _, binary_mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0: 
            return None
        x,y,w,h = cv2.boundingRect(max(contours,key=cv2.contourArea))
        h_img,w_img = mask.shape
        x_center = ( x + w / 2 ) / w_img
        y_center = ( y + h / 2 ) / h_img
        width = w / w_img
        height = h / h_img
        return [0,x_center,y_center,width,height]