# Skin Lesion Detection using ISIC 2018 dataset based on YOLO architecture

## Problem

According to the Skin Cancer Foundation, more than 9,500 people are diagnosed worldwide with skin cancer every day. This presents the need for accurate and precise procedures to identify and detect lesions in medical imaging.

The following report focuses on training and evaluating a YOLO model to detect and classify skin lesions from the ISIC 2018 dataset with the aim of achieving an IoU greater than 0.8. In this report a pretrained YOLOv8 model on the COCO dataset would be finetuned to detect and classify skin lesions

## Dataset

The 2018 ISIC dataset can be downloaded from the following link: [ISIC 2018](https://challenge.isic-archive.com/data/#2018).  
The folder structure is showcased below:

```bash
ISIC_2018/
├── ISIC2018_Task1-2_Training_Input_x2/
│   ├── ISIC_000001.jpg
│   ├── ISIC_000002.jpg
│   └── ...
│
├── ISIC2018_Task1_Training_GroundTruth_x2/
│   ├── ISIC_000001_segmentation.png
│   ├── ISIC_000002_segmentation.png
│   └── ...
│
├── ISIC2018_Task1-2_Validation_Input/
│   ├── ISIC_002195.jpg
│   ├── ISIC_002196.jpg
│   └── ...
│
├── ISIC2018_Task1_Validation_GroundTruth/
│   ├── ISIC_002195_segmentation.png
│   ├── ISIC_002196_segmentation.png
│   └── ...
│
├── ISIC2018_Task1-2_Test_Input/
│   ├── ISIC_002295.jpg
│   ├── ISIC_002296.jpg
│   └── ...
│
└── ISIC2018_Task1_Test_GroundTruth_2/
    ├── ISIC_002295_segmentation.png
    ├── ISIC_002296_segmentation.png
    └── ...
```
