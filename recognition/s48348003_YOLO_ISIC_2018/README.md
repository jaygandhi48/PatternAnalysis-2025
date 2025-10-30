# Skin Lesion Detection using ISIC 2018 dataset based on YOLO architecture

## Problem

According to the Skin Cancer Foundation, more than 9,500 people are diagnosed worldwide with skin cancer every day. This presents the need for accurate and precise procedures to identify and detect lesions in medical imaging. The following report focuses on training and evaluating a YOLO model to detect and classify skin lesions from the ISIC 2018 dataset with the aim of achieving an IoU greater than 0.8. In this report a pretrained YOLOv8 model on the COCO dataset would be finetuned to detect and classify skin lesions

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

### Dataset Statistics

In total, the dataset contains 3,294 images with the following split derived directly from the ISIC challenge:

- **Training:** 2,194 images
- **Validation:** 100 images
- **Testing:** 1,000 images

**Note on split:** The split is unconventional compared to typical 70/15/15 or 80/10/10 splits. However, this follows the official ISIC challenge distribution. Each folder contains exactly the number of images provided by the organizers. The naming convention ensures that each image (`ISIC_xxxxxxx.jpg`) directly corresponds to its segmentation mask (`ISIC_xxxxxxx_segmentation.png`) in the ground truth folders.

The split is approximately 67/3/30 for training, validation, and testing respectively. The following data and the segmentation maps are shown below
![Test Image for ISIC dataset](./readmeImages/testImage.jpg)
![Test Ground truth for ISIC dataset](./readmeImages/testImageTruth.png)

---

## YOLO Architecture

You Only Look Once (YOLO) is a fast object detection algorithm which aims to classify as well as localise objects simultaneously by treating detection as a single regression problem. Unlike traditional algorithms which separate these two tasks, YOLO performs bounding box prediction and object classification in a single pass with extreme speed.

## Working of YOLO

![YOLO Architecture](https://viso.ai/wp-content/uploads/2023/12/YOLOv8-Architecture-Structure-1012x1060.jpg)

YOLO aims to divide original image into N x N grid of equal sizes. Within the grid, each cell is responsible for localisation of an object as well as predicting the class of the object it covers with a confidence score. Since the ISIC data only contains skin lesions, this is a single class problem with the only class being 'lesion'.

Next, YOLO aims to create bounding boxes, which are rectangular shapes that cover an object. If multiple shapes exist in a given image, there could be multiple bounding boxes. The following is the format for the bounding box:

```
(x_center, y_center, width, height, confidence, class)
```

Where:

- `x_center, y_center`: Center coordinates of the bounding box
- `width, height`: Dimensions of the bounding box
- `confidence`: Probability that the box contains an object
- `class`: Class label (in our case, always 'lesion')

Since there can be multiple grid predictions for a given object, not all may be relevent or necessary. Hence the technique of Intersection Over Union (IoU) aims to filter these. It selects images that are above a certain IoU threshold.

The following is the calculation for IoU:

```
IoU = Area of Overlap / Area of Union
```

Or more formally:

```
IoU = (Area of Predicted Box ∩ Area of Ground Truth Box) / (Area of Predicted Box ∪ Area of Ground Truth Box)
```

Further filtering is required as still multiple boxes for a single object may be present. Therefore Non-Maximum Suppression aims to remove boxes of low confidence, removing redundant predictions of overlap predicting the same object.

![YOLO working example](https://images.datacamp.com/image/upload/v1664382700/Object_detection_illustrated_from_image_recognition_and_localization_704ca34bd8.png)

---

## Loss Functions

The loss functions in YOLO consist of three seperate losses which are localisation loss, confidence loss and classification loss.

### Localisation Loss

YOLOv8 uses Complete Intersection over Union (CIoU) loss as default. This measures how close the predicted grid for a given object is to the ground truth box. It differs from regular IoU as it considers:

1. **Overlap** - Normal IoU
2. **Center distance** - How far apart the center points of each boxes lie
3. **Aspect ratio consistency** - Similarity in shape between predicted and ground truth boxes

The CIoU loss is calculated as:

$$L_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b_{gt})}{c^2} + \alpha v$$

Where:

- $\rho^2(b, b_{gt})$: Euclidean distance between center points of predicted and ground truth boxes
- $c$: Diagonal length of the smallest enclosing box
- $\alpha$: Positive trade-off parameter
- $v$: Measures aspect ratio consistency

### Confidence Loss

Confidence loss in each cell predicts the likelihood of it containing an object. The Binary Cross Entropy loss is utilised as shown below using the formula:

$$L_{\text{conf}} = -\left[y_{\text{obj}} \cdot \log(p_{\text{obj}}) + (1 - y_{\text{obj}}) \cdot \log(1 - p_{\text{obj}})\right]$$

Where:

- $y_{\text{obj}}$: Ground truth (1 if object present, 0 otherwise)
- $p_{\text{obj}}$: Predicted confidence score

This loss penalises the model when it predicts high confidence for cells without objects and low confidence for cells with objects.

### Classification Loss

Classification loss measures how accurately the model predicts the class of detected objects. For our single-class problem, this uses Binary Cross Entropy:

$$L_{\text{cls}} = -\left[y_{\text{cls}} \cdot \log(p_{\text{cls}}) + (1 - y_{\text{cls}}) \cdot \log(1 - p_{\text{cls}})\right]$$

Where:

- $y_{\text{cls}}$: Ground truth class label
- $p_{\text{cls}}$: Predicted class probability

Since we only have one class ('lesion'), this loss ensures the model correctly identifies lesions versus background.

The total YOLO loss is a weighted combination of these three components:

$$L_{\text{total}} = \lambda_{\text{loc}} \cdot L_{\text{CIoU}} + \lambda_{\text{conf}} \cdot L_{\text{conf}} + \lambda_{\text{cls}} \cdot L_{\text{cls}}$$

Where $\lambda$ values are weighting factors that balance the contribution of each loss component during training.
