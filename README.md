# 🕳️ Pothole Detection using YOLO and Roboflow

## 📘 Project Overview
This project aims to develop an automated **pothole detection system** using deep learning-based **object detection models (YOLO)**.  
The goal is to detect and locate potholes on road surfaces from real-world images, which can help in road maintenance and safety management.

The dataset was collected, labeled, and augmented using **Roboflow**, and the model was trained using **YOLOv11 Object Detection**.

---

## 📂 Dataset Details

- **Source:** Custom dataset labeled using [Roboflow](https://roboflow.com/)  
- **Classes:**  
  - `pothole`  
  - `objects` (other road elements)

### 📊 Dataset Split
| Dataset Type | Number of Images |
|---------------|------------------|
| Training Set  | 1083 |
| Validation Set| 146 |
| Test Set      | 66 |
| **Total**     | **1295** |

---

## ⚙️ Model Details

- **Model Type:** YOLOv11 (Accurate)  
- **Training Platform:** Roboflow Train  
- **Checkpoint:** Pretrained on MS COCO (Best 47.0% mAP)  
- **Epochs Trained:** 100  
- **Image Resolution:** 640x640  
- **Augmentations Applied:**
  - Blur
  - Rotation
  - Brightness & Contrast Variation
  - Noise Addition
  - Horizontal & Vertical Flip

---

