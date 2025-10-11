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

## 📈 Evaluation Metrics

| Metric | Description | Result |
|---------|--------------|--------|
| **Precision** | Measures how many of the detected potholes are actually potholes | **0.502** |
| **Recall** | Measures how many actual potholes were correctly detected | **0.267** |
| **mAP@50** | Mean Average Precision at IoU threshold 0.5 | **0.312** |
| **mAP@50-95** | Mean Average Precision at IoU thresholds 0.5 to 0.95 | **0.179** |

### 🔍 Class-wise Performance

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|------------|---------|---------|------------|
| **objects** | 0.493 | 0.333 | 0.351 | 0.274 |
| **pothole** | 0.510 | 0.202 | 0.273 | 0.084 |

---

## 🧠 Results Summary

- The model successfully detects **potholes and objects** in most clear images.
- Detection performance is **moderate** due to variations in lighting, texture, and pothole shapes.
- **Precision (50%)** indicates the model detects potholes correctly half the time.
- **Recall (26%)** shows there is still room for improvement in identifying all potholes.

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pothole-detection.git
   cd pothole-detection
