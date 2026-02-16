# Classical Face Detection Pipeline (OpenCV)

## 📌 Project Overview

This project demonstrates a classical computer vision pipeline for face detection using OpenCV techniques. The workflow follows multiple structured steps including image acquisition, preprocessing, face detection, feature extraction, matching, classification, and visualization.

The pipeline shows how traditional computer vision methods work before modern deep learning approaches.

---

## 🧠 Pipeline Steps

### ✅ Step 1 — Image Acquisition
- Capture image from webcam or load from file
- Convert to grayscale for processing

### ✅ Step 2 — Preprocessing
- Histogram Equalization
- Noise reduction
- Image normalization

### ✅ Step 3 — Face Detection
- Haar Cascade Classifier
- Detect face regions
- Draw bounding boxes

### ✅ Step 4 — Feature Extraction
- ORB feature detection
- Extract keypoints and descriptors

### ✅ Step 5 — Feature Matching
- Brute Force matcher
- Match descriptors between images

### ✅ Step 6 — Classification
- Contour analysis
- Simple rule-based classification

### ✅ Step 7 — Visualization
- Draw detected faces
- Display results using OpenCV

---

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## 📁 Files

- `ClassicalCVPipeline.ipynb` → Main implementation notebook
- `haarcascade_frontalface_default.xml` → Face detection model
- Sample images

---

## ▶️ Installation

Install required libraries:

