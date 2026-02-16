# Harris and FAST Feature Detection (Classical Computer Vision)

## 📌 Project Overview

This project demonstrates classical computer vision techniques for feature detection using:

- Harris Corner Detection (with multi-scale and subpixel refinement)
- FAST Feature Detection (with score-based Non-Maximum Suppression)

The goal is to identify important feature points (corners) in images for tasks such as object recognition, tracking, and image matching.

---

## 🧠 Algorithms Used

### 🔹 Harris Corner Detection

Harris detector identifies corners by analyzing intensity changes in multiple directions.

Features:

- Multi-scale filtering
- Gaussian smoothing
- Corner response calculation
- Thresholding
- Subpixel refinement

---

### 🔹 FAST Feature Detection

FAST (Features from Accelerated Segment Test) detects corners efficiently using pixel intensity comparison around a circular neighborhood.

Features:

- Circular pixel comparison
- Score calculation
- Non-Maximum Suppression (NMS)
- Fast real-time detection

---

## ⚙️ Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- SciPy

---

## 📁 Files

- `ClassicalCVPipeline.ipynb` → Main notebook
- `haarcascade_frontalface_default.xml` → Face detection model (if used)
- `haarcascade_eye.xml` → Eye detection model (if used)
- Sample image for testing

---

## ▶️ How to Run

1. Install dependencies:

