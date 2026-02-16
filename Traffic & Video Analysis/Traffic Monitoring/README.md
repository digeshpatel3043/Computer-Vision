# 🚦 Traffic Congestion Detection using YOLOv8

## 📌 Project Overview

This project performs real-time traffic analysis using computer vision and deep learning. It detects and tracks vehicles from video footage using YOLOv8 and estimates vehicle speed based on movement between frames.

The system analyzes traffic density and identifies congestion conditions based on vehicle count and speed.

---

## 🎯 Key Features

- Vehicle detection using YOLOv8
- Multi-object tracking with unique IDs
- Speed estimation using frame-to-frame motion
- Traffic congestion detection logic
- Traffic flow classification
- Real-time visualization
- Processed video output

---

## 🧠 Methodology

### 1️⃣ Object Detection & Tracking

YOLOv8 detects and tracks vehicles in video frames.

Vehicle classes used (COCO dataset):

- Car (2)
- Motorcycle (3)
- Bus (5)
- Truck (7)

---

### 2️⃣ Speed Calculation

Speed is estimated using:

- Center point displacement between frames
- Distance calculation
- FPS-based speed estimation
- Moving average smoothing

---

### 3️⃣ Traffic Analysis

Congestion is detected based on:

- Total vehicle count
- Number of slow-moving vehicles
- Consecutive congestion frames

Traffic levels:

- FREE FLOW
- MODERATE
- HEAVY

---

## ⚙️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib

---
