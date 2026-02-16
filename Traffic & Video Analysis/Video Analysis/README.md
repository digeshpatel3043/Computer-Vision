# ⚽ Sports Player Tracking & Movement Analysis using YOLOv8

## 📌 Project Overview

This project performs sports video analytics using computer vision and deep learning. It detects and tracks players in a sports video using the YOLOv8 object detection model and visualizes player movement trajectories.

The system identifies players (person class), assigns unique tracking IDs, and draws movement paths to analyze player behavior and positioning during gameplay.

---

## 🎯 Key Features

- Player detection using YOLOv8
- Multi-object tracking with unique IDs
- Player movement path visualization
- Real-time player counting
- Frame-by-frame sports analytics visualization
- Works with different sports videos

---

## 🧠 Methodology

### 1️⃣ Object Detection

YOLOv8 detects objects in each frame.

Only the **person class (Class ID = 0)** is selected to track players.

---

### 2️⃣ Player Tracking

- Each detected player receives a unique tracking ID.
- Tracking persists across frames.
- Player positions are stored for movement analysis.

---

### 3️⃣ Movement Path Visualization

- Player center positions are recorded.
- Movement trails are drawn using lines.
- Helps visualize player motion during gameplay.

---

## ⚙️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Jupyter Notebook

---

