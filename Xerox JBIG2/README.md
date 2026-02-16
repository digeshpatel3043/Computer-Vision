# 📄 Intelligent Image Pattern Analysis & Safe Compression System

## 📌 Project Overview

This project explores advanced computer vision techniques for analyzing structured images (documents, symbols, patterns) and understanding how image compression affects both human perception and machine-based processing.

The system combines:

- Pattern substitution and symbol grouping
- Human-visible vs machine-relevant image differences
- Silent data corruption detection
- Safe compression decision rules
- Template-based recognition evaluation

This project demonstrates how computer vision systems respond to image compression and structural changes.

---

## 🎯 Key Features

✅ Symbol detection using connected components  
✅ Pattern grouping using similarity (MSE-based clustering)  
✅ Image reconstruction from detected symbol prototypes  
✅ Visual highlighting of structural changes  
✅ PSNR & SSIM comparison for compression quality  
✅ Edge detection for machine-relevant analysis  
✅ Silent data corruption detection  
✅ Automated template generation  
✅ Rule-based character recognition  
✅ Safe compression decision system  

---

## 🧠 Methodology

### 1️⃣ Pattern Substitution & Symbol Grouping

- Convert image to binary
- Detect connected components
- Extract symbol regions
- Compare similarity using Mean Squared Error (MSE)
- Group similar symbols
- Reconstruct image using prototype patterns

---

### 2️⃣ Human vs Machine Difference Analysis

JPEG compression is applied at different quality levels:

- Quality levels tested: 90, 70, 50, 30, 10
- Metrics used:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)

Machine perception is analyzed using edge detection.

---

### 3️⃣ Silent Data Corruption Detection

- Compare lossless vs lossy images
- Pixel-wise difference detection
- Structural change highlighting
- Region-based corruption flagging

---

### 4️⃣ Safe Compression Decision System

Compression strategy determined using:

- Image entropy
- Edge density
- Connected component count

Output decisions:

- Lossless compression
- Lossy compression
- No compression

---

### 5️⃣ Compression Impact on Recognition

A simple template-matching recognizer:

- Generates templates automatically
- Segments characters
- Performs template matching
- Evaluates recognition accuracy after compression

---

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Classical Computer Vision Techniques

---



