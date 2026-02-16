# Dog vs Cat Classification using Transfer Learning (VGG16)

## 📌 Project Overview

This project implements an image classification system to distinguish between Dogs and Cats using Transfer Learning with the pre-trained VGG16 model.

Instead of training a deep neural network from scratch, a pre-trained convolutional base (VGG16) is used to extract features, improving accuracy and reducing training time.

---

## 🧠 Key Concepts

- Transfer Learning
- Deep Convolutional Neural Networks
- Fine Tuning
- Image Data Augmentation
- Binary Image Classification

---

## ⚙️ Model Architecture

The project uses:

- Pre-trained VGG16 convolutional base
- Custom classification layers
- Dense layers
- Dropout for regularization
- Binary classification output

---

## 📊 Dataset

Dog vs Cat image dataset:

- Images organized into:
train/
dog/
cat/

validation/
dog/
cat/



---

## 🔄 Training Strategy

### Phase 1 — Feature Extraction

- Freeze VGG16 base layers
- Train only top classifier layers

### Phase 2 — Fine Tuning

- Unfreeze selected deeper layers
- Continue training with lower learning rate

---

## 🧪 Data Augmentation

Applied to training dataset:

- Rescaling
- Rotation
- Width & height shift
- Zoom
- Horizontal flip

Validation dataset:

- Rescaling only

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- VGG16 (Transfer Learning)
- NumPy
- Matplotlib

---

## ▶️ Installation

Install required libraries:


2. Update dataset paths if required.

3. Run all cells.

---

## 📈 Results

The model outputs:

- Training & validation accuracy
- Loss graphs
- Prediction visualization
- Correct vs incorrect classifications

---

## 🚀 Applications

- Image classification
- Pet recognition systems
- Transfer learning experiments
- Deep learning practice

---

## 👨‍💻 Author

Digesh Patel  
Deep Learning / Computer Vision Project

