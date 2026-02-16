# CIFAR-10 Image Classification using CNN (TensorFlow / Keras)

## 📌 Project Overview

This project implements an image classification model using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

The goal is to classify images into 10 categories including:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The model uses deep learning techniques with convolutional layers, batch normalization, pooling, and dropout for improved performance.

---

## 🧠 Model Architecture

The CNN model includes:

- Convolutional layers (Conv2D)
- Batch Normalization
- MaxPooling
- Dropout (regularization)
- Fully connected dense layers
- Softmax output layer

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 📊 Dataset

CIFAR-10 dataset:

- 60,000 color images
- Image size: 32x32 pixels
- 10 classes
- Training images: 50,000
- Test images: 10,000

Dataset loaded directly using:

```python
from tensorflow.keras.datasets import cifar10
