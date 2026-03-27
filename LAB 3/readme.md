Here’s a clean, professional **README.md** you can directly copy into your GitHub repo 👇

---

# 🧠 CNN Experiment: Impact of Padding & Stride (MNIST)

This project explores how different **Convolutional Neural Network (CNN)** configurations—specifically **padding** and **stride**—affect model performance, feature maps, and computational efficiency using the **MNIST dataset**.

---

## 📌 Project Objective

The goal of this project is to:

* Understand how **padding (`valid` vs `same`)** affects feature map size
* Analyze how **stride (1 vs 2)** impacts:

  * Spatial dimensions
  * Training time
  * Model accuracy
* Visualize **feature maps** from convolution layers
* Compare configurations using real experimental results

---

## 🏗️ Project Structure

```
📁 cnn-padding-stride-experiment
│── app.py / main.py   # Main script to run experiments
│── README.md          # Project documentation
```

---

## ⚙️ Technologies Used

* Python 🐍
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib

---

## 📊 Dataset

* **MNIST Dataset**

  * 60,000 training images
  * 10,000 testing images
  * Handwritten digits (0–9)
  * Image size: 28×28 (grayscale)

---

## 🧪 Experiment Configurations

The model tests **4 configurations**:

| Padding | Stride | Description               |
| ------- | ------ | ------------------------- |
| valid   | 1      | No padding, fine detail   |
| same    | 1      | Preserves size            |
| valid   | 2      | Downsampling              |
| same    | 2      | Downsampling with padding |

---

## 🧠 Model Architecture

```
Input (28x28x1)
   ↓
Conv2D (32 filters, 3x3)
   ↓
MaxPooling (2x2)
   ↓
Flatten
   ↓
Dense (10 classes - Softmax)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

### 2. Run the script

```bash
python app.py
```

---

## 📈 Output

### 1. Console Output

* Summary table showing:

  * Output dimensions
  * Number of parameters
  * Training time
  * Accuracy

### 2. Visualization Dashboard

* Original image
* Feature maps from convolution filters
* Comparison across all configurations

---

## 🔍 Key Learnings

* **Padding = SAME**

  * Preserves spatial dimensions
  * Better for deeper networks

* **Padding = VALID**

  * Reduces feature size
  * Can lose edge information

* **Stride = 2**

  * Faster computation
  * Lower resolution feature maps

* **Stride = 1**

  * More detailed feature extraction
  * Higher accuracy (generally)

---

## 📸 Feature Map Visualization

The project visually shows how CNN filters respond to input images under different configurations.

---

## 📁 Code Reference

Main experiment logic is implemented in:
👉 

---

## 💡 Future Improvements

* Add more layers (Deep CNN)
* Include Dropout & BatchNorm
* Test on CIFAR-10 dataset
* Add confusion matrix visualization
* Convert into Streamlit web app

---

## 👨‍💻 Author

**Digesh Patel**
PGDM - AI & Data Science

---

