🎯 Objective
The objective of this project is to implement a real-time face and eye detection system using classical computer vision techniques.

This project uses:

OpenCV library
Haar Cascade Classifier
Webcam input
Image processing techniques
The system detects faces and eyes and updates the result every second.

📚 Theory
1. Classical Computer Vision
Classical computer vision relies on hand-crafted features and traditional machine learning methods rather than deep learning.

2. Haar Cascade Classifier
The Haar Cascade algorithm:

Uses Haar-like features
Uses integral images for fast computation
Applies AdaBoost for feature selection
Uses cascade stages for efficient detection
It is widely used for:

Face detection
Eye detection
Object detection (basic cases)
3. Detection Pipeline
Capture image from webcam
Convert image to grayscale
Detect faces
Detect eyes inside face region
Draw bounding boxes
Display result


📊 Results
The system successfully:

Captured real-time webcam frames
Detected faces using Haar Cascade
Detected eyes inside the detected face region
Displayed detection results at 1 frame per second
Performance:

Lightweight
CPU-based
Suitable for real-time basic applications
✅ Conclusion
This project demonstrates a classical computer vision pipeline using Haar Cascade classifiers.

Key Learnings:

Understanding of feature-based object detection
Real-time image processing
Region of Interest (ROI) processing
Practical use of OpenCV
Although deep learning provides higher accuracy, Haar Cascade remains a fast and lightweight method for basic face detection tasks.

This project successfully implements a complete classical face and eye detection system.
