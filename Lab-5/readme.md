# Image Stitching and Panorama Creation
## Project Overview
This project focuses on panorama generation and image stitching using multiple computer vision approaches. It contains three Jupyter Notebook files, each demonstrating a different method for creating panoramic images from overlapping photos.

The project covers:
  - Manual panorama stitching using feature matching and homography
  - Deep feature-based stitching using LoFTR
  - 360-degree panorama creation using OpenCV Stitcher

## Project Files
1. OpenCV-Manual: Manual feature detection and matching using ORB, Homography estimation using RANSAC, Sequential image stitching, OpenCV built-in panorama stitching, Comparison of manual and automatic panorama methods.

2. DeepStitching: Deep feature matching using LoFTR, Homography computation from learned correspondences, Panorama generation using deep matching, Stitching with smaller resized images for memory safety.

3. 360-Panorama: Panorama stitching for 6–8 overlapping images, OpenCV Stitcher in PANORAMA mode, OpenCV Stitcher in SCANS mode, Comparison of both modes, Final best-output selection for 360 panorama creation.

## Objectives
  - Understand panorama generation using overlapping images
  - Implement manual image stitching pipeline
  - Apply feature detection and matching techniques
  - Estimate homography between images
  - Use OpenCV built-in panorama tools
  - Explore deep learning-based feature matching
  - Create wider and 360° style panoramas
  - Compare traditional and deep stitching methods

## Technologies Used
  - Python
  - OpenCV
  - NumPy
  - Matplotlib
  - PyTorch
  - Kornia / Kornia Feature
  - LoFTR
  - KaggleHub

## Core Concepts Used
1. Feature Detection: Feature detection identifies important keypoints such as corners or textured regions in an image.
2. Feature Matching: Feature matching finds corresponding points between overlapping images.
3. Homography Estimation: Homography is a transformation matrix that maps points from one image plane to another.It is used to align overlapping images before stitching.
4. RANSAC: RANSAC removes incorrect matches and estimates a robust homography using only inlier correspondences.
5. Image Warping: Warping transforms one image onto the coordinate system of another image using the estimated homography matrix.
6. Blending: After alignment, overlapping image regions are combined using simple averaging to reduce visible seams.
7. Border Cropping: Black borders caused by image warping are removed to produce a cleaner panorama.

## Applications
  - Panorama generation
  - Image mosaicing
  - Virtual tours
  - Drone image stitching
  - Landscape photography
  - Scene reconstruction
  - Robotics and mapping
  - Computer vision research

## Limitations
  - Stitching quality depends heavily on image overlap
  - Incorrect image order may cause failure
  - Moving objects can reduce panorama quality
  - Exposure differences may create visible seams
  - Deep stitching can be memory intensive
  - 360 stitching may fail if images are not captured consistently
