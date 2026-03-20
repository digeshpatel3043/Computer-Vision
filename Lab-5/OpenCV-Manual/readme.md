## File: ImageStitching-OpenCV-Manual_Panorama.ipynb

## Description
This notebook demonstrates both manual panorama creation and OpenCV automatic panorama stitching. It first downloads a panorama dataset using KaggleHub, loads images, detects ORB features, matches keypoints, computes homography, and sequentially stitches images. It also compares the result with OpenCV’s built-in stitcher.

## Main Steps
1. Download and load panorama dataset
2. Read images from dataset folder
3. Detect ORB keypoints and descriptors
4. Match features between images using BFMatcher
5. Apply Lowe-style ratio filtering
6. Compute homography with RANSAC
7. Warp images onto a common canvas
8. Blend overlapping areas
9. Crop black borders
10. Compare:
      - Manual stitching
      - OpenCV PANORAMA mode
      - OpenCV SCANS mode

## Output Files
Saved inside: output_panorama/
