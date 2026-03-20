## File: 360-Panorama.ipynb

## Description
This notebook is designed for stitching multiple overlapping images into a wide or near-360 panorama using OpenCV’s built-in stitcher. It supports stitching 6–8 images and compares PANORAMA mode & SCANS mode. It also includes optional feature match visualization between the first two images.

## Main Steps
1. Load input images from folder
2. Resize large images for stable processing
3. Display input sequence
4. Detect ORB features between first two images
5. Visualize feature matches
6. Run OpenCV Stitcher in PANORAMA mode
7. Run OpenCV Stitcher in SCANS mode
8. Compare both outputs
9. Select the best final stitched panorama
10. Save final result

## Output Files
Saved inside: output_360_panorama/
