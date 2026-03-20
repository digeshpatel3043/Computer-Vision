## File: DeepStitching.ipynb

## Description
This notebook performs panorama stitching using deep feature matching instead of hand-crafted keypoints like ORB or SIFT. It uses LoFTR (Local Feature TRansformer), a deep learning-based matcher, to find dense correspondences between images. These correspondences are then used to estimate homography and stitch images into a panorama.

## Main Steps
1. Load images from folder
2. Resize them to avoid kernel crash
3. Convert images to grayscale tensors
4. Use pretrained LoFTR model for feature matching
5. Extract matched point correspondences
6. Estimate homography using RANSAC
7. Warp and stitch images
8. Blend overlap regions
9. Crop black borders
10. Save final deep panorama result

## Output Files
Saved inside: deep_stitching_output/
