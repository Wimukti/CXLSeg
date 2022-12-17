# CXLSeg: Chest X-ray Dataset with Lung Segmentation

CXLSeg is a publicly available database of segmented chest x-rays and corresponding masks based on MIMIC-CXR dataset.

This repository is intended to support use of the CXLSeg by providing code for different deep learning tasks. Feedback and contributions are always welcome!

> Version 1.0.0

## Folder Structure

- [x] classification
    - [x] evaluate
    - [x] split
    - [x] train
    - [x] utils
- [x] segmentation
    - [x] evaluate
    - [x] split
    - [x] train
    - [x] utils
- [x] report generation
    - [x] split
- [x] notebooks
    - [x] preview image
- [x] sample images
- 
## Files

- **evaluate** - python scripts to evaluate classification and segmentation models.
- **split** - python script to create test-train-validate datasets for each use case
- **train** - python script to train the corresponding classification and segmentation tasks
- **utils** - additional utility functions
- **preview-images.ipyb** - notebook to preview the original, segmented and mask image.


## Important Notes
This repository provide everything that you need to train classification, segmentation and report generation models.
You need to download the CXLSeg dataset and use the CSVs provided with it.
Additionally, [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset needs to be downloaded in order to utilize this for report generation tasks since the original images are not inlcuded in CXLSeg.