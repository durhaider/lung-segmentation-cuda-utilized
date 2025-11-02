# CUDA Lung Segmentation Kernels

This repository contains only the CUDA kernels used for accelerating lung segmentation from CT scans. These kernels were created to speed up the segmentation pipeline by offloading key stages to the GPU.

Originally, the same processing on CPU was taking around 30 seconds. After moving the heavy steps to CUDA, the segmentation time was reduced to about 7–8 seconds. The main goal of this work was to make the segmentation fast without losing accuracy.

## What this contains

* Custom CUDA kernel files:

  * `preprocess.cu`
  * `preprocess.cuh`
* GPU code for:

  * Filtering and preparing CT voxel data
  * Mask building operations
  * Morphological processing on 3D volumes
  * Keeping only the lung regions

## Key Improvements

* Time reduced from about **30 seconds to 7–8 seconds**
* Uses **pure CUDA kernels** for performance
* Works on volumetric CT data
* Designed for lung CT segmentation tasks

## How it works (simple explanation)

* CT scan slices are combined into a 3D volume
* The volume is passed to the GPU
* CUDA kernels apply smoothing and thresholding
* Air region is separated from body region
* Flood fill removes outside air
* Only lung regions are kept
* Morphological closing cleans the final mask

## Requirements to use these kernels

* NVIDIA GPU with CUDA support
* CUDA Toolkit (12 or higher)
* Any C++ program that loads the CUDA kernels and passes CT volume data

