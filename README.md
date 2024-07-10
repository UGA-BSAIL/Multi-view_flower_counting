# Multi-view_flower_counting
This repository contains the source code and instruction for running models multi-view cotton flower counting. Detailed methodology and results can be found from our paper.
# Install
Pip install all requirements in a Python>=3.8 environment with PyTorch>=1.8.
# Usage
1. Download the trained detector or train your own data followed the YOLOv8 repository.
2. Run the flowertrackingRAFT.py to get the multi-object tracking results for three views and remove the replicates across frames.
3. Run the projection3Dto2D.py to project the tracked flowers from side views to the middle view.
4. Run the hierarchy2.py to cluster all flowers from three views to remove the replicates across cameras and get the number of flowers.
# Datasets
All data can be accessed on figshare:<https://figshare.com/s/59378ccfe665de73b221>. Please refresh the page if you cannot see the datasets.
