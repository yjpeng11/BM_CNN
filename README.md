# Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception

Here are the Python scripts used in the current project.

## Publication
#### Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception
Yujia Peng, Hannah Lee, Tianmin Shu, and Hongjing Lu

## Getting started

### Prerequisites
* Linux Ubuntu 16.04
* Python 3
* NVIDIA GPU + CUDA 9.0

### Step 1: Extracting model inputs through OpenCV

The OpenCV script extracts static image frames from videos and further generates optical flow images.

The script requires a environment with python 3.5, numpy, and OpenCV-Python.

To run the script, run "python3 opencv_opticalflow.py".

### Step 2: Preprocessing model inputs in MATLAB

The script "folder2list_leftright.m" generates a .txt file with a list of videos.

The script "make_test_mat2_leftright.m" takes the .txt file as input to generate a .mat file with directories of the saved static images 
and optical flow images.

### Step 3: Training and testing CNN models

The CNN scripts implement the spatial CNN of appearance, the temporal CNN of motion, and the two-stream CNN.

First, run "python prepare_flow1.py" to process the list of files in .mat into python format. The name2id and data_path need to be changed accordingly.

The running of CNN scripts requires an environment with python 2.7, tensorflow-gpu 1.9.0, tensorflow-tensorboard 1.5.1, QyPy 1.7.0, and keras 2.2.4, h5py 2.8.0, scipy, skiimage, nnumpy.

Run "python CNN_image.py --mode 0" for training the spatial CNN.
Run "python CNN_flow.py --mode 0" for training the temporal CNN.
Run "python CNN_fusion.py --mode 0" for training the two-stream CNN.
Change mode 0 to mode 1 for testing.
