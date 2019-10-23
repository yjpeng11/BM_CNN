# Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception

Here are the Python scripts used in the current project.

## Publication
#### Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception
Yujia Peng, Hannah Lee, Tianmin Shu, and Hongjing Lu

## Getting started

### Prerequisites
* Linux Ubuntu 16.04
* Python 2 and 3
* NVIDIA GPU + CUDA 9.0

### Step 1: Extracting model inputs through OpenCV

The OpenCV script extracts static image frames from videos and further generates optical flow images.

Create an environment with all packages from requirements_opencv.txt installed.
```
python -m virtualenv opencv
source cnn/bin/activate
pip install -r requirements_opencv.txt
```

To run the script, run "python3 opencv_opticalflow.py".

### Step 2: Preprocessing model inputs in MATLAB

The script "folder2list_leftright.m" generates a .txt file with a list of videos.

The script "make_test_mat2_leftright.m" takes the .txt file as input to generate a .mat file with directories of the saved static images 
and optical flow images.

### Step 3: Training and testing CNN models

The CNN scripts implement the spatial CNN of appearance, the temporal CNN of motion, and the two-stream CNN.

First, run "python prepare_flow1.py" to process the list of files in .mat into python format. The name2id and data_path need to be changed accordingly.
```
python -m virtualenv cnn
source cnn/bin/activate
pip install -r requirements_cnn.txt
```

The running of CNN scripts requires an environment with python 2.7. Create an environment with all packages from requirements_cnn.txt installed (Note: please double check the CUDA version on your machine).

Run "python CNN_image.py --mode 0" for training the spatial CNN.
Run "python CNN_flow.py --mode 0" for training the temporal CNN.
Run "python CNN_fusion.py --mode 0" for training the two-stream CNN.
Change mode 0 to mode 1 for testing.
