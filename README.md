# Exploring biological motion perception in two-stream convolutional neural networks

Tensorflow implementation of "Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception"

## Publication
#### Comparison between Two-stream Convolutional Neural Networks and Human Biological Motion Perception
Yujia Peng, Hannah Lee, Tianmin Shu, and Hongjing Lu

## Getting started

Clone this repository 
```
git clone https://github.com/yjpeng11/BM_CNN.git
```

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

To extract static image frames and optical flow data, run:
```
python3 opencv_opticalflow.py
```

### Step 2: Preprocessing model inputs

The script "folder2list_leftright.m" generates a .txt file with a list of videos.

The script "make_test_mat2_leftright.m" takes the .txt file as input to generate a .mat file with directories of the saved static images 
and optical flow images.

To process the list of files in .mat into python-friendly format, run:
```
python prepare_list.py
```
The name2id and data_path need to be changed accordingly.

### Step 3: Training and testing CNN models

The CNN scripts implement the spatial CNN of appearance, the temporal CNN of motion, and the two-stream CNN.
```
python -m virtualenv cnn
source cnn/bin/activate
pip install -r requirements_cnn.txt
```

The running of CNN scripts requires an environment with python 2.7. Create an environment with all packages from requirements_cnn.txt installed (Note: please double check the CUDA version on your machine).

To train the spatial CNN:
```
python CNN_image.py --mode 0
```
To train the temporal CNN:
```
python CNN_flow.py --mode 0
```
To train the two-stream CNN:
First need to extract model features from the spatial and temporal CNN:
```
python CNN_image.py --mode 1
python CNN_flow.py --mode 1
python CNN_fusion.py --mode 0
```

Change mode 0 to mode 1 for testing.

### Step 4: Transfer training
First, extract video features fomr the original spatial and temporal CNNs by running:
```
python CNN_image.py --mode 1
python CNN_flow.py --mode 1
```
This will generate files containing model features of training and testing stimuli. Then, change the # of video classes in CNN_image_freeze.py, CNN_image_freeze.py, and CNN_flow_freeze.py.
For example, if we now have two categories (e.g. facing left or facing right), change default of num-classes to be 2:
```
parser.add_argument('--num-classes', type=int, default=2, help='Number of output labels')
```
to perform restricted transfer train on the spatial CNN:
```
python CNN_image_freeze.py --mode 0
```
To perform restricted transfer train on the temporal CNN:
```
python CNN_flow_freeze.py --mode 0
```
To perform restricted transfer train on the two-stream CNN:
```
python CNN_fusion_freeze.py --mode 0
```
Change the name of the softmax layer for different restricted transfer training ('new_dense_3' right now), because in the command model.load_weights, option by_name was set to be True so only model weights of layers that share the same names of saved models will be loaded. With a wrong name, the saved model weights of the layer will not be successfully loaded and may introduce random model weights.
```
softmax = Dense(args.num_classes, trainable=False, activation='softmax', kernel_initializer=my_init, name='new_dense_3')(dp2)
```
