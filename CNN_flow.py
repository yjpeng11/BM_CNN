import numpy as np
import matplotlib.pyplot as plt
import h5py
import cPickle
import random
import os
import argparse
import progressbar
from scipy.stats import norm
from scipy.misc import imresize
from numpy import linalg as LA
from multiprocessing import Process
from skimage.io import imread, imshow
from skimage.transform import resize
from time import sleep

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, TimeDistributed
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives, initializers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, RMSprop
from keras.applications.vgg16 import VGG16

#%%
def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def preprocess_image_batch(u_paths, v_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []
    """preprocess a batch of optical flow images"""
    for u_path_stack, z_path_stack in zip(u_paths, v_paths):
        img_stack_list = []
        path_stack = u_path_stack + z_path_stack
        for im_path in path_stack:
            img = imread(im_path, mode='L')
            if img_size:
                img = imresize(img,img_size)
            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            # img[:, :] -= 123.68 * 0.299 + 116.779 * 0.587 + 103.939 * 0.114 
            img[:, :] -= 128.0
            #print "img shape", img.shape
            img_stack_list.append(img)
 
        img_stack = np.stack(img_stack_list, axis=-1)
        #img_stack.transpose((1, 2, 0))
        # print "img stack shape", img_stack.shape
        img_list.append(img_stack)

    try:
        #print "img_list shape", np.array(img_list).shape
        #print u_path_stack
        img_batch = np.stack(img_list, axis=0)
        #print "img batch shape", img_batch.shape
        #raw_input("")
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')
 
    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


def getBatchGenerator(x_u, x_v, y, args, is_shuffle = True):
    """sample a batch"""
    files_u0, files_v0 = list(x_u), list(x_v)
    labels0 = list(y)
    cur_batch_index = 0
    nb_batch = 0
    N = len(files_u0)
    files_u = files_u0
    files_v = files_v0
    labels = labels0
    while 1:
        if cur_batch_index == 0 and is_shuffle:
            index_shuf = range(N)
            random.shuffle(index_shuf)
            files_u, files_v = [], []
            labels = []
            for i in index_shuf:
                files_u.append(files_u0[i])
                files_v.append(files_v0[i])
                labels.append(labels0[i])

        start_id = cur_batch_index
        end_id = min(cur_batch_index + args.batch_size, N)

        X = preprocess_image_batch(files_u[start_id:end_id], 
                                   files_v[start_id:end_id],
                                   img_size = (args.img_rows, args.img_cols))
        Y = [label2vec(label, args.num_classes) for label in labels[start_id:end_id]]
        Y = np.vstack(Y)

        cur_batch_index = end_id
        if cur_batch_index >= N:
            cur_batch_index = 0

        yield X, Y


def my_init(shape, dtype=None):
    """customized initialization"""
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)

#%%
# def build_model(args,
#                 weights_path = None):
#     """build image model"""
#     X = Input(shape = (args.img_rows, args.img_cols, args.img_chns), name = 'input_image')
#     CNN_VGG16 = VGG16(include_top = True,
#                 weights = None,
#                 input_shape = (args.img_rows, args.img_cols, args.img_chns))
#     CNN = Model(inputs = CNN_VGG16.input, outputs = CNN_VGG16.get_layer('flatten').output)
#     flatten = CNN(X)
#     dense1 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_1')(flatten)
#     dp1 = Dropout(args.dropout_rate, name = 'dp_1')(dense1)
#     dense2 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_2')(dp1)
#     dp2 = Dropout(args.dropout_rate, name = 'dp_2')(dense2)
#     softmax = Dense(args.num_classes, activation = 'softmax', kernel_initializer = my_init, name = 'dense_3')(dp2)
#
#     model = Model(inputs = X, outputs = softmax)
#
#     if weights_path:
#         model.load_weights(weights_path, by_name = True)
#
#     return model

def build_model(args,
                weights_path = None,
                output_feature = False):
    """build flow model"""
    X = Input(shape = (args.img_rows, args.img_cols, args.img_chns), name = 'input_flow')
    CNN_VGG16 = VGG16(include_top = True,
                weights = None, 
                input_shape = (args.img_rows, args.img_cols, args.img_chns))
    CNN = Model(inputs = CNN_VGG16.input, outputs = CNN_VGG16.get_layer('flatten').output)
    flatten = CNN(X)
    dense1 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_1')(flatten)
    dp1 = Dropout(args.dropout_rate, name = 'dp_1')(dense1)
    dense2 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_2')(dp1)
    dp2 = Dropout(args.dropout_rate, name = 'dp_2')(dense2)
    softmax = Dense(args.num_classes, activation = 'softmax', kernel_initializer = my_init, name = 'dense_3')(dp2)

    CNN_features = Reshape((7, 7, 512))(flatten)
    if output_feature:
        model = Model(inputs = X, outputs = [softmax, CNN_features])
    else:
        model = Model(inputs = X, outputs = softmax)

    if weights_path:
        model.load_weights(weights_path, by_name = True)

    return model


def compute_confusion_matrix(softmax_output, Y_GT, num_classes):
    Y_bar = np.argmax(softmax_output, axis=1)
    # Y_GT = np.argmax(Y_test, axis=1)
    # print Y_bar
    N_test = len(Y_GT)
    ConfusionMat = np.zeros((num_classes, num_classes), dtype=np.float32)
    score = float(sum([1 for i in range(N_test) if Y_GT[i] == Y_bar[i]])) / N_test
    for i in range(N_test):
        ConfusionMat[Y_GT[i], Y_bar[i]] += 1.0
    print ConfusionMat
    for label in range(num_classes):
        ConfusionMat[label, :] /= sum(ConfusionMat[label, :])
    return (score, ConfusionMat)

#%%
if __name__ == '__main__':
    # args
    a=5
    parser = argparse.ArgumentParser(description='CNN for image')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=15, help='Number of output labels')
    parser.add_argument('--max-nb', type=int, default=5000, help='Maximum number of instances of a class')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of training epoches')
    parser.add_argument('--img-rows', type=int, default=224, help='Image height')
    parser.add_argument('--img-cols', type=int, default=224, help='Image width')
    parser.add_argument('--img-chns', type=int, default=20, help='Image channels')
    parser.add_argument('--reduce-factor', type=int, default=1, help='Dataset size reducing factor')
    parser.add_argument('--mode', type=int, default=1, help='Running mode, 0 - training, 1 - testing')
    args = parser.parse_args()

    WORKSPACE = '/mnt/Data/2/ActionCNN_simulation/two_stream/hmdb51'
    
    # load data
    train_data_path = os.path.join(WORKSPACE, 'flow_train_data.pik')
    train_data = cPickle.load(open(train_data_path))
    X_u_train = train_data['flow_u'][::args.reduce_factor]
    X_v_train = train_data['flow_v'][::args.reduce_factor]

    #totalsplit = 4
    #split = 1
    #X_u_train = X_u_train[((split-1)*len(X_u_train)//totalsplit):(split*len(X_u_train)//totalsplit)]
    #X_v_train = X_v_train[((split-1)*len(X_v_train)//totalsplit):(split*len(X_v_train)//totalsplit)]

    print(len(X_u_train))

    Y_train = train_data['labels'][::args.reduce_factor]
    #Y_train = Y_train[((split-1)*len(Y_train)//totalsplit):(split*len(Y_train)//totalsplit)]
    print(len(Y_train))

    counts = [0] * args.num_classes
    for label in Y_train:
        counts[label] += 1
    print "train class counts:", counts
    
    val_data_path = os.path.join(WORKSPACE, 'flow_test_data.pik')
    val_data = cPickle.load(open(val_data_path))
    X_u_val = val_data['flow_u'][::args.reduce_factor]
    X_v_val = val_data['flow_v'][::args.reduce_factor]
    Y_val = val_data['labels'][::args.reduce_factor]
    counts = [0] * args.num_classes
    for label in Y_val:
        counts[label] += 1
    print "val class counts:", counts

    # set up callbacks
    CHECKPOINT_DIR = WORKSPACE + '/checkpoints_flow/'
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_PATH = CHECKPOINT_DIR + 'weights_dropout{0}.hdf5'.format(args.dropout_rate)

    checkpointer = ModelCheckpoint(filepath = CHECKPOINT_PATH, 
                                   monitor='val_loss', 
                                   verbose = 1,
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   mode='auto')
    early_stopper = EarlyStopping(monitor='val_loss', patience = 10, verbose = 0, mode = 'auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 3, min_lr = 1e-6)

    # training or testing the network
    sgd = SGD(lr = args.lr, decay = 0.000, momentum = 0.9, nesterov = True)
    steps_train = int(np.ceil(len(Y_train) / float(args.batch_size)))
    steps_val = int(np.ceil(len(Y_val) / float(args.batch_size)))
    if args.mode == 0: # training
        model = build_model(args,CHECKPOINT_PATH)
        #print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])

        model.fit_generator(generator=getBatchGenerator(X_u_train, X_v_train, Y_train, args),
                            steps_per_epoch = steps_train, epochs = args.num_epochs,
                            validation_data = getBatchGenerator(
                                X_u_val, X_v_val, Y_val, args, is_shuffle = False),
                            validation_steps = steps_val,
                            callbacks = [checkpointer, reduce_lr, early_stopper])

        model.load_weights(CHECKPOINT_PATH)
        # score = model.evaluate(X_test, Y_test, batch_size=5)
        softmax_output = model.predict_generator(generator=getBatchGenerator(
            X_u_val, X_v_val, Y_val, args, is_shuffle = False), steps = steps_val, verbose = 1)
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        print 'ConfusionMat:',ConfusionMat
        print 'score:', score
        # np.save(WORKSPACE + '/conf_mat_val_flow.npy', ConfusionMat)
        #Confusionmat = np.load('sss')
    else: # testing
        model = build_model(args, CHECKPOINT_PATH, output_feature = True)
        print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])
        
        output_val = model.predict_generator(generator=getBatchGenerator(
            X_u_val, X_v_val, Y_val, args, is_shuffle = False), steps = steps_val, verbose = 1)
        print output_val[0].shape
        score, ConfusionMat = compute_confusion_matrix(output_val[0], Y_val, args.num_classes)
        print 'ConfusionMat:',ConfusionMat
        print 'score:', score
        CNN_features_val = output_val[1]
        np.save(WORKSPACE + '/feature_maps_flow_val.npy', CNN_features_val)
        # np.save(WORKSPACE + '/conf_mat_val_flow.npy', ConfusionMat)
        output_train = model.predict_generator(generator=getBatchGenerator(
            X_u_train, X_v_train, Y_train, args, is_shuffle = False), steps = steps_train, verbose = 1)
        CNN_features_train = output_train[1]
        np.save(WORKSPACE + '/feature_maps_flow_train.npy', CNN_features_train)
        
 
        

