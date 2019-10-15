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


def my_init(shape, dtype=None):
    """customized initialization"""
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def build_model(args,
                weights_path = None,
                output_feature = False):
    """build fusion model"""
    X_image = Input(shape = (7, 7, 512), name = 'input_image')
    X_flow  = Input(shape = (7, 7, 512), name = 'input_fusion')
    concat = concatenate([X_image, X_flow], axis = -1)
    fusion = Conv2D(512, (7, 7), activation = 'relu', padding = 'same', 
                    kernel_initializer = my_init, name = 'fusion')(concat)
    flatten = Reshape((512 * 7 * 7, ))(fusion)
    dense1 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_1')(flatten)
    dp1 = Dropout(args.dropout_rate, name = 'dp_1')(dense1)
    dense2 = Dense(4096, activation = 'relu', kernel_initializer = my_init, name = 'dense_2')(dp1)
    dp2 = Dropout(args.dropout_rate, name = 'dp_2')(dense2)
    softmax = Dense(args.num_classes, activation = 'softmax', kernel_initializer = my_init, name = 'dense_3')(dp2)

    if output_feature:
        model = Model(inputs = [X_image, X_flow], outputs = [softmax, dense2])
    else:
        model = Model(inputs = [X_image, X_flow], outputs = softmax)

    if weights_path:
        model.load_weights(weights_path, by_name = True)

    return model


def compute_confusion_matrix(softmax_output, Y_test, num_classes):
    Y_bar = np.argmax(softmax_output, axis=1)
    Y_GT = np.argmax(Y_test, axis=1)
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


def load_labels(path, args):
    """load labels"""
    train_data = cPickle.load(open(path))
    labels = train_data['labels'][::args.reduce_factor]
    counts = [0] * args.num_classes
    for label in labels:
        counts[label] += 1
    print "class counts:", counts
    Y = [label2vec(label, args.num_classes) for label in labels]
    Y = np.vstack(Y)
    return Y


def load_features(path, args):
    """load features"""
    X = np.load(path)
    return X[::args.reduce_factor]


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='CNN for image')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=15, help='Number of output labels')
    parser.add_argument('--max-nb', type=int, default=5000, help='Maximum number of instances of a class')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout-rate', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epoches')
    parser.add_argument('--img-rows', type=int, default=224, help='Image height')
    parser.add_argument('--img-cols', type=int, default=224, help='Image width')
    parser.add_argument('--img-chns', type=int, default=20, help='Image channels')
    parser.add_argument('--reduce-factor', type=int, default=1, help='Dataset size reducing factor')
    parser.add_argument('--mode', type=int, default=0, help='Running mode, 0 - training, 1 - testing')
    args = parser.parse_args()

    WORKSPACE = '/mnt/Data/2/ActionCNN_simulation/two_stream/hmdb51'
    
    # load data
    train_data_path = os.path.join(WORKSPACE, 'flow_train_data.pik')
    Y_train = load_labels(train_data_path, args)
    val_data_path = os.path.join(WORKSPACE, 'flow_test_data.pik')
    Y_val = load_labels(val_data_path, args)
    
    X_image_train_path = os.path.join(WORKSPACE, 'feature_maps_image_train.npy')
    X_image_train = load_features(X_image_train_path, args)
    
    X_image_val_path = os.path.join(WORKSPACE, 'feature_maps_image_val.npy')
    X_image_val = load_features(X_image_val_path, args)

    X_flow_train_path = os.path.join(WORKSPACE, 'feature_maps_flow_train.npy')
    X_flow_train = load_features(X_flow_train_path, args)
    
    X_flow_val_path = os.path.join(WORKSPACE, 'feature_maps_flow_val.npy')
    X_flow_val = load_features(X_flow_val_path, args)

    # set up callbacks
    CHECKPOINT_DIR = WORKSPACE + '/checkpoints_fusion/'
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_PATH = CHECKPOINT_DIR + 'weights_dropout{0}.hdf5'.format(args.dropout_rate)

    checkpointer = ModelCheckpoint(filepath = CHECKPOINT_PATH, 
                                   monitor='val_acc', 
                                   verbose = 1,
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   mode='auto')
    early_stopper = EarlyStopping(monitor='val_acc', patience = 10, verbose = 0, mode = 'auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor = 0.5, patience = 3, min_lr = 1e-6)

    # training or testing the network
    sgd = SGD(lr = args.lr, decay = 0.000, momentum = 0.9, nesterov = True)
    steps_train = int(np.ceil(len(Y_train) / float(args.batch_size)))
    steps_val = int(np.ceil(len(Y_val) / float(args.batch_size)))
    if args.mode == 0: # training
        model = build_model(args, CHECKPOINT_PATH)
        print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])
        model.fit([X_image_train, X_flow_train], 
                  Y_train, 
                  epochs = args.num_epochs, 
                  batch_size = args.batch_size,
                  verbose = 1,
                  validation_data = ([X_image_val, X_flow_val], Y_val),
                  callbacks = [checkpointer, early_stopper, reduce_lr])

        model.load_weights(CHECKPOINT_PATH)
        # score = model.evaluate(X_test, Y_test, batch_size=5)
        softmax_output = model.predict([X_image_val, X_flow_val], verbose = 1)
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        print ConfusionMat
        print 'score:', score
    if args.mode == 1:
        model = build_model(args)
        print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])
        model.load_weights(CHECKPOINT_PATH)
        # score = model.evaluate(X_test, Y_test, batch_size=5)
        softmax_output = model.predict([X_image_val, X_flow_val], verbose = 1)
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        np.save(WORKSPACE + '/conf_mat_val_fusion.npy', ConfusionMat)
        np.save(WORKSPACE + '/conf_mat_val_fusion_softmax.npy', softmax_output)
        np.save(WORKSPACE + '/conf_mat_val_fusion_softmax_Y_val.npy', Y_val)
        print 'ConfusionMat:', ConfusionMat
        print 'score:', score
    else: # extract features
        model = build_model(args, CHECKPOINT_PATH, output_feature = True) 
        print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])
        output_train = model.predict([X_image_train, X_flow_train], verbose = 1)
        output_val = model.predict([X_image_val, X_flow_val], verbose = 1)
        #print output_val[0].shape
        score, ConfusionMat = compute_confusion_matrix(output_val[0], Y_val, args.num_classes)
        print ConfusionMat
        print 'score:', score
        features_train, features_val = output_train[1], output_val[1]
        np.save(WORKSPACE + '/features_fusion_train.npy', features_train)
        np.save(WORKSPACE + '/features_fusion_val.npy', features_val)
        np.save(WORKSPACE + '/conf_mat_val_fusion.npy', ConfusionMat)
        
  
 
        

