"""
Script that creates the network architecture.

Before running the script, it is important that:
- The libraries from the requirements.txt file are installed

Author: R. Lucassen (r.t.lucassen@student.tue.nl)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, concatenate, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


def residual_block(input_data, filters, kernel_size, kernel_initializer):
    """ Function to add a convolutional block to the network.
    Specify the number of convolutional filters in the layers using the 'filters' argument.
    Specify the kernel size of the kernels in the filters using the 'kernel_size' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Activation('relu')(input_data)
    x = Conv2D(filters, kernel_size, activation=None, padding='same', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, activation=None, padding='same', kernel_initializer=kernel_initializer)(x)
    x = Add()([x, input_data])
    return x


def funnel_network_1(input_data, down_fact, kernel_initializer):
    """ Function to add the first funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(32*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(32*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(64*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(64*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(32*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_2(input_data, down_fact, kernel_initializer):
    """ Function to add the second funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(64*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(64*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(32*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_3(input_data, down_fact, kernel_initializer):
    """ Function to add the third funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(64*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_4(input_data, down_fact, kernel_initializer):
    """ Function to add the fourth funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(64*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_5(input_data, down_fact, kernel_initializer):
    """ Function to add the fifth funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(128*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_6(input_data, down_fact, kernel_initializer):
    """ Function to add the sixth funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(128*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_7(input_data, down_fact, kernel_initializer):
    """ Function to add the seventh funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    x = residual_block(x, int(256*down_fact), (3, 3), kernel_initializer)
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(x)
    return x


def funnel_network_8(input_data, down_fact, kernel_initializer):
    """ Function to add the eigth funneling subnetwork.
    Specify the factor used to indicate the number of filters compared to the normal architecture using the 'down_fact' argument.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    x = Conv2D(int(256*down_fact), (2, 1), activation=None, strides=(2, 1), kernel_initializer=kernel_initializer)(input_data)
    return x


def network(optimizer, image_shape=(256, 512, 1), kernel_initializer = 'glorot_uniform', down_fact=1, up_fact=1):
    """ Function create the network architecture.
    Specify the shape of the input image using the 'image_shape' argument. 
    Specify the factor used to indicate the number of filters compared to the normal architecture in the downsampling path and upsampling path
    using the 'down_fact' argument and the 'upsampling_fact' argument, respectively.
    Specify the weight initialization method using the 'kernel_initializer' argument.
    """ 
    # declare variables for cropping the annotated output layer to only include the annotated coordinates
    inside_start = 150
    outside_start = 80
    bias_init = tf.constant_initializer(np.array([148.0, 110.1]))

    # start constructing the network architecture
    network_input = Input(shape=image_shape)

    w1 = Activation('relu')(network_input)
    w1 = Conv2D(int(32*down_fact), (3, 3), activation=None, strides=1, padding='same', kernel_initializer=kernel_initializer)(w1)
    w1 = residual_block(w1, int(32*down_fact), (3, 3), kernel_initializer)
    # split 1
    f1 = funnel_network_1(w1, down_fact, kernel_initializer)
    w2 = Conv2D(int(32*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w1)
    w2 = residual_block(w2, int(32*down_fact), (3, 3), kernel_initializer)
    # split 2
    f2 = funnel_network_2(w2, down_fact, kernel_initializer)
    w3 = Conv2D(int(64*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w2)
    w3 = residual_block(w3, int(64*down_fact), (3, 3), kernel_initializer)
    # split 3
    f3 = funnel_network_3(w3, down_fact, kernel_initializer)
    w4 = Conv2D(int(64*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w3)
    w4 = residual_block(w4, int(64*down_fact), (3, 3), kernel_initializer)
    # split 4
    f4 = funnel_network_4(w4, down_fact, kernel_initializer)
    w5 = Conv2D(int(128*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w4)
    w5 = residual_block(w5, int(128*down_fact), (3, 3), kernel_initializer)
    # split 5
    f5 = funnel_network_5(w5, down_fact, kernel_initializer)
    w6 = Conv2D(int(128*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w5)
    w6 = residual_block(w6, int(128*down_fact), (3, 3), kernel_initializer)
    # split 6
    f6 = funnel_network_6(w6, down_fact, kernel_initializer)
    w7 = Conv2D(int(256*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w6)
    w7 = residual_block(w7, int(256*down_fact), (3, 3), kernel_initializer)
    # split 7
    f7 = funnel_network_7(w7, down_fact, kernel_initializer)
    w8 = Conv2D(int(256*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w7)
    w8 = residual_block(w8, int(256*down_fact), (3, 3), kernel_initializer)
    # split 8
    f8 = funnel_network_8(w8, down_fact, kernel_initializer)
    w9 = Conv2D(int(1024*down_fact), (2, 2), activation=None, strides=(2, 2), kernel_initializer=kernel_initializer)(w8)
    w9 = residual_block(w9, int(1024*down_fact), (1, 3), kernel_initializer)
    w9 = residual_block(w9, int(1024*down_fact), (1, 3), kernel_initializer)
    w9 = UpSampling2D((1, 2))(w9)
    # concatenation 1
    w8 = concatenate([f8, w9])
    w8 = Conv2D(int(256*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w8)
    w8 = residual_block(w8, int(256*up_fact), (1, 3), kernel_initializer)
    w8 = UpSampling2D((1, 2))(w8)
    # concatenation 2
    w7 = concatenate([f7, w8])
    w7 = Conv2D(int(256*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w7)
    w7 = residual_block(w7, int(256*up_fact), (1, 3), kernel_initializer)
    w7 = UpSampling2D((1, 2))(w7)
    # concatenation 3
    w6 = concatenate([f6, w7])
    w6 = Conv2D(int(128*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w6)
    w6 = residual_block(w6, int(128*up_fact), (1, 3), kernel_initializer)
    w6 = UpSampling2D((1, 2))(w6)
    # concatenation 4
    w5 = concatenate([f5, w6])
    w5 = Conv2D(int(128*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w5)
    w5 = residual_block(w5, int(128*up_fact), (1, 3), kernel_initializer)
    w5 = UpSampling2D((1, 2))(w5)
    # concatenation 5
    w4 = concatenate([f4, w5])
    w4 = Conv2D(int(64*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w4)
    w4 = residual_block(w4, int(64*up_fact), (1, 3), kernel_initializer)
    w4 = UpSampling2D((1, 2))(w4)
    # concatenation 6
    w3 = concatenate([f3, w4])
    w3 = Conv2D(int(64*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w3)
    w3 = residual_block(w3, int(64*up_fact), (1, 3), kernel_initializer)
    w3 = UpSampling2D((1, 2))(w3)
    # concatenation 7
    w2 = concatenate([f2, w3])
    w2 = Conv2D(int(32*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w2)
    w2 = residual_block(w2, int(32*up_fact), (1, 3), kernel_initializer)
    w2 = UpSampling2D((1, 2))(w2)
    # concatenation 8
    w1 = concatenate([f1, w2])
    w1 = Conv2D(int(32*up_fact), (1, 1), activation=None, kernel_initializer=kernel_initializer)(w1)
    w1 = residual_block(w1, int(32*up_fact), (1, 3), kernel_initializer)
    w1 = Activation('relu')(w1)
    w1 = Conv2D(2, (1, 3), activation=None, padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_init)(w1)

    # crop the output layer
    inside = Lambda(lambda x : x[:,:,inside_start:, 0])(w1)
    outside = Lambda(lambda x : x[:,:,outside_start:, 1])(w1)
    combined = concatenate([inside, outside],axis=-1)
    network_output = Flatten()(combined)

    model = Model(network_input, network_output)

    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    return model