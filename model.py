from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling1D,
    Flatten,
    Concatenate
    )
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution1D)
from keras.layers.pooling import (
    MaxPooling1D,
    AveragePooling1D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Sequential, load_model


weight_decay = 1e-5
def _conv_bn_relu(nb_filter, kernel_size):
    def f(input):
        conv_a = Convolution1D(nb_filter, kernel_size, activation='relu', padding='same')(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        return act_a
    return f
    
def _conv_bn_relu_x2(nb_filter, kernel_size):
    def f(input):
        conv_a = Convolution1D(nb_filter, kernel_size, 
                               activation='relu', padding='same')(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution1D(nb_filter, kernel_size,
                               activation='relu', padding='same')(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def FCRN_A_base(input):
    block1 = _conv_bn_relu(32,3)(input)
    pool1 = MaxPooling1D(pool_size=(2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu(64,3)(pool1)
    pool2 = MaxPooling1D(pool_size=(2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu(128,3)(pool2)
    pool3 = MaxPooling1D(pool_size=(2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(512,3)(pool3)
    # =========================================================================
    up5 = UpSampling1D(size=(2))(block4)
    block5 = _conv_bn_relu(128,3)(up5)
    # =========================================================================
    up6 = UpSampling1D(size=(2))(block5)
    block6 = _conv_bn_relu(64,3)(up6)
    # =========================================================================
    up7 = UpSampling1D(size=(2))(block6)
    block7 = _conv_bn_relu(32,3)(up7)
    return block7

def FCRN_A_base_v2(input):
    block1 = _conv_bn_relu_x2(32,3)(input)
    pool1 = MaxPooling1D(pool_size=(2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(64,3)(pool1)
    pool2 = MaxPooling1D(pool_size=(2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(128,3)(pool2)
    pool3 = MaxPooling1D(pool_size=(2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(512,3)(pool3)
    # =========================================================================
    up5 = UpSampling1D(size=(2))(block4)
    block5 = _conv_bn_relu_x2(128,3)(up5)
    # =========================================================================
    up6 = UpSampling1D(size=(2))(block5)
    block6 = _conv_bn_relu_x2(64,3)(up6)
    # =========================================================================
    up7 = UpSampling1D(size=(2))(block6)
    block7 = _conv_bn_relu_x2(32,3)(up7)
    return block7

def U_net_base(input, nb_filter = 64):
    block1 = _conv_bn_relu_x2(nb_filter,3)(input)
    pool1 = MaxPooling1D(pool_size=(2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(nb_filter,3)(pool1)
    pool2 = MaxPooling1D(pool_size=(2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(nb_filter,3)(pool2)
    pool3 = MaxPooling1D(pool_size=(2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu_x2(nb_filter,3)(pool3)
    up4 = Concatenate(axis=-1)([UpSampling1D(size=(2))(block4), block3])
    # =========================================================================
    block5 = _conv_bn_relu_x2(nb_filter,3)(up4)
    up5 = Concatenate(axis=-1)([UpSampling1D(size=(2))(block5), block2])
    # =========================================================================
    block6 = _conv_bn_relu_x2(nb_filter,3)(up5)
    up6 = Concatenate(axis=-1)([UpSampling1D(size=(2))(block6), block1])
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter,3)(up6)
    return block7

def buildModel_FCRN_A (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = FCRN_A_base (input_)
    # =========================================================================
    density_pred =  Convolution1D(1, 1, activation='linear',\
                                  name='pred')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def buildModel_FCRN_A_v2 (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = FCRN_A_base_v2 (input_)
    # =========================================================================
    density_pred =  Convolution1D(1, 1, activation='linear',\
                                  name='pred')(act_)
    # =========================================================================
    model = Model (inputs = input_, outputs = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def buildModel_U_net (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = U_net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution1D(1, 1, activation='linear',\
                                  name='pred')(act_)
    # =========================================================================
    model = Model (inputs = input_, outputs = density_pred)
    opt = RMSprop(1e-3)
    model.compile(optimizer = opt, loss = 'mse')
    return model


def train(x, y, length, channels, batch_size=64, lr=3e-4, epochs=500, filepath="model.h5"):
    import tensorflow as tf 
    from tensorflow.keras import layers, regularizers
    from keras.constraints import max_norm, unit_norm
    import keras.callbacks
    from keras.callbacks import TensorBoard
    from keras.callbacks import ModelCheckpoint
    import os

    # define the checkpoint
    # define the checkpoint
    if(os.path.exists(filepath)):
      print(f"{filepath} Checkpoint Loaded")
      model = load_model(filepath)
    else:
      print("No model checkpoint, starting from scratch...")
      model = buildModel_FCRN_A((length, channels))

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss='mse',optimizer=opt,metrics='accuracy')
    print(model.summary())

    history = model.fit(x, y, batch_size=batch_size,epochs=epochs,validation_split=0.1,verbose=True, callbacks=callbacks_list)

    print(model.summary())

    model.save("my_model")

    import matplotlib.pyplot as plt 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss over time')
    plt.legend(['train','val'])
    plt.savefig('loss.png', bbox_inches='tight')

    return model


def evaluate(model_path, x):
    model = load_model(model_path)
    return model.predict(x)



