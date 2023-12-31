# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:59:41 2022

@author: ZNDX001
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.framework import ops
import os
import math
from tensorflow import float32
 
#（1）通道注意力
def channel_attention(inputs, ratio=8):
    channel = inputs.shape[-1]
    
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)
 
    x_max = layers.Reshape([1,1,-1])(x_max)
    x_avg = layers.Reshape([1,1,-1])(x_avg)

    x_max = layers.Dense(channel/ratio)(x_max)
    x_avg = layers.Dense(channel/ratio)(x_avg)
 
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)

    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)

    x = layers.Add()([x_max, x_avg])
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])
    return x

def eca_block(inputs, b=1, gama=2):
    in_channel = inputs.shape[-1]
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    if kernel_size % 2:
        kernel_size = kernel_size
    else:
        kernel_size = kernel_size + 1
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = tf.nn.sigmoid(x)
    x = layers.Reshape((1,1,in_channel))(x)
    outputs = layers.multiply([inputs, x])
    return outputs

def XabBlock_s(input_tensor, middle_filters, out_filters, kernel_size, stride1, stride2, l2rate, keep_prop):
    x = layers.SeparableConvolution2D(filters = middle_filters, kernel_size = kernel_size, strides = stride1, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConvolution2D(filters = out_filters, kernel_size = kernel_size, strides = stride2, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = dropblock(x, keep_prop, 3)
    return x

def XabBlock_c(input_tensor, middle_filters, out_filters, kernel_size, stride1, stride2, l2rate, keep_prop):
    x = layers.Conv2D(filters = middle_filters, kernel_size = kernel_size, strides = stride1, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = out_filters, kernel_size = kernel_size, strides = stride2, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = dropblock(x, keep_prop, 3)
    return x

def UpBlock(input_tensor, filters, kernel_size, stride1, l2rate, keep_prop):
    x = layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = stride1, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = dropblock(x, keep_prop, 3)
    return x

def finalBlock(input_tensor, filters1, filters2, l2rate):
    x = layers.Conv2D(filters = filters1, kernel_size = (1,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = filters2, kernel_size = (1,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(l2rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def dropblock(x, keep_prob, block_size):
    _,w,h,c = x.shape.as_list()
    gamma = (1 - keep_prob) * w / (block_size * (w - block_size + 1))
    sampling_mask_shape = tf.stack([1, w - block_size + 1, 1, c])
    noise_dist = tf.compat.v1.distributions.Bernoulli(probs=gamma)
    mask = noise_dist.sample(sampling_mask_shape)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    pad_shape = [[0, 0], [tl, br], [0, 0], [0, 0]]
    mask = tf.pad(mask, pad_shape)
    mask = tf.nn.max_pool(mask, [1, block_size, 1, 1], [1, 1, 1, 1], 'SAME')
    mask = tf.cast(1 - mask,tf.float32)
    return tf.multiply(x,mask)

def channel_shuffle(x):
    g = 2
    b, h, w, c = x.shape
    x = tf.reshape(x, [-1, h, w, g, c // g])
    x = tf.transpose(x, perm = [0, 1, 2, 4, 3])
    x = tf.reverse(x,[-1])
    x = tf.reshape(x, [-1, h, w, c])
    return x

def UNet3plus_2c(input_shape):
    inputs = keras.Input(shape = input_shape)
    
    XEe1_1 = XabBlock_c(inputs, 400, 400, (3,1), 1, 1, 0.003, 0.9)
    XEe1_1 = channel_attention(XEe1_1)
    XEe1_1 = channel_shuffle(XEe1_1)
    
    XEe1_2 = XabBlock_c(XEe1_1, 200, 200, (3,1), 1, 1, 0.003, 0.9)
    XEe1_2 = channel_attention(XEe1_2)
    XEe1_2 = channel_shuffle(XEe1_2)
    
    XEe1_3 = XabBlock_c(XEe1_2, 100, 100, (3,1), 1, 1, 0.003, 0.9)
    XEe1_3 = channel_attention(XEe1_3)
    XEe1_3 = channel_shuffle(XEe1_3)

    XEe1 = XabBlock_c(XEe1_2, 128, 128, (3,1), 1, 1, 0.003, 0.9)
    XEe1 = channel_shuffle(XEe1)
    
    XEe2 = layers.MaxPooling2D(pool_size=(2, 1), strides=2)(XEe1)
    XEe2 = XabBlock_c(XEe2, 256, 256, (3,1), 1, 1, 0.004, 0.9)
    XEe2 = channel_shuffle(XEe2)

    XEe3 = layers.MaxPooling2D(pool_size=(2, 1), strides=2)(XEe2)
    XEe3 = XabBlock_c(XEe3, 512, 512, (3,1), 1, 1, 0.005, 0.9)
    XEe3 = channel_shuffle(XEe3)
    
    XEe4 = layers.MaxPooling2D(pool_size=(2, 1), strides=2)(XEe3)
    XEe4 = XabBlock_c(XEe4, 1024, 1024, (3,1), 1, 1, 0.006, 0.9)
    XEe4 = channel_shuffle(XEe4)
    
    XEe4_up = tf.image.resize(XEe4, [32,1], method='bicubic')
    XEe4_up = ConvBlock2(XEe4_up, 512, (3,1), 1, 0.01, 0.9)
    XEe2_PT_XDe3 = layers.MaxPooling2D(pool_size=(2, 1), strides=2)(XEe2)
    XEe2_PT_XDe3 = ConvBlock2(XEe2_PT_XDe3, 512, (3,1), 1, 0.01, 0.9)
    XEe1_PT_XDe3 = layers.MaxPooling2D(pool_size=(4, 1), strides=4)(XEe1)
    XEe1_PT_XDe3 = ConvBlock2(XEe1_PT_XDe3, 512, (3,1), 1, 0.01, 0.9)
    XDe3 = layers.concatenate([XEe4_up, XEe3, XEe2_PT_XDe3, XEe1_PT_XDe3], axis = 3)
    XDe3 = channel_shuffle(XDe3)
    XDe3 = XabBlock_s(XDe3, 512, 512, (3,1), 1, 1, 0.007, 0.9)
    XDe3 = channel_shuffle(XDe3)
    
    XDe3_up = tf.image.resize(XDe3, [64,1], method='bicubic')
    XDe3_up = ConvBlock2(XDe3_up, 256, (3,1), 1, 0.01, 0.9)
    XEe4_UP_XDe2 = tf.image.resize(XEe4, [64,1], method='bicubic')
    XEe4_UP_XDe2 = ConvBlock2(XEe4_UP_XDe2, 256, (3,1), 1, 0.01, 0.9)
    XEe1_PT_XDe2 = layers.MaxPooling2D(pool_size=(2, 1), strides=2)(XEe1)
    XEe1_PT_XDe2 = ConvBlock2(XEe1_PT_XDe2, 256, (3,1), 1, 0.01, 0.9)
    XDe2 = layers.concatenate([XDe3_up, XEe4_UP_XDe2, XEe2, XEe1_PT_XDe2], axis = 3)
    XDe2 = channel_shuffle(XDe2)
    XDe2 = XabBlock_s(XDe2, 256, 256, (3,1), 1, 1, 0.008, 0.9)
    XDe2 = channel_shuffle(XDe2)

    XDe2_up = tf.image.resize(XDe2, [128,1], method='bicubic')
    XDe2_up = ConvBlock2(XDe2_up, 128, (3,1), 1, 0.01, 0.9)
    XDe3_UP_XDe1 = tf.image.resize(XDe3, [128,1], method='bicubic')
    XDe3_UP_XDe1 = ConvBlock2(XDe3_UP_XDe1, 128, (3,1), 1, 0.01, 0.9)
    XEe4_UP_XDe1 = tf.image.resize(XEe4, [128,1], method='bicubic')
    XEe4_UP_XDe1 = ConvBlock2(XEe4_UP_XDe1, 128, (3,1), 1, 0.01, 0.9)
    XDe1 = layers.concatenate([XDe2_up, XDe3_UP_XDe1, XEe4_UP_XDe1, XEe1], axis = 3)
    XDe1 = channel_shuffle(XDe1)
    XDe1 = XabBlock_s(XDe1, 128, 128, (3,1), 1, 1, 0.009, 0.9)
    XDe1 = channel_shuffle(XDe1)
    
    output4 = tf.image.resize(XEe4, [128,1], method='bicubic')
    output4 = ConvBlock2(output4, 256, (3,1), 1, 0.01, 0.9)

    output3 = tf.image.resize(XDe3, [128,1], method='bicubic')
    output3 = ConvBlock2(output3, 128, (3,1), 1, 0.01, 0.9)

    output2 = tf.image.resize(XDe2, [128,1], method='bicubic')
    output2 = ConvBlock2(output2, 64, (3,1), 1, 0.01, 0.9)
    
    output1 = ConvBlock2(XDe1, 64, (3,1), 1, 0.01, 0.9)
    
    outputs = layers.concatenate([output4, output3, output2, output1], axis = 3)
    outputs = channel_shuffle(outputs)
    
    outputs = layers.Conv2D(filters = 100, kernel_size = (3,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.ReLU()(outputs)
    outputs = layers.Conv2D(filters = 40, kernel_size = (3,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.ReLU()(outputs)
    outputs = layers.Conv2D(filters = 15, kernel_size = (3,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.ReLU()(outputs)
    outputs = layers.Conv2D(filters = 5, kernel_size = (3,1), strides = 1, padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(outputs)
    
    model = Model(inputs, outputs)
    return model

def reader(data_path,batch_size):
    num = int((len([lists for lists in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, lists))]))/2)
    step = 0
    dfname = np.array(range(num))
    permutation = np.random.permutation(num)
    dfname = dfname[permutation]
    while True:
        X = np.zeros((batch_size,128,800))
        Y = np.zeros((batch_size,128,5))
        for db in range(batch_size):
            datafileX = data_path+'/data'+str(dfname[int(step*batch_size+db)])+'.npy'
            X[db,:,:] = np.load(datafileX)
            datafileY = data_path+'/label'+str(dfname[int(step*batch_size+db)])+'.npy'
            Y[db,:,:] = np.load(datafileY)
        
        X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1, Y.shape[2])
        
        X,Y = randomize(X,Y)
               
        yield X,Y
        step = (step+1)% (num//batch_size)
            
def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    dataset = dataset[permutation]
    labels = labels[permutation]
    return dataset, labels

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
    
def R_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2

def PW_loss(y_true, y_pred):
    weight_high = tf.cast(100, dtype=float32)
    weight_low = tf.cast(30, dtype=float32)
    threshold = tf.cast(0.001, dtype=float32)
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    mask = tf.greater(y_true, threshold)

    a = tf.square(tf.subtract(y_true, y_pred))
    loss = tf.reduce_mean(tf.where(mask, weight_high*a, weight_low*a)) 
    return loss


if __name__ == '__main__':
    train_path = u'C:/DeepEER2/data/23_04_18/train'
    validation_path = u'C:/DeepEER2/data/23_04_18/validation'  
    savepath = 'C:/DeepEER2/model_out/23_04_18/SepConv/47'
    mkdir(savepath)
    
    tf.keras.backend.clear_session()
    ops.reset_default_graph()

    model = UNet3plus_2c(input_shape=[128,1,800])
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.000, amsgrad=False),
                 loss= PW_loss,
                 metrics = [R_squared])

    batch_size = 128
    epochs =150
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    history = model.fit_generator(generator = reader(train_path,batch_size),
                        epochs = epochs,
                        steps_per_epoch = 80000//batch_size,
                        validation_data = reader(validation_path,batch_size),
                        validation_steps = 10000//batch_size,
                        callbacks = [callback])
    
    fig = plt.figure()
    plt.ylabel('R2',size=15)
    plt.xlabel('Epoch',size=15)       
    plt.plot(history.history['R_squared'],label = 'R2_training')
    plt.plot(history.history['val_R_squared'],label = 'R2_validation')
    plt.legend()
    plt.savefig(savepath+'/ACC.jpg')
    
    fig = plt.figure()
    plt.xlabel('Epoch',size=15)    
    plt.ylabel('Loss',size=15)
    plt.plot(history.history['loss'],label = 'Loss_training')
    plt.plot(history.history['val_loss'],label = 'Loss_validation')
    plt.legend()
    plt.show()
    plt.savefig(savepath+'/LOSS.jpg')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)       
    ax1.set_ylabel('R2',size=15,color='g')
    ax1.set_xlabel('Epoch',size=15)       
    plt.plot(history.history['R_squared'],color='g')
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Loss',size=15,color='r')
    plt.plot(history.history['loss'],'r')
    plt.show()
    plt.savefig(savepath+'/train.jpg')
        
    model.save(savepath+'/model.h5')


