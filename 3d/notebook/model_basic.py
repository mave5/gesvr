# deconv net
import numpy as np
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout, LSTM
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.models import Sequential

def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# model
def model(param_model):

    c=param_model['img_ch']
    h=param_model['img_rows']
    w=param_model['img_cols']
    lr=param_model['lr']
    

    inputs = Input((c, h, w))

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    model = Model(input=inputs, output=pool1)
    print 'output pool1: ', model.output_shape
    
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    model = Model(input=inputs, output=pool2)
    print 'output pool2: ', model.output_shape

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    model = Model(input=inputs, output=pool3)
    print 'output pool3: ', model.output_shape

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    model = Model(input=inputs, output=conv4)
    print 'output conv4: ', model.output_shape

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)
    model = Model(input=inputs, output=conv5)
    print 'output conv4: ', model.output_shape


    conv10 = Convolution2D(c, 1, 1, activation='sigmoid', border_mode='same')(conv8)




    model = Model(input=inputs, output=conv10)
    print 'output: ', model.output_shape


    #load previous weights
    #if weights_path:
    #    model.load_weights(weights_path)

    #model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr), Nestrov=True, loss='mean_squared_error', metrics=loss)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr))

    return model
