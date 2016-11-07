#==============================================================================
# libs
#==============================================================================

import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from numpy.random import permutation
import time
import cv2
import random
#from model_dice import *
from model_basic import *
import utils
#from model_bn import *
#from skimage import exposure
#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator

import os
print 'All libs successfully loaded!'

print os.getcwd() 

path2set="../dcom/TrainingSet/"
path2numpy = path2set+"numpy/"
path2nfolds=path2numpy+'nfolds/'
foldnm=1

#%%
#==============================================================================
# functions
#==============================================================================

# fix random seed for reproducibility
seed = 2016
np.random.seed(seed)

# load data
def load_data(path):
    print ('_' *50)
    print 'please wait to load data ...'
    start_time=time.time()
    tmp = np.load(path)
    X=tmp['X']
    Y=tmp['Y'] 
    print ('X shape: ', X.shape)
    print ('Y shape: ',Y.shape)
    print ('Min and Max X: ', np.min(X), np.max(X))
    print ('Min and Max Y: ', np.min(Y), np.max(Y))
    elapsed_time=time.time()-start_time
    print 'Elapsed time: %d seconds' % elapsed_time
    print ('_' *50)
    return X,Y

# preprocess, resize data
def preprocess(X,Y,param_prep):
    
    h=param_prep['img_rows']
    w=param_prep['img_cols']    
    crop=param_prep['crop']
    imr=param_prep['resize_factor']
    
    print ('_' *50)
    start_time=time.time()
    Y=np.asarray(Y,dtype='uint8')
    # crop h*w
    if crop==1:
        H,W=X.shape[2:]
        hc=(H-h)/2
        wc=(W-w)/2
        X=X[:,:,hc:H-hc,wc:W-wc]
        Y = Y[:,:,hc:H-hc,wc:W-wc]
    print ('X shape: ', X.shape)
    print ('Y shape: ',Y.shape)

    print 'please wait to resize images ...'
    if imr<1:
        img_h=int(X.shape[2]*imr) # rows
        img_w=int(X.shape[3]*imr) # columns
        X_r=np.zeros([X.shape[0],1,img_h,img_w],dtype=X.dtype)
        Y_r=np.zeros([Y.shape[0],1,img_h,img_w],dtype=Y.dtype)
        for k1 in range(len(X)):
            X_r[k1, 0] = cv2.resize(X[k1, 0], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            Y_r[k1, 0] = cv2.resize(Y[k1, 0], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    else:
        X_r=X
        Y_r=Y

    print ('X_r size: ', X_r.shape)
    print ('Y_r size: ',Y_r.shape)
    print ('Min and Max  X_r: ', np.min(X_r), np.max(X_r))
    print ('Min and Max  Y_r: ',  np.min(Y_r),  np.max(Y_r))

    elapsed_time=time.time()-start_time
    print 'Elapsed time: %d seconds' % elapsed_time
    print ('_' *50)
    return X_r,Y_r


# random crop
def random_crop(X,Y,img_hw,seed):
 # reproducible data augmentation
    random.seed(seed)
    np.random.seed(seed)    
    
    print ('_' *50)
    print 'please wait ...'
    start_time=time.time()
    H,W=X.shape[2:]
    hc=(H-img_hw[0])/2
    wc=(W-img_hw[1])/2
    Xc=np.zeros((X.shape[0],X.shape[1],img_hw[0],img_hw[1]),dtype=X.dtype)
    Yc=np.zeros((Y.shape[0],Y.shape[1],img_hw[0],img_hw[1]),dtype=Y.dtype)
    # random crops    
    for k1 in range(X.shape[0]):
        hcr=np.random.randint(hc)
        wcr=np.random.randint(wc)
        Xc[k1,:,:,:]=X[k1,:,hcr:hcr+img_hw[0],wcr:wcr+img_hw[1]]
        Yc[k1,:,:,:]=Y[k1,:,hcr:hcr+img_hw[0],wcr:wcr+img_hw[1]]
    elapsed_time=time.time()-start_time
    print 'Elapsed time: %d seconds' % elapsed_time
    print ('_' *50)
    return Xc,Yc        


def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))

def image_with_mask(img, mask):
    maximg=np.max(img)
    mask=np.asarray(mask,dtype='uint8')
    if np.max(mask)<=1:
        mask=mask*255
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0
    #print np.sum(mask_edges)
    img_color[mask_edges, 0] = maximg  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    img_color=img_color/float(np.max(img))
    return img_color

# 4D array image display
def disp_img_mask(img,mask=None,ind=None):
    # img and mask are 4d arrays, N*C*H*W
    
    # check for random dispaly or based on input
    if ind==None:
        n1=np.random.randint(img.shape[0])
    else:
        n1=ind
        
    I1=img[n1,:]
    print I1.shape
    
    if mask==None:
        M1=np.zeros(I1.shape,dtype='uint8')
    else:
        M1=mask[n1,:]
    print M1.shape
    
    r,c=2,5
    for k in range(r*c):
        plt.subplot(r,c,k+1)
        imgmask=image_with_mask(I1[k,:],M1[k,:])
        plt.imshow(imgmask)
        plt.title('s: %s, maxI: %s' %(n1,np.max(I1[k,:])))
        


# calcualte dice
def calc_dice(X,Y):
    dice=np.zeros([X.shape[0],X.shape[1]])
    for k in range(Y.shape[0]):
        for k2 in range(Y.shape[1]):
            x=X[k,k2] >.5
            y =Y[k, k2]>.5

            # number of ones
            intersectXY=np.sum((x&y==1))
            unionXY=np.sum(x)+np.sum(y)

            if unionXY!=0:
                dice[k,k2]=2* intersectXY/(unionXY*1.0)
                #print 'dice is: %0.2f' %dice[k]
            else:
                dice[k,k2]=1
                #print 'dice is: %0.2f' % dice[k]
            print 'processing subject: %d, slice: %d, dice= %0.2f' %(k, k2, dice[k,k2])
    return dice

# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th')
        
def iterate_minibatches(inputs1 , targets,  batchsize, shuffle=True, augment=True):
    assert len(inputs1) == len(targets)
 
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        x = inputs1[excerpt]
        y = targets[excerpt] 
        for  xxt,yyt in datagen.flow(x, y , batch_size=x.shape[0]):
            x = xxt.astype(np.float32) 
            y = yyt 
            break

    #yield x, np.array(y, dtype=np.uint8)         
    return x, np.array(y, dtype=np.uint8)         

#%%
#==============================================================================
# main
#==============================================================================

# Direct the output to a log file and to screen
loggerFileName = './output/logs'+'/log.txt'
utils.initialize_logger(loggerFileName)

# load train data
X_train,Y_train=load_data(path2nfolds + 'trainfold'+str(foldnm)+'.npz')

# load test data
X_test,Y_test=load_data(path2nfolds + 'testfold'+str(foldnm)+'.npz')

#%%
param_prep={
    'img_rows': 192,
    'img_cols': 192,
    'crop'    : True,
    'resize_factor': 1
}

# preprocess train data
X_train_r,Y_train_r=preprocess(X_train,Y_train,param_prep)

# preprocess test data
X_test_r,Y_test_r=preprocess(X_test,Y_test,param_prep)

# display sample image
disp_img_mask(X_train_r,Y_train_r)

# normalization
print 'normalization ...'
X_train_r = X_train_r.astype('float32')
#Y_train_r = Y_train_r.astype('float32')
mean = np.mean(X_train_r)  # mean
sigma = np.std(X_train_r)  # std
X_train_r -= mean
X_train_r /= sigma
#Y_train_r /= 255.  # scale masks to [0, 1]
print ('X_r size: ', X_train_r.shape)
print ('Y_r size: ', Y_train_r.shape)
print ('Min and Max  X_r: ', np.min(X_train_r), np.max(X_train_r))
print ('Min and Max  Y_r: ', np.min(Y_train_r), np.max(Y_train_r))

X_test_r = X_test_r.astype('float32')
#Y_test_r = Y_test_r.astype('float32')
X_test_r -= mean
X_test_r /= sigma
#Y_test_r /= 255.  # scale masks to [0, 1]
print ('X_r size: ', X_test_r.shape)
print ('Y_r size: ', Y_test_r.shape)
print ('Min and Max  X_r: ', np.min(X_test_r), np.max(X_test_r))
print ('Min and Max  Y_r: ', np.min(Y_test_r), np.max(Y_test_r))

# build the model
param_model={
    'img_ch': X_train_r.shape[1],
    'img_rows': X_train_r.shape[2],
    'img_cols': X_train_r.shape[3],
    'lr' : 1e-3
    }
model = model(param_model)

# load pre-trained models
#weights_path='./weights/weights-improvement-44--0.76.hdf5'
weights_path='./output/weights/fold'+str(foldnm)+'-aug_weights.hdf5'
if os.path.isfile( weights_path):
    #model.load_weights(weights_path)
    print 'weights loaded!'


# checkpoint
# create numpy folder if does no exist
if  not os.path.exists('./output/weights'):
    os.makedirs('./output/weights')
    print 'weights folder created'
#filepath="./output/weights/fold"+str(foldnm)+"-aug_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
filepath="./output/weights/fold"+str(foldnm)+"-aug_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only='True',mode='min')


# Fit the model
# number of epochs
nb_epoch=500
start_time=time.time()
for epoch in range(nb_epoch):
    print '-'*50
    print 'epoch: %s' %epoch
    #seed = np.random.randint(0, 999999)
    # augment validation
    #X_test_aug,Y_test_aug=iterate_minibatches( X_test_r, Y_test_r , X_test_r.shape[0], shuffle=False)
    # augment training
    X_train_aug,Y_train_aug=iterate_minibatches( X_train_r, Y_train_r , X_train_r.shape[0], shuffle=True)
    #disp_img_mask(X_train_aug,Y_train_aug)
    model.fit(X_train_aug, Y_train_aug, validation_data=(X_test_r, Y_test_r), nb_epoch=1, batch_size=1,verbose=0,shuffle=True,callbacks=[checkpoint])


print 'model was trained!'
elapsed_time=(time.time()-start_time)/60
print 'elapsed time: %d  mins' %elapsed_time

#%%
# predict
print 'please wait to test ...'
weights_path='./output/weights/fold'+str(foldnm)+'-aug_weights.hdf5'
if os.path.isfile( weights_path):
    model.load_weights(weights_path)
    print 'weights loaded!'

score_train=model.evaluate(X_train_aug,Y_train_aug)
print 'score train: ', score_train

score_test=model.evaluate(X_test_r,Y_test_r)
print 'score train: ', score_test

Y_pred1=model.predict(X_train_aug)
diceMetric=calc_dice(Y_train_aug,Y_pred1>.5)
print 'Dice Metric: %0.2f' %np.mean(diceMetric)

Y_pred=model.predict(X_test_r)
diceMetric=calc_dice(Y_test_r,Y_pred>.5)
print 'Dice Metric: %0.2f' %np.mean(diceMetric)
# display sample image
disp_img_mask(X_test_r,Y_pred>.5)
#disp_img_mask(X_test_aug,Y_test_aug>.5)
#disp_img_mask(X_train_aug,Y_train_aug>.5)

#%%

