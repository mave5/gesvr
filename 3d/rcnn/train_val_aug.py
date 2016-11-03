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
from model_dice2 import *
#from model_bn import *
#from skimage import exposure

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
def preprocess(X,Y,img_hw,crop,imr):
    print ('_' *50)
    start_time=time.time()
    
    # crop h*w
    if crop==1:
        H,W=X.shape[2:]
        hc=(H-img_hw[0])/2
        wc=(W-img_hw[1])/2
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

# sample
def disp_img_mask(img,mask):
    n1=np.random.randint(img.shape[0])
    I1=img[n1,:]
    print I1.shape
    M1=mask[n1,:]
    print M1.shape
    for k in range(10):
        plt.subplot(2,5,k+1)
        imgmask=image_with_mask(I1[k,:],M1[k,:])
        plt.imshow(imgmask)
        plt.title('subject: %s, slice: %s' %(n1,k))
        


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

#%%
#==============================================================================
# main
#==============================================================================

# load train data
X_train,Y_train=load_data(path2nfolds + 'trainfold'+str(foldnm)+'.npz')
# load test data
X_test,Y_test=load_data(path2nfolds + 'testfold'+str(foldnm)+'.npz')


# preprocess train data
img_hw= np.array([176,176])
crop_ena=1
imr=1
X_train_r,Y_train_r=preprocess(X_train,Y_train,img_hw,crop_ena,imr)
# preprocess test data
X_test_r,Y_test_r=preprocess(X_test,Y_test,img_hw,crop_ena,imr)

# display sample image
disp_img_mask(X_train_r,Y_train_r)

# normalization
print 'normalization ...'
X_train_r = X_train_r.astype('float32')
#Y_train_r = Y_train_r.astype('float32')
mean = np.mean(X_train_r)  # mean
sigma = np.std(X_train_r)  # std
#X_train_r -= mean
#X_train_r /= sigma
#Y_train_r /= 255.  # scale masks to [0, 1]
print ('X_r size: ', X_train_r.shape)
print ('Y_r size: ', Y_train_r.shape)
print ('Min and Max  X_r: ', np.min(X_train_r), np.max(X_train_r))
print ('Min and Max  Y_r: ', np.min(Y_train_r), np.max(Y_train_r))

X_test_r = X_test_r.astype('float32')
#Y_test_r = Y_test_r.astype('float32')
#X_test_r -= mean
#X_test_r /= sigma
#Y_test_r /= 255.  # scale masks to [0, 1]
print ('X_r size: ', X_test_r.shape)
print ('Y_r size: ', Y_test_r.shape)
print ('Min and Max  X_r: ', np.min(X_test_r), np.max(X_test_r))
print ('Min and Max  Y_r: ', np.min(Y_test_r), np.max(Y_test_r))

# build the model
#weights_path='./weights/weights-improvement-44--0.76.hdf5'
weights_path=None
c=X_train_r.shape[1]
h=X_train_r.shape[2]
w=X_train_r.shape[3]
lr=1e-5
model = model(weights_path,c,h,w,lr)

# checkpoint
# create numpy folder if does no exist
path2weights='./output/weights/'
if  not os.path.exists(path2weights):
    os.makedirs(path2weights)
    print 'weights folder created'
filepath=path2weights+str(foldnm)+"-aug_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only='True',mode='min')

# Fit the model
start_time=time.time()
for epoch in range(150):
    seed = np.random.randint(0, 999999)
    print '-'*50
    print 'epoch: %s' %epoch
    X_train_aug,Y_train_aug=random_crop(X_train,Y_train,img_hw,seed)
    X_test_aug,Y_test_aug=random_crop(X_test,Y_test,img_hw,seed)
    model.fit(X_train_aug, Y_train_aug, validation_data=(X_test_aug, Y_test_aug), nb_epoch=1, batch_size=1,verbose=1,shuffle=True,callbacks=[checkpoint])

print 'model was trained!'
elapsed_time=(time.time()-start_time)/60
print 'elapsed time: %d  mins' %elapsed_time

#%%
# predict
print 'please wait to test ...'
#weights_path=path2weights+'1-aug_weights-improvement-00-0.07.hdf5'
if os.path.isfile( weights_path):
    model.load_weights(weights_path)
    print 'weights loaded!'

print 'please wait to test ...'
Y_pred=model.predict(X_test_r)
diceMetric=calc_dice(Y_test_r,Y_pred)
print 'Dice Metric: %0.2f' %np.mean(diceMetric)
# display sample image
disp_img_mask(X_test_r,Y_pred>.5)
disp_img_mask(X_test_r,Y_test_r>.5)
