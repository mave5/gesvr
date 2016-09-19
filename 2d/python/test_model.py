import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from numpy.random import permutation
import time
import cv2
from model_dice2 import *
import os
print 'All libs successfully loaded!'

print os.getcwd() 

path2set="../dcom/TrainingSet/"
path2numpy = path2set+"numpy/"
path2nfolds=path2numpy+'nfolds/'
foldnm=4


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
def preprocess(X,Y,img_hw,crop,imr=1):
    print ('_' *50)
    start_time=time.time()
    # crop h*w
    if crop==1:
        H,W=X.shape[:2]
        hc=(H-img_hw[0])/2
        wc=(W-img_hw[1])/2
        X=X[hc:H-hc,wc:W-wc,:]
        Y = Y[hc:H-hc,wc:W-wc,:]
    print ('X shape: ', X.shape)
    print ('Y shape: ',Y.shape)

    # convert from  H*W*N to N*C*H*W
    X=np.expand_dims(X,axis=0)
    X=np.transpose(X,(3,0,1,2))
    Y=np.expand_dims(Y,axis=0)
    Y=np.transpose(Y,(3,0,1,2))
    
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



def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))

def image_with_mask(img, mask,color='R'):
    maximg=np.max(img)    
    if np.max(mask)==1:
        mask=mask*255
        mask=np.asarray(mask,dtype='uint8')
    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)<3:     
        img_color = grays_to_RGB(img)
    else:
        img_color=img
    if np.sum(mask)>0:    
        mask_edges = cv2.Canny(mask, 100, 200) > 0
    else:
        mask_edges=mask    
    #print np.sum(mask_edges)
    if color=='R':    
        img_color[mask_edges, 0] = maximg  # set channel 0 to bright red, green & blue channels to 0
        img_color[mask_edges, 1] = 0
        img_color[mask_edges, 2] = 0
    elif color=='G':    
        img_color[mask_edges, 0] = 0  # set channel 0 to bright red, green & blue channels to 0
        img_color[mask_edges, 1] = maximg
        img_color[mask_edges, 2] = 0
    else:     
        img_color[mask_edges, 0] = 0  # set channel 0 to bright red, green & blue channels to 0
        img_color[mask_edges, 1] = 0
        img_color[mask_edges, 2] = maximg
    img_color=img_color/float(np.max(img))
    return img_color

# sample
def disp_img_mask(img,mask):
    n1=np.random.randint(img.shape[0])
    I1=img[n1,0]
    M1=mask[n1,0]
    imgmask=image_with_mask(I1,M1)
    plt.imshow(imgmask)


# calcualte dice
def calc_dice(X,Y):
    dice=np.zeros([len(X),1])
    for k in range(len(Y)):
        x=X[k,0] >.5
        y =Y[k, 0]>.5

        # number of ones
        intersectXY=np.sum((x&y==1))
        unionXY=np.sum(x)+np.sum(y)

        if unionXY!=0:
            dice[k]=2* intersectXY/(unionXY*1.0)
            #print 'dice is: %0.2f' %dice[k]
        else:
            dice[k]=1
            #print 'dice is: %0.2f' % dice[k]
        #print 'processing %d, dice= %0.2f' %(k,dice[k])
    return np.mean(dice)
    
# clean masks
def clean_mask(Y_pred,Y_true):
    # ground truth min area
    s1 = np.sum(np.sum(Y_true, axis=3), axis=2)  # sum over each mask
    nz1 = np.where(s1 > 0)  # find non zeros masks
    minarea = np.min(s1[nz1])  # min area
    maxarea=np.max(s1[nz1])    # max area
    meanarea=np.mean(s1[nz1]) # average area
    print 'min area: %f, max area: %f,  and average area: %f' % (minarea, maxarea, meanarea)

    # predictions
    s2 = np.sum(np.sum(Y_pred, axis=2), axis=2)  # sum over each mask
    nz2 = np.where(s2 > minarea)  # find indices greater than min area
    Y_clean=np.zeros(Y_true.shape,dtype='float32')
    Y_clean[nz2[0],:,:,:]=Y_pred[nz2[0],:,:,:]
    print 'masks were cleaned.'
    return Y_clean
    
#######################################################################################################################
#######################################################################################################################

# load train data
X_train,Y_train=load_data(path2nfolds + 'trainfold'+str(foldnm)+'.npz')
# load test data
X_test,Y_test=load_data(path2nfolds + 'testfold'+str(foldnm)+'.npz')


# preprocess train data
img_hw= np.array([176,176])
crop_ena=1
X_train_r,Y_train_r=preprocess(X_train,Y_train,img_hw,crop_ena)
# preprocess test data
X_test_r,Y_test_r=preprocess(X_test,Y_test,img_hw,crop_ena)

# display sample image
#disp_img_mask(X_train_r,Y_train_r)

# normalization
print 'normalization ...'
X_train_r = X_train_r.astype('float32')
Y_train_r = Y_train_r.astype('float32')
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
Y_test_r = Y_test_r.astype('float32')
#X_test_r -= mean
#X_test_r /= sigma
#Y_test_r /= 255.  # scale masks to [0, 1]
print ('X_r size: ', X_test_r.shape)
print ('Y_r size: ', Y_test_r.shape)
print ('Min and Max  X_r: ', np.min(X_test_r), np.max(X_test_r))
print ('Min and Max  Y_r: ', np.min(Y_test_r), np.max(Y_test_r))

# build the model
weights_path='./weights/fold4-weights-improvement-132--0.77.hdf5'
#weights_path=None
h=X_train_r.shape[2]
w=X_train_r.shape[3]
lr=1e-5
model = model(weights_path,h,w,lr)


# predict
print 'please wait to test ...'
Y_pred=model.predict(X_test_r)
plt.imshow(Y_pred[40,0]>0.5)
diceMetric=calc_dice(Y_test_r,Y_pred)
print 'Dice Metric: %0.2f' %diceMetric
# display sample image
disp_img_mask(X_test_r+mean,Y_pred>.5)


# clean mask
Y_clean=clean_mask(Y_pred,Y_test_r)
dice_clean=calc_dice(Y_test_r,Y_clean)
print 'dice clean: %0.2f' %dice_clean
# display sample image
disp_img_mask(X_test_r+mean,Y_clean>.5)


# save results as tif
path2results='./results/fold'
for k in range(Y_pred.shape[0]):
    imgmask=image_with_mask(X_test_r[k,0]+mean,Y_pred[k,0]>0.5)
    imgmask=image_with_mask(imgmask,Y_test_r[k,0]>0.5,'G')
    plt.imsave(path2results+str(foldnm)+'_'+str(k)+'.tif',imgmask,cmap=plt.cm.gray)
    #plt.imsave(path2results+str(foldnm)+'_'+str(k)+'.tif',X_test[:,:,k],cmap=plt.cm.gray)
    #plt.imsave(path2results+str(foldnm)+'_'+str(k)+'_mask.tif',Y_pred[k,0]>0.5,cmap=plt.cm.gray)

