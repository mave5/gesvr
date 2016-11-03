
# clear worksapce
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from numpy.random import permutation
import time
import cv2
from funcs import model_basic as model_basic
#from model_skip import *
import os
print 'All libs successfully loaded!'
from funcs.image import ImageDataGenerator
from funcs import utils

# current working directory
print os.getcwd() 

path2set="../dcom/TrainingSet/"
path2numpy = path2set+"numpy/"
path2nfolds=path2numpy+'nfolds/'
foldnm=1

# fix random seed for reproducibility
seed = 2016
np.random.seed(seed)

#%%

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

# data augmentation
def data_aug(X,Y,img_hw):
    print ('_' *50)
    print 'please wait ...'
    start_time=time.time()
    H,W=X.shape[:2]
    hc=(H-img_hw[0])/2
    wc=(W-img_hw[1])/2
    # center crop    
    Xc=X[hc:H-hc,wc:W-wc,:]
    Yc=Y[hc:H-hc,wc:W-wc,:]
    # random crops    
    for k1 in range(X.shape[2]):
        for k2 in range(10):
            hcr=np.random.randint(hc)
            wcr=np.random.randint(wc)
            tmpx=X[hcr:hcr+img_hw[0],wcr:wcr+img_hw[1],k1]
            tmpx=np.expand_dims(tmpx,axis=2)
            Xc=np.append(Xc,tmpx,axis=2)
            tmpy=Y[hcr:hcr+img_hw[0],wcr:wcr+img_hw[1],k1]
            tmpy=np.expand_dims(tmpy,axis=2)
            Yc=np.append(Yc, tmpy,axis=2)
    elapsed_time=time.time()-start_time
    print 'Elapsed time: %d seconds' % elapsed_time
    print ('_' *50)
    return Xc,Yc        


# random crop
def random_crop(X,Y,img_hw):
    print ('_' *50)
    print 'please wait ...'
    start_time=time.time()
    H,W=X.shape[:2]
    hc=(H-img_hw[0])/2
    wc=(W-img_hw[1])/2
    Xc=np.zeros((img_hw[0],img_hw[1],X.shape[2]),dtype=X.dtype)
    Yc=np.zeros((img_hw[0],img_hw[1],Y.shape[2]),dtype=Y.dtype)
    # random crops    
    for k1 in range(X.shape[2]):
        hcr=np.random.randint(hc)
        wcr=np.random.randint(wc)
        Xc[:,:,k1]=X[hcr:hcr+img_hw[0],wcr:wcr+img_hw[1],k1]
        Yc[:,:,k1]=Y[hcr:hcr+img_hw[0],wcr:wcr+img_hw[1],k1]
    elapsed_time=time.time()-start_time
    #Xc=Xc+np.round(0.05*np.max(Xc)*np.random.randn(Xc.shape[0],Xc.shape[1],Xc.shape[2]))
    print 'Elapsed time: %d seconds' % elapsed_time
    print ('_' *50)
    return Xc,Yc        

    
# preprocess, resize data
def preprocess(X,Y,params):
    
    X = X.astype('float32')
    
    # params
    img_hw=params['row'],params['col']
    crop=params['crop']    
    imr=params['imr']
    norm_ena=params['normalization']
    crop_type=params['crop_type']
    
    print ('_' *50)
    start_time=time.time()
    # crop h*w
    if (crop==1) and (crop_type=='center'):
        H,W=X.shape[:2]
        hc=(H-img_hw[0])/2
        wc=(W-img_hw[1])/2
        X=X[hc:H-hc,wc:W-wc,:]
        Y = Y[hc:H-hc,wc:W-wc,:]
    elif (crop==1) and (crop_type=='random'):
        X,Y=random_crop(X,Y,img_hw)            
            


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

    # normalization
    if norm_ena==True:
        for k in range(X_r.shape[0]):
            mean = np.mean(X_r[k,0])  # mean       
            sigma = np.std(X_r[k,0])  # std
            X_r[k,:,:,:] = X_r[k,:,:,:]-mean
            X_r[k,:,:,:] = X_r[k,:,:,:]/ sigma

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

def image_with_mask(img, mask):
    maximg=np.max(img)    
    if np.max(mask)==1:
        mask=mask*255
        mask=np.asarray(mask,dtype='uint8')
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = maximg  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    img_color=img_color/float(np.max(img))
    return img_color

# sample
def disp_img_mask(img,mask,r=1,c=1,d=0):
    N=r*c    
    n1=np.random.randint(img.shape[d],size=N)    
    if d==0:    
        I1=img[n1,0]
        M1=mask[n1,0]
    else:
        I1=img[:,:,n1]
        M1=mask[:,:,n1]
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M1[k])
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
        

# calcualte dice
def calc_dice(X,Y,d=0):
    dice=np.zeros([X.shape[d],1])
    for k in range(X.shape[d]):
        if d==0:        
            x=X[k,0] >.5
            y =Y[k, 0]>.5
        else:
            x=X[:,:,k] >.5
            y =Y[:,:,k]>.5

        # number of ones
        intersectXY=np.sum((x&y==1))
        unionXY=np.sum(x)+np.sum(y)

        if unionXY!=0:
            dice[k]=2* intersectXY/(unionXY*1.0)
            #print 'dice is: %0.2f' %dice[k]
        else:
            dice[k]=1
            #print 'dice is: %0.2f' % dice[k]
        print 'processing %d, dice= %0.2f' %(k,dice[k])
    return np.mean(dice)

# remapt to irgiran size
def remap(Xin,Xorig):
    H,W,N=Xorig.shape # original size
    n,c,h,w=Xin.shape # croped size
    if n!=N:
        print 'different image numbers in the arrays'

    hc=(H-h)/2
    wc=(W-w)/2
    Xin=np.squeeze(Xin)
    Xin=np.transpose(Xin,(1,2,0))
    Xout=np.zeros(Xorig.shape,dtype=Xorig.dtype)
    Xout[hc:hc+h,wc:wc+w,:]=Xin
    return Xout
    
# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=75,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.01,
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
    
# histogram equalization
def hist_equ(X):
    X=np.asarray(X,dtype='uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    Xeq=np.zeros(X.shape,dtype=X.dtype)
    for k in range(X.shape[2]):
        Xeq[:,:,k] = clahe.apply(X[:,:,k])   
    return Xeq        
    
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


params_preprocess={
    'row' : 176,
    'col' : 176,
    'imr' : 1,
    'crop': True,
    'crop_type': 'random',
    'normalization':True,
    }

# pre processing 
X_train_r,Y_train_r=preprocess(X_train,Y_train,params_preprocess)

# preprocess test data
X_test_r,Y_test_r=preprocess(X_test,Y_test,params_preprocess)

# display sample image
disp_img_mask(X_train_r,Y_train_r)

#%%

# training params
params_train={
        'row': X_train_r.shape[2],
        'col': X_train_r.shape[3],           
        'weights': None,        
        'lr': 1e-4,
        'optimizer': 'Adam',
        'loss': 'dice_coef',
        }

# build the model
model = model_basic.model(params_train)
model.summary()

with open('model.json', 'w') as f:
  f.write(model.to_json())
  
# checkpoint
# create numpy folder if does no exist
path2weights='./output/weights'
if  not os.path.exists('./output/weights'):
    os.makedirs(path2weights)
    print 'weights folder created'
filepath=path2weights+"/fold"+str(foldnm)+"-new-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only='True',mode='min')


# Fit the model
start_time=time.time()
scores=[]
for e in range(1500):
    print 'epoch: %s' %e
    seed = np.random.randint(0, 999999)
    X_train_r,Y_train_r=preprocess(X_train,Y_train,params_preprocess)
    X_train_aug,Y_train_aug=iterate_minibatches( X_train_r, Y_train_r , X_train_r.shape[0], shuffle=True)
    model.fit(X_train_aug, Y_train_aug, validation_data=(X_test_r, Y_test_r), nb_epoch=1, batch_size=8,verbose=1,shuffle=True,callbacks=[checkpoint])
    score=model.evaluate(X_test_r, Y_test_r)    
    scores=np.append(scores,score[1])
    

plt.plot(scores)    
print 'model was trained!'
elapsed_time=(time.time()-start_time)/60
print 'elapsed time: %d  mins' %elapsed_time

#%%
# predict
print 'please wait to test ...'
weights_path='./output/weights/fold'+str(foldnm)+"-new-weights.hdf5"
if os.path.isfile( weights_path):
    model.load_weights(weights_path)
    print 'weights loaded!'

# predic training data
print '-' *50
print 'please wait to test ...'
Y_train_pred=model.predict(X_train_r)
DM_train=calc_dice(Y_train_r,Y_train_pred)
print 'Train data Dice Metric: %0.2f' %DM_train

# predict
print '-' *50
print 'please wait to test ...'
Y_test_pred=model.predict(X_test_r)
DM_test=calc_dice(Y_test_r,Y_test_pred)
print 'Dice Metric: %0.2f' %DM_test

#%%
