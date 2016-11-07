import numpy as np
#from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from numpy.random import permutation
import time
import cv2
#from model_dice2 import *
from model_skip import *
import os
print 'All libs successfully loaded!'

print os.getcwd() 

#path2set="../dcom/TrainingSet/"
dataset='Test1Set/'
path2set="../dcom/"+dataset
path2numpy = path2set+"numpy/"


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

def postprocess(Y_pred):
    for k1 in range(Y_pred.shape[0]):
        # find contours
        BW=np.asarray(Y_pred[k1,0]>0.5,dtype='uint8')
        cs,_ = cv2.findContours( BW.astype('uint8'), mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE )
        for k2 in range(len(cs)):    
            c=cs[k2]
            m = cv2.moments(c)
            #Area= m['m00']
            #Perimeter= cv2.arcLength(c,True)
            # centroid    = m10/m00, m01/m00 (x,y)
            #Centroid= ( m['m10']/m['m00'],m['m01']/m['m00'] )

            # set up the 'FilledImage' bit of regionprops.
            filledI = np.zeros(BW.shape[0:2]).astype('uint8')
            cv2.drawContours( filledI, cs, k, color=255, thickness=-1 )
            

    
#######################################################################################################################
#######################################################################################################################

# list of numpy files
npflist=os.listdir(path2numpy)
Nf=len(npflist)
print 'total number of files: %d' %Nf

# crop size
crop_ena=1
img_hw= np.array([176,176])
weights_path1='./weights/skip_fold1-aug-weights-improvement-00--0.79.hdf5'
weights_path2='./weights/skip_fold2-aug-weights-improvement-00--0.81.hdf5'
weights_path3='./weights/skip_fold3-aug-weights-improvement-00--0.85.hdf5'
weights_path4='./weights/skip_fold4-aug-weights-improvement-00--0.83.hdf5'
#weights_path1='./weights/fold1-aug-weights-improvement-00--0.79.hdf5'
#weights_path2='./weights/fold2-aug-weights-improvement-136--0.80.hdf5'
#weights_path3='./weights/fold3-aug-weights-improvement-00--0.84.hdf5'
#weights_path4='./weights/fold4-aug-weights-improvement-00--0.78.hdf5'

# build the model   
lr=1e-4
model1 = model(weights_path1,img_hw[0],img_hw[1],lr)
model2 = model(weights_path2,img_hw[0],img_hw[1],lr)
model3 = model(weights_path3,img_hw[0],img_hw[1],lr)
model4 = model(weights_path4,img_hw[0],img_hw[1],lr)

#initialize dice
diceMetric1=np.zeros([len(npflist),1])
diceMetric2=np.zeros([len(npflist),1])
diceMetric3=np.zeros([len(npflist),1])
diceMetric4=np.zeros([len(npflist),1])
DM_ens=np.zeros([len(npflist),1])
patientlist=np.zeros([len(npflist),1],dtype='uint8')
DM1=np.zeros([len(npflist),1])
DM2=np.zeros([len(npflist),1])
DM3=np.zeros([len(npflist),1])
DM4=np.zeros([len(npflist),1])
DM_ens_orig=np.zeros([len(npflist),1])

pidk=0
for npfnm in npflist:
    # patient id
    patientID=int(npfnm[1:-4])
    print 'patient ID:  %s' %patientID
    patientlist[pidk]=patientID
    
    # load test data
    X_test,Y_test=load_data(path2numpy + npfnm)
    # preprocess test data
    X_test_r,Y_test_r=preprocess(X_test,Y_test ,img_hw,crop_ena)
    # display sample image
    #disp_img_mask(X_train_r,Y_train_r)

    # predict
    print 'please wait to test ...'
    Y_pred1=model1.predict(X_test_r)
    Y_pred2=model2.predict(X_test_r)
    Y_pred3=model3.predict(X_test_r)  
    Y_pred4=model4.predict(X_test_r)
    Y_pred=np.mean([Y_pred1,Y_pred2,Y_pred3,Y_pred4],axis=0)
    
       
    #plt.imshow(Y_pred[0,0]>0.5)
    diceMetric1[pidk]=calc_dice(Y_test_r,Y_pred1)*100
    diceMetric2[pidk]=calc_dice(Y_test_r,Y_pred2)*100
    diceMetric3[pidk]=calc_dice(Y_test_r,Y_pred3)*100    
    diceMetric4[pidk]=calc_dice(Y_test_r,Y_pred4)*100    
    DM_ens[pidk]=calc_dice(Y_test_r,Y_pred)*100    
    print ('-'*50)
    print 'Croped size: %s' %img_hw    
    print 'Dice Metric1: %2.1f' %diceMetric1[pidk]
    print 'Dice Metric2: %2.1f' %diceMetric2[pidk]
    print 'Dice Metric3: %2.1f' %diceMetric3[pidk]
    print 'Dice Metric4: %2.1f' %diceMetric4[pidk]
    print 'Dice Metric Ensemble: %2.1f' %DM_ens[pidk]
    # display sample image
    #disp_img_mask(X_test_r,Y_pred>.5)

    # remap to original size    
    Y_po1=remap(Y_pred1,Y_test)    
    Y_po2=remap(Y_pred2,Y_test)    
    Y_po3=remap(Y_pred3,Y_test)    
    Y_po4=remap(Y_pred4,Y_test)
    Y_po=np.mean([Y_po1,Y_po2,Y_po3,Y_po4],axis=0)    
    #Y_po=np.mean([Y_po2,Y_po3,Y_po4],axis=0)    
    DM1[pidk]=calc_dice(Y_test,Y_po1,2)*100
    DM2[pidk]=calc_dice(Y_test,Y_po2,2)*100
    DM3[pidk]=calc_dice(Y_test,Y_po3,2)*100    
    DM4[pidk]=calc_dice(Y_test,Y_po4,2)*100    
    DM_ens_orig[pidk]=calc_dice(Y_test,Y_po,2)*100    
    print 'Dice Metric1: %2.1f' %DM1[pidk]
    print 'Dice Metric2: %2.1f' %DM2[pidk]
    print 'Dice Metric3: %2.1f' %DM3[pidk]
    print 'Dice Metric4: %2.1f' %DM4[pidk]
    print 'patient %d, Dice Metric Ensemble : %2.1f' %(pidk,DM_ens_orig[pidk])

    pidk=pidk+1
    
    # save results as tif
    path2results='./results/'
    for k in range(Y_pred.shape[0]):
        imgmask=image_with_mask(X_test[:,:,k],Y_po[:,:,k]>0.5)
        imgmask=image_with_mask(imgmask,Y_test[:,:,k]>0.5,'G')
        plt.imsave(path2results+dataset+npfnm[:-4]+'_'+str(k)+'.tif',imgmask,cmap=plt.cm.gray)

# average dice
avg_dice1=np.mean(diceMetric1[diceMetric1>0])
avg_dice2=np.mean(diceMetric2[diceMetric2>0])
avg_dice3=np.mean(diceMetric3[diceMetric3>0])
avg_dice4=np.mean(diceMetric4[diceMetric4>0])
avg_dice=np.mean(DM_ens[DM_ens>0])
print ('-'*50)
print 'Croped size: %s' %img_hw    
print 'Average Dice1= %2.1f' %avg_dice1
print 'Average Dice2= %2.1f' %avg_dice2
print 'Average Dice3= %2.1f' %avg_dice3
print 'Average Dice4= %2.1f' %avg_dice4
print 'Ensemble Average Dice= %2.1f' %avg_dice

# original
avg_DM1=np.mean(DM1[DM1>0])
avg_DM2=np.mean(DM2[DM2>0])
avg_DM3=np.mean(DM3[DM3>0])
avg_DM4=np.mean(DM4[DM4>0])
avg_DM=np.mean(DM_ens_orig[DM_ens_orig>0])
print ('-'*50)
print 'Average Dice1= %2.1f' %avg_DM1
print 'Average Dice2= %2.1f' %avg_DM2
print 'Average Dice3= %2.1f' %avg_DM3
print 'Average Dice4= %2.1f' %avg_DM4
print 'Ensemble Average Dice= %2.1f' %avg_DM

np.savetxt(path2results+'ensemble-results.csv',(patientlist,DM_ens_orig), '%2.2f', delimiter=',')