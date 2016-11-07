# clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
#from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from numpy.random import permutation
import time
import cv2
from model_dice2 import *
import os,shutil
import glob
import dicom
from skimage import draw
#from skimage import exposure
print 'All libs successfully loaded!'

# current working directory
print os.getcwd() 

# path to dicom
patientID='48' # Test1: 17 to 32 and Test2: 33 to 48 
pp_ena=0 # post processing enable/disable

#path2set="../dcom/TrainingSet/"
#dataset='Test1Set/'
dataset='Test2Set/'
path2set="../dcom/"+dataset
path2numpy = path2set+"numpy/"
path2dicom = path2set+"patient"+patientID


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

# sample image display
def disp_img_mask(img,mask,d=0):
    if len(mask.shape)>2:    
        n1=np.random.randint(img.shape[d])    
        if d==0:    
            I1=img[n1,0]
            M1=mask[n1,0]
        else:
            I1=img[:,:,n1]
            M1=mask[:,:,n1]
    else:
        I1=img
        M1=mask
        n1=0
                
    imgmask=image_with_mask(I1,M1)
    plt.imshow(imgmask)
    plt.title(n1)


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


# convert polygon to binary mask
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def mask2poly(Y):
    # find contours
    BW=np.asarray(Y>0.5,dtype='uint8')
    cs,_ = cv2.findContours( BW, mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE )
    for k in range(len(cs)):
        ck=cs[k]
        ck=np.asarray(ck)
        ck=np.squeeze(ck)
        if len(ck.shape)==1:
            ck=np.expand_dims(ck,0)
        if k==0:
            csnp=ck
        elif len(ck)>1:
            csnp=np.append(csnp,ck,axis=0)
    if len(cs)==0:
           csnp=ck=np.asarray(cs)     
    return csnp     
    
# select the contour with largest area
def pick_largest_area(Y):
    # find contours
    BW=np.asarray(Y>0.5,dtype='uint8')
    cs,_ = cv2.findContours( BW.astype('uint8'), mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE )
    for k2 in range(len(cs)):    
        c=cs[k2]
        m = cv2.moments(c)
        if k2==0:
            Area=m['m00']
            if m['m00']!= 0:            
                cnt= ( m['m10']/m['m00'],m['m01']/m['m00'] )
        else:
            Area=np.append(Area, m['m00'])
            if m['m00']!=0:            
                cnt=np.append(cnt,(m['m10']/m['m00'],m['m01']/m['m00']))
            
    print 'Area: %s and Center: %s ' %(Area,cnt)
    kmax=np.argmax(Area)
    c1=cs[kmax]
    c1=np.squeeze(c1)
    cnt1=cnt[kmax*2:kmax*2+2]
    # return contour with largest area    
    return c1,cnt1


def find_centers(Y):
    # find contours
    BW=np.asarray(Y>0.5,dtype='uint8')
    cs,_ = cv2.findContours( BW.astype('uint8'), mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE )
    
    cnt=np.array([0,0])   
    for k2 in range(len(cs)):    
        c=cs[k2]
        m = cv2.moments(c)
        if k2==0:
            #Area=m['m00']
            if m['m00']!= 0:            
                cnt= ( m['m10']/m['m00'],m['m01']/m['m00'] )
            else:
                cnt=np.array([0,0])
        else:
            #Area=np.append(Area, m['m00'])
            if m['m00']!=0:            
                cnt=np.append(cnt,(m['m10']/m['m00'],m['m01']/m['m00']))
    # return centers and contours            
    return cnt,cs
    
def find_nearest(Y,basecnt):
    Y=Y>.5
    basecnt=np.array(basecnt)
    # find centers
    cnt,cs=find_centers(Y)
    
    # number of contours    
    nn=len(cnt)/2
    print 'number of contours: %s' %nn
    if (cnt[0]==0 and nn<=1):
        cnt=basecnt
        print 'center 0'


    delta=np.zeros((nn,1))    
    for k2 in range(nn):    
        delta[k2]=np.linalg.norm(cnt[k2*2:k2*2+2]-basecnt)

    kmin=np.argmin(delta)
    if len(cs)!=0:    
        c1=cs[kmin]
        c1=np.squeeze(c1)
    else:
        c1=cs
    return c1,delta

            
def polygon2bw(C,X):
    img = np.zeros(X.shape[:2], dtype=np.uint8)
    if len(C)!=0:
        rr,cc=draw.polygon(C[:,1],C[:,0])
        img[rr,cc] = 1
    return img


# read dicom files 
def read_dicom(PathDicom):
    # delete tif folder if exist
    path2tif=PathDicom+'/tif'
    if  os.path.exists(path2tif):
        shutil.rmtree(path2tif)
        print 'tif folder was deleted.'

    # get the list and and path to dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # read text file which contains the list of manual contours
    path2txt= glob.glob(PathDicom+'/*.txt')
    txt = open(path2txt[0])
    manuallst=txt.readlines()
        
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(manuallst)/2)

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    x = np.arange(0.0, (ConstPixelDims[0])*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1])*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2])*ConstPixelSpacing[2], ConstPixelSpacing[2])            

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    ArrayMask = np.zeros(ConstPixelDims, dtype='uint8')

    # loop through all the DICOM files
    k=0
    for manualfilename in manuallst:
        # only dicom file with internal manual contour is loaded
        filenameDCM=dirName+'/'+manualfilename[31:-22]+'.dcm'
        if manualfilename[40:48]=='icontour': # only internal contours
            ds = dicom.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[:, :, k] = ds.pixel_array  
            # read manual polygon    
            manfn=dirName[:-8]+manualfilename[12:-31]+'/'+manualfilename[31:-2]
            if os.path.exists(manfn):                 
                polygonxy=np.loadtxt(manfn)
                ArrayMask[:,:,k]=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
            k=k+1   
            print 'dicom file: ', filenameDCM
    
    # if h<w then convert h*w to w*h
    if ArrayDicom.shape[0]<ArrayDicom.shape[1]:
        rotated_hw=1 # flag to check if image rotated or not
        ArrayDicom=np.transpose(ArrayDicom,(1,0,2))
        ArrayDicom=ArrayDicom[::-1]
        ArrayMask=np.transpose(ArrayMask,(1,0,2))
        ArrayMask=ArrayMask[::-1]
        print 'transpose was done!'
    else:
        rotated_hw=0
    
    print 'Dicom files loaded!'
    print   'X: %s,%s,%s' %(ArrayDicom.shape) 
    print   'Y: %s,%s,%s' %(ArrayMask.shape) 
    print  'rotation: %s' %rotated_hw
    print ('-'*50)
    return ArrayDicom,ArrayMask,rotated_hw,manuallst


def rotate_hw(X):
    print ('-' *50)    
    print   'X: %s,%s,%s' %(X.shape) 
    X=X[::-1] # mirror
    X=np.transpose(X,(1,0,2)) # rotate
    print 'rotated h,w!'
    print   'X: %s,%s,%s' %(X.shape)     
    print ('-' *50)    
    return X        
    
#######################################################################################################################
#######################################################################################################################


# load dicom files
X_test,Y_test,rotated_hw,manuallist=read_dicom(path2dicom)

# preprocess test data
crop_ena=1
img_hw= np.array([176,176])
X_test_r,Y_test_r=preprocess(X_test,Y_test,img_hw,crop_ena)
# display sample image
#disp_img_mask(X_test_r,Y_test_r)

# path to best trained weights
weights_path1='./weights/fold1-aug-weights-improvement-99--0.79.hdf5'
weights_path2='./weights/fold2-aug-weights-improvement-136--0.80.hdf5'
weights_path3='./weights/fold3-aug-weights-improvement-00--0.84.hdf5'
weights_path4='./weights/fold4-aug-weights-improvement-00--0.78.hdf5'

# build nfold models   
model1 = model(weights_path1,img_hw[0],img_hw[1],1e-4)
model2 = model(weights_path2,img_hw[0],img_hw[1],1e-4)
model3 = model(weights_path3,img_hw[0],img_hw[1],1e-4)
model4 = model(weights_path4,img_hw[0],img_hw[1],1e-4)
 
# predicts
print 'please wait to test ...'
Y_pred1=model1.predict(X_test_r)
Y_pred2=model2.predict(X_test_r)
Y_pred3=model3.predict(X_test_r)  
Y_pred4=model4.predict(X_test_r)
Y_pred=np.mean([Y_pred1,Y_pred2,Y_pred3,Y_pred4],axis=0) # ensemble
print 'predicts are ready!'    
#plt.imshow(Y_pred[np.random.randint(Y_pred.shape[0]),0]>0.5)
print ('-'*50)

# calculate nfold dice
diceMetric1=calc_dice(Y_test_r,Y_pred1)*100
diceMetric2=calc_dice(Y_test_r,Y_pred2)*100
diceMetric3=calc_dice(Y_test_r,Y_pred3)*100    
diceMetric4=calc_dice(Y_test_r,Y_pred4)*100    
# ensemble dice
DM_ens=calc_dice(Y_test_r,Y_pred)*100    
print ('-'*50)
print 'Dice calculated from croped size: %s' %img_hw    
print 'Dice Metric1: %2.1f' %diceMetric1
print 'Dice Metric2: %2.1f' %diceMetric2
print 'Dice Metric3: %2.1f' %diceMetric3
print 'Dice Metric4: %2.1f' %diceMetric4
print 'Dice Metric Ensemble: %2.1f' %DM_ens
# display sample image
#disp_img_mask(X_test_r,Y_pred>.5)

# remap to original size    
Y_po1=remap(Y_pred1,Y_test)    
Y_po2=remap(Y_pred2,Y_test)    
Y_po3=remap(Y_pred3,Y_test)    
Y_po4=remap(Y_pred4,Y_test)
Y_po=np.mean([Y_po1,Y_po2,Y_po3,Y_po4],axis=0)    
DM1=calc_dice(Y_test,Y_po1,2)*100
DM2=calc_dice(Y_test,Y_po2,2)*100
DM3=calc_dice(Y_test,Y_po3,2)*100    
DM4=calc_dice(Y_test,Y_po4,2)*100    
DM_ens_orig=calc_dice(Y_test,Y_po,2)*100    
print 'Dice Metric1: %2.1f' %DM1
print 'Dice Metric2: %2.1f' %DM2
print 'Dice Metric3: %2.1f' %DM3
print 'Dice Metric4: %2.1f' %DM4
print 'Dice Metric Ensemble: %2.1f' %DM_ens_orig
print ('-'*50)
disp_img_mask(X_test,Y_po>.5,2)

# post process the base image
if pp_ena==1:
    C1,base_cnt=pick_largest_area(Y_po[:,:,0])
    plt.imshow(X_test[:,:,0],cmap='Greys_r')
    plt.plot(C1[:,0],C1[:,1])
    Y_po[:,:,0]=polygon2bw(C1,Y_po)
    plt.imshow(Y_po[:,:,0])
    disp_img_mask(X_test[:,:,0],Y_po[:,:,0])

# post process the remaining slices
    prev_cnt=base_cnt
    for k in range(1,Y_po.shape[2]):
        Ck,deltak=find_nearest(Y_po[:,:,k],prev_cnt)
        Y_po[:,:,k]=polygon2bw(Ck,Y_po)
        prev_cnt,tmp=find_centers(Y_po[:,:,k])

    disp_img_mask(X_test,Y_po>.5,2)

    DM_post=calc_dice(Y_test,Y_po,2)*100

    print 'Dice Metric before postprocessing: %2.1f' %DM_ens_orig
    print 'DM post processed: %2.1f' %DM_post
   

##############################################################################
##############################################################################
# saving results  

path2results='./results/'
path2patient=path2results+dataset+'pateint'+patientID+'/'
path2tif=path2patient+'tif/'
if  not os.path.exists(path2tif):
    os.makedirs(path2tif)
    print 'tif folder created'
    
# rotate hw if required
if rotated_hw==1:
    X_test2=rotate_hw(X_test)
    X_test2=X_test2.copy()
    Y_po2=rotate_hw(Y_po)
    Y_po2=Y_po2.copy()
else:
    X_test2=X_test
    Y_po2=Y_po    
disp_img_mask(X_test2,Y_po2>.5,2)


# save results into text files
k=0
for manualfilename in manuallist:
    # only dicom file with internal manual contour is loaded
    if manualfilename[40:48]=='icontour': # only internal contours
        manualfilename=manualfilename.replace('manual','auto')
        if np.sum(Y_po2[:,:,k]>.5)<1:
            Y_po2[:,:,k]=Y_po2[:,:,k-1]
        cs=mask2poly(Y_po2[:,:,k])
        np.savetxt(path2patient+manualfilename[29:55],cs, delimiter=' ',fmt='%.1f')
        k=k+1   
        print 'contour saved into text file: ', manualfilename[29:55]

# save results as tif files
for k in range(Y_po.shape[2]):
    imgmask=image_with_mask(X_test2[:,:,k],Y_po2[:,:,k]>0.5)
    #imgmask=image_with_mask(imgmask,Y_test[:,:,k]>0.5,'G')
    plt.imsave(path2tif+patientID+'_'+str(k)+'.tif',imgmask,cmap=plt.cm.gray)
