import time
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt


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


def image_with_mask(img, mask,color=(0,255,0)):
    maximg=np.max(img)    
    img=np.asarray(img,dtype='float32')
    img=np.asarray((img/maximg)*255,dtype='uint8')
    mask=np.asarray(mask,dtype='uint8') 
    if np.max(mask)==1:
        mask=mask*255

    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)==2:	
	img_color = grays_to_RGB(img)
    else:
	img_color =img

    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = color[1]
    img_color[mask_edges, 2] = color[2]
    img_color=img_color#/float(np.max(img))
    return img_color

# sample
def disp_img_mask(img,mask,r=1,c=1,d=0):
    if mask is None:
        mask=np.zeros(img.shape,dtype='uint8')
    N=r*c    
    if d==2:
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=1)
        mask=np.transpose(mask,(2,0,1))
        mask=np.expand_dims(mask,axis=1)
    n1=np.random.randint(img.shape[0],size=N)
    
    #if d==0:    
    I1=img[n1,0]
    M1=mask[n1,0]
    M11=(M1==1)
    M12=(M1==2)
    #else:
    #   I1=img[:,:,n1]
    #  M1=mask[:,:,n1]
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M11[k])
        imgmask=image_with_mask(imgmask,M12[k],(255,0,0))
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
    return n1        
        

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
    
    
# histogram equalization
def hist_equ(X):
    X=np.asarray(X,dtype='uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    Xeq=np.zeros(X.shape,dtype=X.dtype)
    for k in range(X.shape[2]):
        Xeq[:,:,k] = clahe.apply(X[:,:,k])   
    return Xeq        

def disp_hist(X,n1=1,r=1,c=1,d=0):
    if d==2:
        X=np.transpose(X,(2,0,1))
        X=np.expand_dims(X,axis=1)
    I1=X[n1,0]
    min1=np.min(I1)
    max1=np.max(I1)
    
    N=len(n1)
    for k in range(N):    
        plt.subplot(r,c,k+1)
        plt.hist(I1[k,0].ravel(),256,[min1,max1])
        plt.title(n1[k])


# convert contours to centers
def contour2roi(Y):
    Y=np.array(Y>0,dtype='uint8')    
    N=Y.shape[0]
    cXY=np.zeros((N,2))
    
    for k in range(N):    
        ret,thresh = cv2.threshold(255*Y[k,0],127,255,0)
        contours,im2= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c = contours[0]
    
        # compute the center of the contour
        M = cv2.moments(c)
        cX,cY=0,0
        if M["m00"]!= 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        
        cXY[k,:]=(cX,cY)
    return cXY
    
    



# sample
def disp_img_2masks(img,mask1,mask2,r=1,c=1,d=0,indices=None):
    if mask1 is None:
        mask1=np.zeros(img.shape,dtype='uint8')
    if mask2 is None:
        mask2=np.zeros(img.shape,dtype='uint8')
        
    N=r*c    
    if d==2:
        # convert to N*C*H*W
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=1)
        
        mask1=np.transpose(mask1,(2,0,1))
        mask1=np.expand_dims(mask1,axis=1)

        mask2=np.transpose(mask2,(2,0,1))
        mask2=np.expand_dims(mask2,axis=1)
        
    if indices is None:    
        # random indices   
        n1=np.random.randint(img.shape[0],size=N)
    else:
        n1=indices
    
    I1=img[n1,0]
    M1=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask1.shape[1]):
        M1=np.logical_or(M1,mask1[n1,c1,:])
    
    #M1=mask1[n1,0]
    #M2=mask2[n1,0]
    M2=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask2.shape[1]):
        M2=np.logical_or(M2,mask2[n1,c1,:])
    
    C1=(0,255,9)
    C2=(255,0,0)
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M1[k],C1)
        imgmask=image_with_mask(imgmask,M2[k],C2)
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
    plt.show()            
    return n1    

