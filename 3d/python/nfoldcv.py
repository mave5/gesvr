# create n-fold cross-validation data
import numpy as np
import os
from sklearn.cross_validation import KFold
import cv2
import matplotlib.pylab as plt

# path to data
path2set="../dcom/TrainingSet/"
path2numpy = path2set+"numpy/"

#%%

# temp function 
def temp_func1(tmp,Nstack):
    t1=tmp['X'] 
    x=np.zeros((t1.shape[0],t1.shape[1],Nstack),dtype=t1.dtype)
    x[:,:,0:t1.shape[2]]=t1
    x=np.transpose(x,(2,0,1))
    x=np.expand_dims(x,axis=0)
    
    t2=tmp['Y']
    y=np.zeros((t2.shape[0],t2.shape[1],Nstack),dtype=t2.dtype)
    y[:,:,0:t2.shape[2]]=t2
    y=np.transpose(y,(2,0,1))
    y=np.expand_dims(y,axis=0)

    return x,y

def concatdata(indices,npflist,Nstack=10):
    for k in indices:
        if npflist[k][:2]=='ED':
            # load ED image
            path2npfn=path2numpy+npflist[k]
            print path2npfn
            tmp1=np.load(path2npfn) # load numpy file
            
            # load ES image            
            path2npfn2=path2numpy+'ES'+npflist[k][2:]
            print path2npfn2
            tmp2=np.load(path2npfn2) # load numpy file
            
        if k==indices[0]:
            X,Y=temp_func1(tmp1,Nstack)
            x2,y2=temp_func1(tmp2,Nstack)            
            X=np.append(X,x2,axis=0)    
            Y=np.append(Y,y2,axis=0)
        else:
            x1,y1=temp_func1(tmp1,Nstack)
            x2,y2=temp_func1(tmp2,Nstack)
            X=np.append(X,x1,axis=0)    
            Y=np.append(Y,y1,axis=0)
            
            X=np.append(X,x2,axis=0)    
            Y=np.append(Y,y2,axis=0)
            
        print 'X shape: ', X.shape
        print 'Y shape: ', Y.shape
    return X,Y


def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))

def image_with_mask(img, mask):
    maximg=np.max(img) 
    mask=np.array(mask,dtype='uint8')
    if np.max(mask)==1:
        mask=mask*255
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0
    print np.sum(mask_edges)
    img_color[mask_edges, 0] = maximg  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    img_color=img_color/float(np.max(img))
    return img_color

#%%
       
# list of numpy files
npflist=os.listdir(path2numpy)
Nf=len(npflist)/2
print 'total number of files: %d' %Nf

# n-fold cross validation
kf = KFold(Nf, n_folds=4)
foldn=0

# create numpy folder if does no exist
path2nfolds=path2numpy+'nfolds/'
if  not os.path.exists(path2nfolds):
    os.makedirs(path2nfolds)
    print 'nfolds folder created'
else:
    print 'n-fold dir exists!'

for train, test in kf:
    print("%s %s" % (train, test))   
    Xtrain,Ytrain=concatdata(train,npflist)
    print 'Xtrain shape: ', Xtrain.shape
    print 'Ytrain shape: ', Ytrain.shape
    Xtest,Ytest=concatdata(test,npflist)
    print "Xtest shape: ",  (Xtest.shape)
    print 'Ytest shape:', Ytest.shape
    print 'wait to save data as numpy files'
    foldn=foldn+1
    np.savez(path2nfolds+'trainfold'+str(foldn), X=Xtrain,Y=Ytrain)
    np.savez(path2nfolds+'testfold'+str(foldn), X=Xtest,Y=Ytest)
    print 'numpy file was saved!'


#%%

plt.subplot(121)
# display sample image
n1=np.random.randint(Xtrain.shape[0])
n2=np.random.randint(Xtrain.shape[1])
img=Xtrain[n1,n2]
mask=Ytrain[n1,n2]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)

# display sample image
plt.subplot(122)
n1=np.random.randint(Xtest.shape[0])
n2=np.random.randint(Xtest.shape[1])
img=Xtest[n1,n2]
mask=Ytest[n1,n2]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)


