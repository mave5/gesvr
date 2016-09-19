# n-fold cross-validation
import numpy as np
import os
from sklearn.cross_validation import KFold
import cv2
import matplotlib.pylab as plt


# path to data
path2set="../dcom/TrainingSet/"
path2numpy = path2set+"numpy/"


def concatdata(indices,npflist):
    for k in indices:
        path2npfn=path2numpy+npflist[k]
        print path2npfn
        tmp=np.load(path2npfn) # load numpy file
        print 'X shape: ', tmp['X'].shape
        print 'Y shape: ', tmp['Y'].shape
        if k==indices[0]:
            X=tmp['X'] 
            Y=tmp['Y']
        else:
            X=np.append(X,tmp['X'],axis=2)    
            Y=np.append(Y,tmp['Y'],axis=2)
    return X,Y


def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))

def image_with_mask(img, mask):
    maximg=np.max(img)    
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


# list of numpy files
npflist=os.listdir(path2numpy)
Nf=len(npflist)
print 'total number of files: %d' %Nf

# n-fold cross validation
kf = KFold(Nf, n_folds=4)
foldn=0
# create numpy folder if does no exist
path2nfolds=path2numpy+'nfolds/'
if  not os.path.exists(path2nfolds):
    os.makedirs(path2nfolds)
    print 'nfolds folder created'
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



# display sample image
n1=np.random.randint(Xtrain.shape[2])
img=Xtrain[:,:,n1]
mask=Ytrain[:,:,n1]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)

# display sample image
n1=np.random.randint(Xtest.shape[2])
img=Xtest[:,:,n1]
mask=Ytest[:,:,n1]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)


