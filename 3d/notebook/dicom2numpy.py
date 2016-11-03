# convert dicom to numpy files
import dicom
import numpy as np
import os,shutil
import numpy
import matplotlib.pylab as plt
import glob
import cv2
from skimage import draw

# path to dicom
patientID='02'
path2set="../dcom/TrainingSet/"
#path2set="../dcom/Test1Set/"
PathDicom = path2set+"patient"+patientID

print 'libs successfully loaded!'
print '-' *50
#%%
def rotateimgs(imgs):
    (h,w,n) = imgs.shape
    center = (w / 2, h / 2)
    imgs_r=np.zeros((w,h,n),dtype=imgs.dtype)
    for k in range(n):    
        img=imgs[:,:,k]
        M = cv2.getRotationMatrix2D(center, 90, -1.0)        
        imgs_r[:,:,k] = cv2.warpAffine(img,M,(h,w))
    return imgs_r        

def rotateimg(img):
    (h,w) = img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 90, -1.0)        
    img_r = cv2.warpAffine(img,M,(h,w))
    return img_r        

# convert polygon to binary mask
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask
    

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
    #print np.sum(mask_edges)
    img_color[mask_edges, 0] = maximg  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    img_color=img_color/float(np.max(img))
    return img_color


# read ED dicom into 3D arrays
def read_dicom_ed(PathDicom,contour_type='icontour'):       
    # path to dicom files
    p2dcom=PathDicom+'/P'+patientID+'dicom'

    # delete tif folder if exist
    path2tif=PathDicom+'/tifED'
    if  os.path.exists(path2tif):
        shutil.rmtree(path2tif)
        print 'tif folder was deleted.'
    # create new tif folder
    os.mkdir(path2tif)
    
    
    # get the list and and path to dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(p2dcom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    
    # load total number of phases
    nphases=RefDs.CardiacNumberofImages
    
    # read text file which contains the list of manual contours
    path2txt= glob.glob(PathDicom+'/*.txt')
    txt = open(path2txt[0])
    manuallst=txt.readlines()

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(manuallst)/2)

    # loop through all the DICOM files
    k=0
    for manualfilename in manuallst:
        # only dicom file with contour_type is loaded
        filenameDCM=dirName+'/'+manualfilename[31:-22]+'.dcm'
        if manualfilename[40:48]==contour_type: # only internal contours
            if int(manualfilename[36:39])%nphases==0:
                print 'dicom file: ', filenameDCM
                ds = dicom.read_file(filenameDCM)
                # read manual polygon    
                manfn=dirName[:-8]+manualfilename[12:-31]+'/'+manualfilename[31:-2]
                polygonxy=np.loadtxt(manfn)
                # store the raw image data
                if k==0:
                    ArrayDicom = np.expand_dims(ds.pixel_array,axis=2)
                    tmp1=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
                    ArrayMask=np.expand_dims(tmp1,axis=2)
                else:
                    ArrayDicom = np.append(ArrayDicom,np.expand_dims(ds.pixel_array,axis=2),axis=2)  
                    tmp1=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
                    ArrayMask = np.append(ArrayMask,np.expand_dims(tmp1,axis=2),axis=2)  

                plt.imsave(path2tif+'/'+manualfilename[31:-22]+'.tif',ArrayDicom[:,:,k],cmap=plt.cm.gray)
                plt.imsave(path2tif+'/'+manualfilename[31:-22]+'_mask.tif',ArrayMask[:,:,k],cmap=plt.cm.gray)

                k=k+1   
    return ArrayDicom,ArrayMask

# read ED dicom into 3D arrays
def read_dicom_es(PathDicom,contour_type='icontour'):       
    p2dcom=PathDicom+'/P'+patientID+'dicom'
    
    # delete tif folder if exist
    path2tif=PathDicom+'/tifES'
    if  os.path.exists(path2tif):
        shutil.rmtree(path2tif)
        print 'tif folder was deleted.'
    # create new tif folder
    os.mkdir(path2tif)

    # get the list and and path to dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(p2dcom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    
    # load total number of phases
    nphases=RefDs.CardiacNumberofImages
    
    # read text file which contains the list of manual contours
    path2txt= glob.glob(PathDicom+'/*.txt')
    txt = open(path2txt[0])
    manuallst=txt.readlines()

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(manuallst)/2)

    # loop through all the DICOM files
    k=0
    for manualfilename in manuallst:
        # only dicom file with contour_type is loaded
        filenameDCM=dirName+'/'+manualfilename[31:-22]+'.dcm'
        if manualfilename[40:48]==contour_type: # only internal contours
            if int(manualfilename[36:39])%nphases!=0:
                print 'dicom file: ', filenameDCM
                ds = dicom.read_file(filenameDCM)
                # read manual polygon    
                manfn=dirName[:-8]+manualfilename[12:-31]+'/'+manualfilename[31:-2]
                polygonxy=np.loadtxt(manfn)
                # store the raw image data
                if k==0:
                    ArrayDicom = np.expand_dims(ds.pixel_array,axis=2)
                    tmp1=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
                    ArrayMask=np.expand_dims(tmp1,axis=2)
                else:
                    ArrayDicom = np.append(ArrayDicom,np.expand_dims(ds.pixel_array,axis=2),axis=2)  
                    tmp1=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
                    ArrayMask = np.append(ArrayMask,np.expand_dims(tmp1,axis=2),axis=2)  
                plt.imsave(path2tif+'/'+manualfilename[31:-22]+'.tif',ArrayDicom[:,:,k],cmap=plt.cm.gray)
                plt.imsave(path2tif+'/'+manualfilename[31:-22]+'_mask.tif',ArrayMask[:,:,k],cmap=plt.cm.gray)
                k=k+1                   
    return ArrayDicom,ArrayMask

#%%
#==============================================================================
# main code
#==============================================================================


# swap h and w
def swap_hw(X,Y):
    X=np.transpose(X,(1,0,2))
    X=X[::-1]
    Y=np.transpose(Y,(1,0,2))
    Y=Y[::-1]
    return X,Y
    
# read ED images and masks
X_ED,Y_ED=read_dicom_ed(PathDicom)
print 'X,Y', X_ED.shape,Y_ED.shape
# read ES images and masks
X_ES,Y_ES=read_dicom_es(PathDicom)
print 'X,Y', X_ES.shape,Y_ES.shape


# if h<w then convert h*w to w*h
if X_ED.shape[0]<X_ED.shape[1]:
    X_ED,Y_ED=swap_hw(X_ED,Y_ED)
    X_ES,Y_ES=swap_hw(X_ES,Y_ES)  
    print 'transpose was done!'

print 'X,Y', X_ED.shape,Y_ED.shape
print 'X,Y', X_ES.shape,Y_ES.shape    

# create numpy folder if does no exist
path2numpy=path2set+'numpy'
if  not os.path.exists(path2numpy):
    os.makedirs(path2numpy)
    print 'numpy folder created'

# save as numpy files
print 'wait to save data as numpy files'
np.savez(path2numpy+'/ED_p'+patientID, X=X_ED,Y=Y_ED)
np.savez(path2numpy+'/ES_p'+patientID, X=X_ES,Y=Y_ES)
print 'numpy file was saved!'

# display sample image
n1=np.random.randint(X_ED.shape[2])
img=X_ED[:,:,n1]
mask=Y_ED[:,:,n1]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)


