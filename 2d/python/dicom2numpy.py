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
patientID='16'
path2set="../dcom/TrainingSet/"
PathDicom = path2set+"patient"+patientID

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

# create tif folder
os.makedirs(path2tif)
print 'tif folder created'

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

x = numpy.arange(0.0, (ConstPixelDims[0])*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1])*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2])*ConstPixelSpacing[2], ConstPixelSpacing[2])            

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
ArrayMask = numpy.zeros(ConstPixelDims, dtype='uint8')

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
        polygonxy=np.loadtxt(manfn)
        ArrayMask[:,:,k]=poly2mask(polygonxy[:,1], polygonxy[:,0], ConstPixelDims[0:2])
        k=k+1   

    print 'dicom file: ', filenameDCM
print 'Dicom files loaded!'

# if h<w then convert h*w to w*h
if ArrayDicom.shape[0]<ArrayDicom.shape[1]:
    ArrayDicom=np.transpose(ArrayDicom,(1,0,2))
    ArrayDicom=ArrayDicom[::-1]
    ArrayMask=np.transpose(ArrayMask,(1,0,2))
    ArrayMask=ArrayMask[::-1]
    print 'transpose was done!'
    
# save as tif images
k=0
for manualfilename in manuallst:
    # read the file
    if manualfilename[40:48]=='icontour':    
        #print manualfilename
        # save as tif        
        plt.imsave(path2tif+'/'+manualfilename[31:-22]+'.tif',ArrayDicom[:,:,k],cmap=plt.cm.gray)
        plt.imsave(path2tif+'/'+manualfilename[31:-22]+'_mask.tif',ArrayMask[:,:,k],cmap=plt.cm.gray)
        k=k+1   
print 'tif files saved!'


# create numpy folder if does no exist
path2numpy=path2set+'numpy'
if  not os.path.exists(path2numpy):
    os.makedirs(path2numpy)
    print 'numpy folder created'

# save as numpy files
print 'wait to save data as numpy files'
np.savez(path2numpy+'/p'+patientID, X=ArrayDicom,Y=ArrayMask)
print 'numpy file was saved!'

# display sample image
n1=np.random.randint(ArrayDicom.shape[2])
img=ArrayDicom[:,:,n1]
mask=ArrayMask[:,:,n1]
imgmask=image_with_mask(img,mask)
plt.imshow(imgmask)




