# create auto list
import dicom
import numpy as np
import os,shutil
import numpy
import matplotlib.pylab as plt
import glob

#%%

# path to dicom
patientID='36'
#path2set="../dcom/Test1Set/"
path2set="../dcom/Test2Set/"
PathDicom = path2set+"patient"+patientID

#%%


# read text file which contains the list of manual contours
path2txt= glob.glob(PathDicom+'/*.txt')
txt = open(path2txt[0])
manuallst=txt.readlines()


auto1lst=[]
for manualfilename in manuallst:
    print manualfilename
    auto1filename=manualfilename.replace('manual','auto')
    auto1filename=auto1filename.replace('contours-auto','contours-auto1')
    print auto1filename
    auto1lst.append(auto1filename)
    
print auto1lst    

# write text file which contains the list of manual contours
path2txtauto1= path2txt[0].replace('.txt','_auto1.txt')
txt = open(path2txtauto1,'w')
txt.writelines(auto1lst)
txt.close()
    


auto2lst=[]
for manualfilename in manuallst:
    print manualfilename
    auto2filename=manualfilename.replace('manual','auto')
    auto2filename=auto2filename.replace('contours-auto','contours-auto2')
    print auto2filename
    auto2lst.append(auto2filename)
    
print auto2lst    

# write text file which contains the list of manual contours
path2txtauto2= path2txt[0].replace('.txt','_auto2.txt')
txt2 = open(path2txtauto2,'w')
txt2.writelines(auto2lst)
txt2.close()
