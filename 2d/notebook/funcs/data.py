# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from uuid import uuid4
import numpy as np
import SharedArray
import random
import config
import utils
import cv2
import os

def create_data(images_paths, input_directory, output_directory, generate_mask, image_height, image_width):
    '''
    Creates data and dumps it on disk
    
    Args:
        images_paths(list of str)
        input_directory(str): the directory of the images
        output_directory(str): the output directory
        generate_mask(boolean): whether to generate masks or not
        image_heght(int)
        image_width(int)
    
    Returns:
        void
    '''
    
    # create the numpy arrays
    number_of_images = len(images_paths)
    images = np.ndarray((number_of_images, 1, image_height, image_width), dtype=np.uint8)
    denoised_images = np.ndarray((number_of_images, 1, image_height, image_width), dtype=np.uint8)
    masks = np.ndarray((number_of_images, 1, image_height, image_width), dtype=np.uint8)
    
    index=0
    for image_path in images_paths:
        if index%100==0:
            print index
        
        # read & resize an image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        denoised = cv2.fastNlMeansDenoising(image,None,10,7,21)
        
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.resize(denoised, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        
        images[index,0] = image
        denoised_images[index,0] = denoised
        
        # read & resize a mask
        if generate_mask:
            mask_path = image_path.replace('.tif','_mask.tif')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
            masks[index,0] = mask
            
        index += 1
    
    # Dump the created data to disk
    utils.create_directory(output_directory)
    joblib.dump(images_paths,output_directory+'/images_paths.joblib')
    joblib.dump(images,output_directory+'/images.joblib')
    joblib.dump(denoised_images,output_directory+'/denoised_images.joblib')
    if generate_mask:
        joblib.dump(masks,output_directory+'/masks.joblib')

def get_perspective_transformation(image, kwargs):
    '''
    Computes a perspective transformation matrix
    
    Args:
        image: an input image
        kwargs: the augmentation options
    
    Returns:
        transformation_matrix
    '''
    height, width = np.shape(image)[0:2]
    
    tx = np.random.uniform(-kwargs['width_shift_range'], kwargs['width_shift_range']) * width
    ty = np.random.uniform(-kwargs['height_shift_range'], kwargs['height_shift_range']) * height
    
    theta = 0
    
    if kwargs['zoom_range']:zx, zy = np.random.uniform(kwargs['zoom_range'][0], kwargs['zoom_range'][1], 2)
        
    else:
        zx, zy = 1, 1
        
    transformation_matrix = np.array([[zx * np.cos(theta),     -np.sin(theta),    tx],
                                      [     np.sin(theta), zy * np.cos(theta),    ty],
                                      [0                 ,                  0,    1]])

    return transformation_matrix

def augment_image(args):
    '''
    Augments an image by performing various transformations.
    
    Args:
        imageIndex(int): an index to be used to fill the shared arrays.
        seed(int): a seed is very important because child processes tend to have the same seed as the parent.
        images_array_name(str): the automatically generated name for the images shared array.
        masks_array_name(str): the automatically generated name for the masks shared array
        image(numpy): an image
        mask(numpy): a mask
        kwargs(dictionary): the augmentation options as a dictionary
    
    Returns:
        image, mask
    '''
    
    image_index, seed, images_array_name, masks_array_name, image, mask, kwargs = args
    
    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    
    height,width = image.shape[0:2]
    
    if random.random()<kwargs['augmentation_probability']:
        if random.random()<kwargs['equalize_historgram_probability']:
            image = cv2.equalizeHist(image)
        
        if random.random()<kwargs['invert_image_probability']:
            image = 255-image
    
        if random.random()<kwargs['smoothing_probability']:
            #sigma = random.uniform(0,kwargs['max_smoothing_sigma'])
            sigma = kwargs['max_smoothing_sigma']
            image = cv2.GaussianBlur(image,(0,0),sigma)
        
        # Perspective Transform
        if random.random()<kwargs['geometric_transform_probability']:
            height, width = np.shape(image)[0:2]
            transformationMaxtrix = get_perspective_transformation(image, kwargs)
            image = cv2.warpPerspective(image,transformationMaxtrix,(width, height),borderMode=kwargs['border_mode'], borderValue=kwargs['border_value'])
            if masks_array_name:
                mask = cv2.warpPerspective(mask,transformationMaxtrix,(width, height),borderMode=kwargs['border_mode'], borderValue=kwargs['border_value'])
        
        # Rotate Image
        if random.random() < kwargs['rotation_probability']:
            theta = np.random.uniform(-kwargs['rotation_range'], kwargs['rotation_range'])
            M = cv2.getRotationMatrix2D((width/2,height/2),theta,1)
            image = cv2.warpAffine(image,M,(width,height))
            if masks_array_name:
                mask = cv2.warpAffine(mask,M,(width,height))
    
        # Flipping
        if random.random() < kwargs['horizontal_flip_probability']:
            image = np.fliplr(image)
            if masks_array_name:
                mask = np.fliplr(mask)
        
        if random.random() < kwargs['vertical_flip_probability']:
            image = np.flipud(image)
            if masks_array_name:
                mask = np.flipud(mask)
    
    image = image.astype(float)
    
#    if len(np.shape(image))==2:
#        image = image - mean
#        
#        if std!=0:
#            image /= std
#    else:
#        for channel_index in range(np.shape(image)[2]):
#            image[:,:,channel_index] = image[:,:,channel_index] - means[channel_index]
#        
#            if stds[channel_index]!=0:
#                image[:,:,channel_index] = image[:,:,channel_index] / stds[channel_index]
        
    
    images_array = SharedArray.attach(images_array_name)
    if len(image.shape)==3:
        image = image.transpose((2,0,1))
    images_array[image_index] = image
    
    # if an annotation, is provided --> you should attach the mask
    if masks_array_name:
        masks_array = SharedArray.attach(masks_array_name)
        masks_array[image_index,0] = mask
        
        masks_array[image_index,0] /= 255.

def augment(X, masks, pool, seed, kwargs):
    '''
    Loads & Augments (if reqiured) a chunk of images
    
    Args:
        X(numpy): a group of images
        masks(numpy): a group of masks (or none)
        pool: for multiprocessing
        kwargs(dictionary): the paramters required for augmentation
    
    Returns:
        void
    '''
    
    # reproducible data augmentation
    random.seed(seed)
    np.random.seed(seed)
    
    X_shape = X.shape
    
    height = X_shape[2]
    width = X_shape[3]
        
    # Create SharedArray for the images and the mask
    images_array_name = str(uuid4())
    images_array = SharedArray.create(images_array_name, [X_shape[0], X_shape[1], height, width], dtype=np.float32)
    
    # Create a shared array for the masks
    masks_array_name = None
    if type(masks) == np.ndarray:
        masks_shape = masks.shape
        masks_array_name = images_array_name+'_mask'
        masks_array = SharedArray.create(masks_array_name, [masks_shape[0], 1, height, width], dtype=np.float32)
    
    
    args = []
    for image_index in range(0,X_shape[0]):
        seed = random.randint(0, 999999)# This ensures that each process is seeded differently. Otherwise, all workers will produce the same augmentation
        image = X[image_index]
        if X_shape[1]>1:
            image = image.transpose((1,2,0))
        else:
            image = image[0]
        mask = None
        if type(masks) == np.ndarray:
            mask = masks[image_index,0]
        args.append((image_index, seed, images_array_name, masks_array_name, image, mask, kwargs))
    
    if pool:# multiprocessing enabled
        pool.map(augment_image, args)
    else:
        for arg_record in args:
            augment_image(arg_record)

    # Create an array and a mask
    X = np.array(images_array)
    SharedArray.delete(images_array_name)
    del images_array
    if type(masks) == np.ndarray:# create the shared array for the masks.
        masks = np.array(masks_array)
        SharedArray.delete(masks_array_name)
        del masks_array
    
    return X, masks
    
if __name__ == '__main__':
    # Create the directory
    writing_directory = './output/logs'
    utils.create_directory(writing_directory)
    
    # Direct the output to a log file and to screen
    loggerFileName = writing_directory+'/'+os.path.basename(__file__).replace('.py','')
    utils.initialize_logger(loggerFileName)
    
    images_paths = joblib.load(config.input_train_images_paths)
    input_directory = config.input_train_data_directory
    output_directory = config.train_directory
    generate_mask = True
    image_height = config.image_height
    image_width = config.image_width
    print 'Creating Training Data'
    create_data(images_paths, input_directory, output_directory, generate_mask, image_height, image_width)
    
    images_paths = joblib.load(config.input_valid_images_paths)
    input_directory = config.input_train_data_directory
    output_directory = config.valid_directory
    generate_mask = True
    image_height = config.image_height
    image_width = config.image_width
    print 'Creating Validation Data'
    create_data(images_paths, input_directory, output_directory, generate_mask, image_height, image_width)
    
    images_paths = joblib.load(config.input_test_images_paths)
    input_directory = config.input_test_data_directory
    output_directory = config.test_directory
    generate_mask = False
    image_height = config.image_height
    image_width = config.image_width
    print 'Creating Testing Data'
    create_data(images_paths, input_directory, output_directory, generate_mask, image_height, image_width)

    train_augmentation_parameters = {
    'augmentation_probability': 1.,
    'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
    'smoothing_probability':.0,'max_smoothing_sigma': 1.,# Random smoothing
    'geometric_transform_probability':.5,
    'height_shift_range':.1, 'width_shift_range':.1,#For example 0.1
    'zoom_range': (0.9,1.1),#(0.9,1.1). To disable use 'None'.
    'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
    'rotation_probability':0., 'rotation_range':10,
    'horizontal_flip_probability':.5,
    'vertical_flip_probability':.5,
    }
    
    images_train = joblib.load(config.train_directory+'/denoised_images.joblib')
    masks_train = joblib.load(config.train_directory+'/masks.joblib')
    pool = None
    
    X_train,y_train = augment(images_train, masks_train, pool, config.seed, train_augmentation_parameters)