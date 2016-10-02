# -*- coding: utf-8 -*-

import cv2

############################################################################
# General Configurations
############################################################################

image_height, image_width = 96, 128
original_image_height, original_image_width = 420, 580

input_train_data_directory = './data/train'
input_test_data_directory = './data/test'
splits_directory = './output/splits'
seed=26
test_size = .1

input_train_images_paths = splits_directory+'/'+'images_paths_train.joblib'
input_valid_images_paths = splits_directory+'/'+'images_paths_valid.joblib'
input_test_images_paths = splits_directory+'/'+'images_paths_test.joblib'

# The train, valid, test chunks
train_directory = './output/train'
valid_directory = './output/valid'
test_directory = './output/test'

############################################################################
# Segmentation Models Configurations
############################################################################

# Model ID: seg1
sr_seg1_denoised_1ch_FCN_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+1,
}

# Model ID: seg1
sr_seg1_denoised_1ch_FCN_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':24, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':110,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_seg1_denoised_1ch_FCN_model_1_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) FCN Model 1 Denoised Image with 1 channel Input. 
2) Cross
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'FCN_Model1', 
'task': 'segment', 
'train_positive_only': True, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: seg2
sr_seg2_denoised_1ch_FCN_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+2,
}

# Model ID: seg2
sr_seg2_denoised_1ch_FCN_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':24, 
'nb_epochs':120, 
'max_patience': 20,
'start_fine_tuning_at_epoch':100,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_seg2_denoised_1ch_FCN_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) FCN Model 2 Denoised Image with 1 channel Input
2) cross
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'FCN_Model2', 
'task': 'segment', 
'train_positive_only': True, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}


# Model ID: seg3
sr_seg3_denoised_1ch_FCN_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+3,
}

# Model ID: seg3
sr_seg3_denoised_1ch_FCN_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':24, 
'nb_epochs':120, 
'max_patience': 20,
'start_fine_tuning_at_epoch':100,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_seg3_denoised_1ch_FCN_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) FCN Model 3 Denoised Image with 1 channel Input
2) Cross
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'FCN_Model3', 
'task': 'segment', 
'train_positive_only': True, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: seg4
sr_seg4_denoised_1ch_FCN_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+4,
}

# Model ID: seg4
sr_seg4_denoised_1ch_FCN_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':24, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':130,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_seg4_denoised_1ch_FCN_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) FCN VGG16 Denoised Image with 1 channel Input
2) Cross
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'FCN_VGG16', 
'task': 'segment', 
'train_positive_only': True, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: seg5
sr_seg5_denoised_1ch_FCN_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+5,
}

# Model ID: seg5
sr_seg5_denoised_1ch_FCN_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':24, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':90,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_seg5_denoised_1ch_FCN_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) FCN Model 4 Denoised Image with 1 channel Input
2) Cross
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'FCN_Model4', 
'task': 'segment', 
'train_positive_only': True, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

############################################################################
# Segmentation Ensemble Configurations
############################################################################

# Ensemble ID: segmentation ensemble
segment_ensemble_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
'seed':seed+6,
}

# Ensemble ID: segmentation ensemble
segment_ensemble_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'segment_ensemble',
'experiment_description': '''
1) Segment Ensemble
''',
'pool': None,
'load_weights_from': None, 'model': 'segment_ensemble', 'task': 'segment', 'train_positive_only': False, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': [
'sr_seg1_denoised_1ch_FCN_model_1_cross_relu_initCH32_lowAUG_noBN',
'sr_seg2_denoised_1ch_FCN_model_2_cross_relu_initCH32_lowAUG_noBN',
'sr_seg3_denoised_1ch_FCN_model_3_cross_relu_initCH32_lowAUG_noBN',
'sr_seg4_denoised_1ch_FCN_vgg16_cross_relu_initCH32_lowAUG_noBN',
'sr_seg5_denoised_1ch_FCN_model_4_cross_relu_initCH32_lowAUG_noBN',
]
}

############################################################################
# Is_Positive Models Configurations
############################################################################
# Model ID: 1
sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+7,
}

# Model ID: 1
sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':95,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 1 Original Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model1', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
}

# Model ID: 2
sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+8,
}

# Model ID: 2
sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':200, 
'max_patience': 20,
'start_fine_tuning_at_epoch':190,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 2 Original Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model2', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
}


# Model ID: 3
sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+9,
}

# Model ID: 3
sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':140,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 3 Original Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model3', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
}

# Model ID: 4
sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+10,
}

# Model ID: 4
sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':120,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) VGG16 Original Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'VGG16', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
}

# Model ID: 5
sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+11,
}

# Model ID: 5
sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':110, 
'max_patience': 20,
'start_fine_tuning_at_epoch':105,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 4 Original Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model4', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
}

############################################################################

# Model ID: 6
sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+12,
}

# Model ID: 6
sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':95,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 1 Denoised Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model1', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: 7
sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+13,
}

# Model ID: 7
sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':200, 
'max_patience': 20,
'start_fine_tuning_at_epoch':190,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 2 Denoised Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model2', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}


# Model ID: 8
sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+14,
}

# Model ID: 8
sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':140,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 3 Denoised Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model3', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: 9
sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+15,
}

# Model ID: 9
sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':120,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) VGG16 Denoised Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'VGG16', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

# Model ID: 10
sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+16,
}

# Model ID: 10
sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':110, 
'max_patience': 20,
'start_fine_tuning_at_epoch':105,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 4 Denoised Image with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model4', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib'],
}

############################################################################

# Model ID: 16
sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+17,
}

# Model ID: 16
sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':95,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params',
'experiment_description': '''
1) Model 1 Denoised Image with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model1', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib',train_directory+'/segment_ensemble.joblib',],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib',valid_directory+'/segment_ensemble.joblib',],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib',test_directory+'/segment_ensemble.joblib',],
}

# Model ID: 17
sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+18,
}

# Model ID: 17
sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':200, 
'max_patience': 20,
'start_fine_tuning_at_epoch':190,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 2 Denoised Image with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model2', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib',train_directory+'/segment_ensemble.joblib',],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib',valid_directory+'/segment_ensemble.joblib',],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib',test_directory+'/segment_ensemble.joblib',],
}


# Model ID: 18
sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+19,
}

# Model ID: 18
sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':140,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 3 Denoised Image with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model3', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib',train_directory+'/segment_ensemble.joblib',],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib',valid_directory+'/segment_ensemble.joblib',],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib',test_directory+'/segment_ensemble.joblib',],
}

# Model ID: 19
sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+20,
}

# Model ID: 19
sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':120,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) VGG16 Denoised Image with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'VGG16', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib',train_directory+'/segment_ensemble.joblib',],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib',valid_directory+'/segment_ensemble.joblib',],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib',test_directory+'/segment_ensemble.joblib',],
}

# Model ID: 20
sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+21,
}

# Model ID: 20
sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':110, 
'max_patience': 20,
'start_fine_tuning_at_epoch':105,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 4 Denoised Image with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model4', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/denoised_images.joblib',train_directory+'/segment_ensemble.joblib',],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/denoised_images.joblib',valid_directory+'/segment_ensemble.joblib',],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/denoised_images.joblib',test_directory+'/segment_ensemble.joblib',],
}

############################################################################

# Model ID: 21
sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+22,
}

# Model ID: 21
sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':95,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 1 Denoised ROI with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model1', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi'],
}

# Model ID: 22
sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+23,
}

# Model ID: 22
sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':200, 
'max_patience': 20,
'start_fine_tuning_at_epoch':190,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 2 Denoised ROI with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model2', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi'],
}


# Model ID: 23
sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+24,
}

# Model ID: 23
sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':140,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 3 Denoised ROI with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model3', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi'],
}

# Model ID: 24
sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+25,
}

# Model ID: 24
sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':120,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) VGG16 Denoised ROI with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'VGG16', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi'],
}

# Model ID: 25
sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+26,
}

# Model ID: 25
sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':110, 
'max_patience': 20,
'start_fine_tuning_at_epoch':105,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 4 Denoised ROI with 1 channel Input
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model4', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi'],
}

############################################################################

# Model ID: 26
sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+27,
}

# Model ID: 26
sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':100, 
'max_patience': 20,
'start_fine_tuning_at_epoch':95,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 1 Denoised ROI with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model1', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi', train_directory+'/predicted_mask_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi', valid_directory+'/predicted_mask_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi', test_directory+'/predicted_mask_roi'],
}

# Model ID: 27
sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+28,
}

# Model ID: 27
sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':200, 
'max_patience': 20,
'start_fine_tuning_at_epoch':190,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 2 Denoised ROI with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model2', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi', train_directory+'/predicted_mask_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi', valid_directory+'/predicted_mask_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi', test_directory+'/predicted_mask_roi'],
}


# Model ID: 28
sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+29,
}

# Model ID: 28
sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':150, 
'max_patience': 20,
'start_fine_tuning_at_epoch':140,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 3 Denoised ROI with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model3', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi', train_directory+'/predicted_mask_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi', valid_directory+'/predicted_mask_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi', test_directory+'/predicted_mask_roi'],
}

# Model ID: 29
sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+30,
}

# Model ID: 29
sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':130, 
'max_patience': 20,
'start_fine_tuning_at_epoch':120,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) VGG16 Denoised ROI with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'VGG16', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi', train_directory+'/predicted_mask_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi', valid_directory+'/predicted_mask_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi', test_directory+'/predicted_mask_roi'],
}

# Model ID: 30
sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN_aug_params = {
'augmentation_probability': 1.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5, 'vertical_flip_probability':.5,
'seed':seed+31,
}

# Model ID: 30
sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 2,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,

'initial_channels': 32, 
'dropout_rate': 0.5, 
'leakiness':-1,
'use_batch_normalization': False,
'bn_axis': 1, 
'strides': (1,1), 
'atrous_rate': None,
'border_mode': 'same',

'learning_rate': 0.00005, 
'loss': 'binary_crossentropy',

'batch_size':32, 
'nb_epochs':110, 
'max_patience': 20,
'start_fine_tuning_at_epoch':105,
'start_augmenting_at_epoch':0,

'number_of_rows':6, 
'number_of_cols': 9,

'experiment_name' : 'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN',
'experiment_description': '''
1) Model 4 Denoised ROI with 1 channel Input
2) 2ch
''',
'pool': None,

'load_weights_from': None, 
'architecture': 'Model4', 
'task': 'is_positive', 
'train_positive_only': False, 

'images_train_dirs': [train_directory+'/original_denoised_roi', train_directory+'/predicted_mask_roi'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/original_denoised_roi', valid_directory+'/predicted_mask_roi'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/original_denoised_roi', test_directory+'/predicted_mask_roi'],
}


############################################################################
# Create an Ensemble of Is_Positive Models
############################################################################

# Ensemble ID: initial
is_positive_ensemble_initial_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial
is_positive_ensemble_initial_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
#
#
'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
#
'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

#
'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial3
is_positive_ensemble_initial3_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial3
is_positive_ensemble_initial3_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial3',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,


}
}

# Ensemble ID: initial4
is_positive_ensemble_initial4_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial4
is_positive_ensemble_initial4_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial4',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial5
is_positive_ensemble_initial5_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial5
is_positive_ensemble_initial5_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial5',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
}
}

# Ensemble ID: initial6
is_positive_ensemble_initial6_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial6
is_positive_ensemble_initial6_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial6',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial6', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial7
is_positive_ensemble_initial7_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial7
is_positive_ensemble_initial7_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial7',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial6', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial8
is_positive_ensemble_initial8_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial8
is_positive_ensemble_initial8_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial8',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial9
is_positive_ensemble_initial9_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial9
is_positive_ensemble_initial9_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial9',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial10
is_positive_ensemble_initial10_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial10
is_positive_ensemble_initial10_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial10',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial11
is_positive_ensemble_initial11_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial11
is_positive_ensemble_initial11_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial11',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial12
is_positive_ensemble_initial12_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial12
is_positive_ensemble_initial12_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial12',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {


'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
}
}

# Ensemble ID: initial13
is_positive_ensemble_initial13_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial13
is_positive_ensemble_initial13_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial13',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
}
}

# Ensemble ID: initial14
is_positive_ensemble_initial14_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial14
is_positive_ensemble_initial14_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial14',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble_initial', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {
'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial15
is_positive_ensemble_initial15_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial15
is_positive_ensemble_initial15_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial15',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,


}
}

# Ensemble ID: initial16
is_positive_ensemble_initial16_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial16
is_positive_ensemble_initial16_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial16',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial17
is_positive_ensemble_initial17_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial17
is_positive_ensemble_initial17_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial17',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial18
is_positive_ensemble_initial18_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial18
is_positive_ensemble_initial18_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial18',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial19
is_positive_ensemble_initial19_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial19
is_positive_ensemble_initial19_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial19',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,


}
}

# Ensemble ID: initial20
is_positive_ensemble_initial20_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial20
is_positive_ensemble_initial20_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial20',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial21
is_positive_ensemble_initial21_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial21
is_positive_ensemble_initial21_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial21',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}

# Ensemble ID: initial22
is_positive_ensemble_initial22_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial22
is_positive_ensemble_initial22_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial22',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,


}
}

# Ensemble ID: initial23
is_positive_ensemble_initial23_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

# Ensemble ID: initial23
is_positive_ensemble_initial23_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble_initial23',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

}
}



# Ensemble ID: is positive ensemble
is_positive_ensemble_aug_params = {
'augmentation_probability': 0.0,
'equalize_historgram_probability':.0 ,'invert_image_probability':.0,
'smoothing_probability':0,'max_smoothing_sigma': 1.,# Random smoothing
'geometric_transform_probability':.5,
'height_shift_range':.2, 'width_shift_range':.2,#For example 0.1
'zoom_range': (0.9,1.11),#(0.9,1.1). To disable use 'None'.
'border_mode': cv2.BORDER_CONSTANT,'border_value' : 0,
'rotation_probability':0.5, 'rotation_range':10,
'horizontal_flip_probability':.5,
'vertical_flip_probability':.5,
}

is_positive_ensemble_params = {
'img_rows': image_height,
'img_cols': image_width,
'img_depth': 1,
'mask_depth': 1,
'original_image_height': original_image_height, 
'original_image_width': original_image_width,
'initial_channels': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'leakiness':-1,'use_batch_normalization': False,'loss': 'binary_crossentropy',
'batch_size':32, 'nb_epochs':50, 'max_patience': 5,'start_fine_tuning_at_epoch':40,
'number_of_rows':6, 'number_of_cols': 9,
'experiment_name' : 'is_positive_ensemble',
'experiment_description': '''
1) Ensemble Stage_2
''',
'pool': None,
'load_weights_from': None, 'model': 'is_positive_ensemble', 'task': 'is_positive', 'train_positive_only': True, 
'images_train_dirs': [train_directory+'/images.joblib'],
'masks_train_dirs': [train_directory+'/masks.joblib'],
'images_valid_dirs': [valid_directory+'/images.joblib'],
'masks_valid_dirs': [valid_directory+'/masks.joblib'],
'images_test_dirs': [test_directory+'/images.joblib'],
'models_to_be_ensembled': {

'sr_1_original_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_2_original_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_3_original_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_4_original_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_5_original_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_6_denoised_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_7_denoised_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_8_denoised_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_9_denoised_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_10_denoised_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_16_denoised_2ch_model_1_cross_relu_initCH32_lowAUG_noBN_params':1,
'sr_17_denoised_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_18_denoised_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_19_denoised_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_20_denoised_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_21_denoised_patch_1ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_22_denoised_patch_1ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_23_denoised_patch_1ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_24_denoised_patch_1ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_25_denoised_patch_1ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,

'sr_26_denoised_patch_2ch_model_1_cross_relu_initCH32_lowAUG_noBN':1,
'sr_27_denoised_patch_2ch_model_2_cross_relu_initCH32_lowAUG_noBN':1,
'sr_28_denoised_patch_2ch_model_3_cross_relu_initCH32_lowAUG_noBN':1,
'sr_29_denoised_patch_2ch_vgg16_cross_relu_initCH32_lowAUG_noBN':1,
'sr_30_denoised_patch_2ch_model_4_cross_relu_initCH32_lowAUG_noBN':1,
#
'is_positive_ensemble_initial':1,
'is_positive_ensemble_initial3':1,
'is_positive_ensemble_initial4':1,
'is_positive_ensemble_initial5':1,
'is_positive_ensemble_initial6':1,
'is_positive_ensemble_initial7':1,
'is_positive_ensemble_initial8':1,
'is_positive_ensemble_initial9':1,
'is_positive_ensemble_initial10':1,
'is_positive_ensemble_initial11':1,
'is_positive_ensemble_initial12':1,
'is_positive_ensemble_initial13':1,
'is_positive_ensemble_initial14':1,
'is_positive_ensemble_initial15':1,
'is_positive_ensemble_initial16':1,
'is_positive_ensemble_initial17':1,
'is_positive_ensemble_initial18':1,
'is_positive_ensemble_initial19':1,
'is_positive_ensemble_initial20':1,
'is_positive_ensemble_initial21':1,
'is_positive_ensemble_initial22':1,
'is_positive_ensemble_initial23':1,

}
}
