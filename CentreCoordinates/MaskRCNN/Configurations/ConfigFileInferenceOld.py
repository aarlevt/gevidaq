# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:18:34 2020

@author: meppenga
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:21:35 2020

@author: Martijn Eppenga
"""

import numpy as np
from MaskRCNN.Configurations.Config import Config


class cellConfig(Config):

    
    NAME = "cell"
    
    ValidLabels                = ['DeadCell','RoundCell','FlatCell']
    WeigthPath = r'C:\MaskRCNN\MaskRCNNGit\Data\TrainingWeightFiles\cell20200529T1620\mask_rcnn_cell_0064.h5'

    
    # For Training
    TrainDirectories           = ['M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/ML images/NewAnnotation']
    UseSubFoldersTrain         = True
    ValidationDirectories      = ['M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/ML images/NewAnnotation']
    UseSubFoldersVal           = True
    ExludeWeights              = None
    UseRotation                = True
    RotationAngles             = [0,90,180,270]

    
    
    # For inference
    RemoveDoubleDetectedCells  = True
    DoubleDetectedThersholdIoU = 0.6
    
    CCoor_STD_DEV = 0.1

    # Adjust to GPU memory
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1+3
    STEPS_PER_EPOCH = 45
    VALIDATION_STEPS = 5

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5
    #DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    #BACKBONE = "resnet50"
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024#512#1024
    IMAGE_MAX_DIM = 1024#512#1024
    IMAGE_MIN_SCALE = 1

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64,128,256)#16/2, 32/2, 64/2,128/2,256/2) for non resize. (32, 64, 128, 256, 512)
 
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    #RPN_NMS_THRESHOLD = 0.9
    RPN_NMS_THRESHOLD=0.99

    # How many anchors per image to use for RPN training
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([126,126,126])
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MINI_MASK_SHAPE = (100,100)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    #TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    #MAX_GT_INSTANCES = 200
    MAX_GT_INSTANCES = 500
    # Max number of final detections per image
    #DETECTION_MAX_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 1000
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7