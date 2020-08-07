# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:02:43 2020

@author: meppenga
"""

from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig
from MaskRCNN.Engine.MaskRCNN import MaskRCNN as modellib
import MaskRCNN.Miscellaneous.visualize as visualize
import matplotlib.pyplot as plt



# Setup config file
config = cellConfig()
config.CCoor_STD_DEV   = 0.1
config.IMAGE_MIN_SCALE = 2.0
config.WeigthPath      = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\FinalResults\ModelWeights.h5'
# Return mini mask (all mask are 28 by 28 pixels). One can use CreateFullMask
# from the utils to resize the mask to the same shape as the image. This is not
# recommended as it is time consuming and many operations can be done using the
# small mask and it bounding box coordinates and the ReshapeMask2BBox function
# from the utils.
config.RETURN_MINI_MASK = True

# These four setting use the old configurations. The results are slightly different due to scaling.
# However the new version will prevent any OOM erros from occuring
# config.IMAGE_MIN_SCALE = 2.0
# config.IMAGE_MIN_DIM = 1024
# config.IMAGE_MAX_DIM = 1024
# config.IMAGE_RESIZE_MODE = "pad64"

# If you use spiking hek images, uncomment the next lines, and use:
# Image = skimage.transform.resize(Image,[1024,1024],preserve_range=True).astype(Image.dtype)
# config.IMAGE_RESIZE_MODE = "square"
# config.IMAGE_MIN_DIM = 1024
# config.IMAGE_MAX_DIM = 1024
# config.IMAGE_MIN_SCALE = 1



# Create model
Predictor = modellib(config, 'inference', model_dir=config.LogDir)
Predictor.compileModel()
Predictor.LoadWeigths(config.WeigthPath, by_name=True)

# Run detection on image
Image = plt.imread(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\Tag\Validation\Round1_Coords2_R0C1650_PMT_0Zmax.png')
R = Predictor.detect([Image])
Result= R[0]

# show Result (The centre coordinates does not make any sense for this image 
# you can omit them by setting Result['Centre_coor'] = [])
fig, ax = plt.subplots(1,num='Example inference')
ax.clear()
visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'],
                            ['BG'] + config.ValidLabels, Result['scores'], ax=ax,
                            centre_coors=Result['Centre_coor'], Centre_coor_radius = 2,TextSize=5)
