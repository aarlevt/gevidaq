# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:02:43 2020

@author: meppenga
"""
from MaskRCNN.Configurations.ConfigFileInference import cellConfig
from CustomDataLoader import CellDataSetLoader
from MaskRCNN.Engine.MaskRCNN import MaskRCNN as modellib
import os
import MaskRCNN.Miscellaneous.visualize as visualize
import matplotlib.pyplot as plt

def getLatestWeight(path, name='cell'):
    """Find lates weight file. It search for weight file within folders which 
    include the input name" in it name"""
    files = []
    for file in os.listdir(path):
        if name in file:
            files.append(os.path.join(path, file))
    
    path =  max(files, key=os.path.getctime)
    files = []
    for file in os.listdir(path):
        if '.h5' in file:
            files.append(os.path.join(path, file))
    if len(files) == 0:
        raise Exception('Could not find any weigth file which has "'+name+'" in its folder name')
    return max(files, key=os.path.getctime)


# Setup config file
config = cellConfig()
config.UseRotation = False  
config.UseMirror   = False 
# config.CCoor_STD_DEV = 0.10
config.ValidLabels = ['Square','Circle','Else']
config.NUM_CLASSES = 1+3
config.LogDir = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Examples'

# get latest trained weights
weight_file = getLatestWeight(config.LogDir, config.NAME) # Okay
print('Weigth file: '+weight_file)

# Create data loader
dataset_Val = CellDataSetLoader(config)
dataset_Val.addCellImage(r'C:\MaskRCNN\MaskRCNNGit\Data\DMD Registration Data\Images','Validation', config.UseSubFoldersVal)
dataset_Val.prepare()

# Create model
Predictor = modellib(config, 'inference', model_dir=config.LogDir)
Predictor.compileModel()
Predictor.LoadWeigths(weight_file, by_name=True)

# Inference on first image
Image = dataset_Val.load_image(0)
R = Predictor.detect([Image])
Result= R[0]

# show Result
fig, ax = plt.subplots(1,num='Example inference')
ax.clear()
visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'],
                            ['BG'] + config.ValidLabels, Result['scores'], centre_coors=Result['Centre_coor'], ax=ax)
