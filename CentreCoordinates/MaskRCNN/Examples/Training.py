# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:32:36 2020

@author: meppenga
"""

from MaskRCNN.Configurations.ConfigFileTraining import cellConfig
from CustomDataLoader import CellDataSetLoader
from MaskRCCn.Engine.MaskRCNN import MaskRCNN as modellib

config = cellConfig()
config.UseRotation = True  
config.UseMirror   = True 
config.CentreExist = False
config.CCoor_STD_DEV = 0.10
# config.ValidLabels = ['Square','Circle','Else']
config.NUM_CLASSES = 1+3
config.STEPS_PER_EPOCH = 200
config.VALIDATION_STEPS = 50
config.LEARNING_RATE  = 0.0001



config.LogDir = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Training\Results'
weight_file   = r'C:\MaskRCNN\MaskRCNNGit\Data\TrainingWeightFiles\cell20200529T1620\mask_rcnn_cell_0064.h5' 
# weight_file   = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Training\Results\cell20200723T2108\mask_rcnn_cell_0006.h5' 



dataset_train = CellDataSetLoader(config)
dataset_train.addCellImage(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2','Training', config.UseSubFoldersTrain)
dataset_train.prepare()

dataset_Val = CellDataSetLoader(config)
dataset_Val.addCellImage(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2','Validation', config.UseSubFoldersVal)
dataset_Val.prepare()


testTrainer = modellib(config,'training', model_dir=config.LogDir)
testTrainer.compileModel()
testTrainer.LoadWeigths(weight_file, by_name=True)
testTrainer.Train(dataset_train, dataset_Val,epochs=50)


