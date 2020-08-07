# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:07:46 2020

@author: meppenga
"""

from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig
from MaskRCNN.Miscellaneous.GUISelectCell import CellGui
# config = Config()

if __name__ == "__main__": 
    # from ConfigFileInferenceOld import cellConfig
    config = cellConfig()
    config.IMAGE_RESIZE_MODE = "square"
    config.IMAGE_MIN_DIM = 512
    config.IMAGE_MAX_DIM = 1024
    config.IMAGE_MIN_SCALE = 2.0
    config.LogDir = ''
    config.WeigthPath = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Training\Results\cell20200723T2108\mask_rcnn_cell_0002.h5' 
    config.WeigthPath = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Training\Results\cell20200723T2227\mask_rcnn_cell_0050.h5'
    config.WeigthPath = r'C:\MaskRCNN\MaskRCNNGit\Data\TrainingWeightFiles\cell20200529T1620\mask_rcnn_cell_0064.h5'
    config.WeigthPath = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\FinalResults\ModelWeights.h5'
    GUI = CellGui(config)
    # GUI.RunDetection(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\Tag\Validation\Round1_Coords2_R0C1650_PMT_0Zmax.png')
    GUI.RunDetection(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\2020-4-08 Archon citrine library 100FOVstrial_3_library_cellspicked\Training\Round1_Coords3_R0C3300_PMT_0Zmax.png')     
 