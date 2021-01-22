# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:50:10 2020

@author: xinmeng
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from skimage import data, exposure
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters.rank import entropy
from skimage.segmentation import clear_border
from skimage.measure import label, perimeter, find_contours
from skimage.morphology import closing, square, opening, reconstruction, skeletonize, \
                                convex_hull_image, dilation, thin, binary_erosion, disk
from skimage.measure import regionprops, moments, moments_central, moments_hu
from skimage.color import label2rgb, gray2rgb
from skimage.restoration import denoise_tv_chambolle
from skimage.io import imread
from PIL import Image
from scipy.signal import convolve2d, medfilt
import scipy.interpolate as interpolate
from scipy.ndimage.filters import gaussian_filter1d
import numpy.lib.recfunctions as rfn
import pandas as pd
import copy
import os
import plotly.express as px
import sys
import concurrent.futures
import time
from datetime import datetime, date
# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname+'/../')

#----------------------------------TF-2----------------------------------------

from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig
from MaskRCNN.Engine.MaskRCNN import MaskRCNN as modellib
import MaskRCNN.Miscellaneous.visualize as visualize
from ImageAnalysis.ImageProcessing import ProcessImage
#================================================================ProcessImage===============================================
class ProcessImageML():
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        """
        # =============================================================================
        # Initialize the detector instance and load the model.
        # =============================================================================
        """
        # Load configuration file
        # Setup config file
        self.config = cellConfig()
        self.config.LogDir = ''
        self.config.CCoor_STD_DEV = 0.1
        # self.config.WeigthPath = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\SpikingHek.h5"
        self.config.WeigthPath = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\FinalResults\ModelWeights.h5"
        
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
        self.Detector = modellib(self.config, 'inference', model_dir=self.config.LogDir)
        self.Detector.compileModel()
        self.Detector.LoadWeigths(self.config.WeigthPath, by_name=True)
    #%%
    """
    # ======================================================================================================================
    # ************************************  Retrive scanning scheme and read in images. ************************************ 
    # ======================================================================================================================
    """

    def ReadinImgs_Roundstack(self, Nest_data_directory, rowIndex, colIndex):
        """
        Read in images from nest directory.
        
        Parameters
        ----------
        Nest_data_directory : string.
            The directory to folder where the screening data is stored.
        rowIndex, colIndex:
            Row and column index in stage coordinates.
    
        Returns
        -------
        PMT_image_wholetrace_stack : 2-D ndarray or stack of 2-D ndarray.
            Loaded images.
        """
        fileNameList = []
        ImgSequenceNum = 0
        for file in os.listdir(Nest_data_directory):
            if 'PMT_0Zmax' in file and 'R{}C{}'.format(rowIndex, colIndex) in file:
                fileNameList.append(file)
        
        fileNameList.sort(key=lambda x: int(x[x.index('Round')+5:x.index('_Coord')])) # Sort the list according to Round number
#        print(fileNameList)
        
        for eachfile in fileNameList:
            ImgSequenceNum += 1
            img_fileName = os.path.join(Nest_data_directory, eachfile)
            temp_loaded_image = imread(img_fileName, as_gray=False)
            temp_loaded_image = temp_loaded_image[np.newaxis, :, :]
            if ImgSequenceNum == 1:
                PMT_image_wholetrace_stack = temp_loaded_image
            else:
                PMT_image_wholetrace_stack = np.concatenate((PMT_image_wholetrace_stack, temp_loaded_image), axis=0)
                    
        return PMT_image_wholetrace_stack
    
    def retrive_scanning_scheme(self, Nest_data_directory, file_keyword = 'PMT_0Zmax'):
        """
        Return lists that contain round sequence and coordinates strings, like ['Coords1_R0C0', 'Coords2_R0C1500']

        Parameters
        ----------
        Nest_data_directory : string.
            The directory to folder where the screening data is stored.
        file_keyword : string.
            The keyowrd used to search for file name.
        
        Returns
        -------
        RoundNumberList : List.
            List of all round numbers in screening.
        CoordinatesList : List.
            List of all stage coordinates in screening scheme.
        fileNameList: List.
            List of file names strings.
        """
        fileNameList = []
#        ImgSequenceNum = 0
        for file in os.listdir(Nest_data_directory):
            if file_keyword in file:
                fileNameList.append(file)
        
        RoundNumberList = []
        CoordinatesList = []
        for eachfilename in fileNameList:
            # Get how many rounds are there
            try:
                RoundNumberList.append(eachfilename[eachfilename.index('Round'):eachfilename.index('_Grid')])
            except:
                RoundNumberList.append(eachfilename[eachfilename.index('Round'):eachfilename.index('_Coord')])
                
            RoundNumberList = list(dict.fromkeys(RoundNumberList)) # Remove Duplicates
            
            CoordinatesList.append(eachfilename[eachfilename.index('Coord'):eachfilename.index('_PMT')])
            CoordinatesList = list(dict.fromkeys(CoordinatesList))
            
#        print(RoundNumberList, CoordinatesList, fileNameList)
        return RoundNumberList, CoordinatesList, fileNameList
    #%%
    """
    # ================================================================================================================
    # ************************************  Run detection on single image  ************************************* 
    # ================================================================================================================
    """
    def DetectionOnImage(self, Rawimage, axis = None, show_result = False):
        """ 
        Convert image pixel values to unit8 to run on MaskRCNN, and then run MaskRCNN on it.
        """        
        # image = ProcessImage.convert_for_MaskRCNN(Rawimage)
        
        # Run the detection on input image.
        results        = self.Detector.detect([Rawimage])
        
        MLresults      = results[0]
        
        if show_result == True:
            
            # Set class_names = [None,None,None,None] to mute class name display.
            visualize.display_instances(Rawimage, MLresults['rois'], MLresults['masks'], MLresults['class_ids'],
                                            class_names = [None,None,None,None],
                                            centre_coors = MLresults['Centre_coor'], Centre_coor_radius = 2, 
                                            WhiteSpace = (0, 0))#MLresults['class_ids'],MLresults['scores'], 

        if axis != None:
            # If axis is given, draw on axis.
            visualize.display_instances(Rawimage, MLresults['rois'], MLresults['masks'], MLresults['class_ids'],
                                            ['BG'] + self.config.ValidLabels, ax=axis,
                                            centre_coors = MLresults['Centre_coor'], Centre_coor_radius = 2, 
                                            WhiteSpace = (0, 0))#MLresults['class_ids'],MLresults['scores'], 
            # ax.imshow(fig)
       
            return MLresults
        else:
            return MLresults
    """
    # ================================================================================================================
    # ************************************  Organize cell properties dictionary  ************************************* 
    # ================================================================================================================
    """

    def FluorescenceAnalysis(self, folder, round_num, save_mask = True):
        """
        # =============================================================================
        # Given the folder and round number, return a dictionary for the round
        # that contains each scanning position as key and structured array of detailed 
        # information about each identified cell as content.
        #
        #   Returned structured array fields:
        #   - BoundingBox of cell ROI
        #   - Mean intensity of whole cell area
        #   - Mean intensity of cell membrane part
        #   - Contour soma ratio
        # =============================================================================
        
        Parameters
        ----------
        folder : string.
            The directory to folder where the screening data is stored.
        round_num : string.
            The target round number of analysis.
        save_mask: bool.
            Whether to save segmentation masks.
            
        Returns
        -------
        cell_Data : pd.DataFrame.
            Sum of return from func: retrieveDataFromML, for whole round.
        """
        RoundNumberList, CoordinatesList, fileNameList = self.retrive_scanning_scheme(folder, file_keyword = 'Zmax')
        # RoundNumberList, CoordinatesList, fileNameList = self.retrive_scanning_scheme(folder, file_keyword = 'Zfocus')
        
        if not os.path.exists(os.path.join(folder, 'MLimages_{}'.format(round_num))):
            # If the folder is not there, create the folder
            os.mkdir(os.path.join(folder, 'MLimages_{}'.format(round_num))) 
        if not os.path.exists(os.path.join(folder, 'Picked cells')):
            # If the folder is not there, create the folder
            os.mkdir(os.path.join(folder, 'Picked cells')) 
            
        for EachRound in RoundNumberList:
            
            cells_counted_in_round = 0
            
            if EachRound == round_num:
                
                # Start numbering cells at each round
                self.cell_counted_inRound = 0  
                
                for EachCoord in CoordinatesList:
                    
                # =============================================================================
                #             For tag fluorescence:
                # =============================================================================    
                    print(EachCoord)
                    #-------------- readin image---------------
                    for Eachfilename in enumerate(fileNameList):
                        if EachCoord in Eachfilename[1] and EachRound in Eachfilename[1]:
                            if '0Zmax' in Eachfilename[1]:
                                ImgNameInfor = Eachfilename[1][0:len(Eachfilename[1])-14] # get rid of '_PMT_0Zmax.tif' in the name.
                            elif '0Zfocus' in Eachfilename[1]:
                                ImgNameInfor = Eachfilename[1][0:len(Eachfilename[1])-16] # get rid of '_PMT_0Zfocus.tif' in the name.
                            tag_imagefilename = os.path.join(folder, Eachfilename[1])
                    #------------------------------------------
    
                    # =========================================================================
                    #                     USING MASKRCNN...
                    # =========================================================================
                    # Imagepath      = self.Detector._fixPathName(tag_imagefilename)
                    Rawimage     = imread(tag_imagefilename)
                    
#                    if ClearImgBef == True:
#                        # Clear out junk parts to make it esaier for ML detection.
#                        RawimageCleared = self.preProcessMLimg(Rawimage, smallest_size=300, lowest_region_intensity=0.16)
#                    else:
#                        RawimageCleared = Rawimage.copy()
                                        
                    image = ProcessImage.convert_for_MaskRCNN(Rawimage)
                    
                    # Run the detection on input image.
                    results        = self.Detector.detect([image])
                    
                    MLresults      = results[0]
                    
                    if save_mask == True:
                        fig, ax = plt.subplots()
                        # Set class_names = [None,None,None,None] to mute class name display.
                        visualize.display_instances(image, MLresults['rois'], MLresults['masks'], MLresults['class_ids'],
                                                        class_names = [None,None,None,None], ax=ax,
                                                        centre_coors = MLresults['Centre_coor'], Centre_coor_radius = 2, 
                                                        WhiteSpace = (0, 0))#MLresults['class_ids'],MLresults['scores'], 
                        # ax.imshow(fig)
                        fig.tight_layout()
                        # Save the detection image
                        fig_name = os.path.join(folder, 'MLimages_{}\{}.tif'.format(round_num, ImgNameInfor))
                        plt.savefig(fname = fig_name, dpi=200, pad_inches=0.0, bbox_inches='tight')
                    
                    # segmentationImg = Image.fromarray(fig) #generate an image object
                    # segmentationImg.save(os.path.join(folder, 'MLimages_{}\{}.tif'.format(round_num, ImgNameInfor)))#save as tif
                    
                    if self.cell_counted_inRound == 0:
                        cell_Data, self.cell_counted_inRound, total_cells_counted_in_coord = \
                            ProcessImage.retrieveDataFromML(Rawimage, MLresults, str(ImgNameInfor), self.cell_counted_inRound)
                    else:                       
                        Cell_Data_new, self.cell_counted_inRound, total_cells_counted_in_coord = \
                            ProcessImage.retrieveDataFromML(Rawimage, MLresults, str(ImgNameInfor), self.cell_counted_inRound)
                        if len(Cell_Data_new) > 0:
                            cell_Data = cell_Data.append(Cell_Data_new)
                    
                    # Count in total how many flat and round cells are identified.
                    cells_counted_in_round += total_cells_counted_in_coord
                    
                print("Number of round/flat cells in this round: {}".format(cells_counted_in_round))
                
        # Save to excel
        cell_Data.to_excel(os.path.join(os.path.join(folder, round_num + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_CellsProperties.xlsx')))
                
        return cell_Data
                
    
    #%%
    def showPlotlyScatter(self, DataFrame, x_axis, y_axis, saving_directory):
        """
        Display the scatters through interactive library Plotly.

        Parameters
        ----------
        DataFrame : pd.dataframe
            The feed in datafram.
        x_axis : str.
            Name of the field as x-axis.
        y_axis : str.
            Name of the field as y-axis.
        saving_directory : str.
            The directory to save the html file.

        Returns
        -------
        None.

        """
        fig = px.scatter(DataFrame, x = x_axis, y=y_axis, hover_name= DataFrame.index, color= 'Lib_Tag_contour_ratio',
                         hover_data= ['Contour_soma_ratio_Lib', 'Lib_Tag_contour_ratio', 'ImgNameInfor_Lib'], width=1050, height=950)
#        fig.update_layout(hovermode="x")
        fig.write_html(saving_directory, auto_open=True)

    
if __name__ == "__main__":
    import skimage
    # from skimage.io import imread
    # =============================================================================
    
    tag_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-4-08 Archon citrine library 100FOVs\trial_3_library_cellspicked'
    lib_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-7-30 Archon1 comparision 100 FOV\code_test'

    tag_round = 'Round1'
    lib_round = 'Round2'
    
    ProcessML = ProcessImageML()
    # ProcessML.config.WeigthPath = r"C:\MaskRCNN\MaskRCNNGit\MaskRCNN\MaskRCNN\Data\Xin_training\cell20210107T1533\mask_rcnn_cell_0050.h5"
    print(ProcessML.config.WeigthPath)
    # 5.6s for each detection
    img = skimage.io.imread\
    (r"C:\MaskRCNN\MaskRCNNGit\MaskRCNN\MaskRCNN\Data\Xin_training\detection trial\Round1_Grid8_Coords11_R20150C24800_PMT_0Zmax.tif")
    for _ in range(1):    
        starttime = time.time()
        ProcessML.DetectionOnImage(img, show_result = True)
        endtime = time.time()
        print(starttime-endtime)
    
      
    
    
    