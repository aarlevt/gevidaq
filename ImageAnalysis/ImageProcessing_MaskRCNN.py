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
        self.config.WeigthPath = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\FinalResults\ModelWeights.h5'
        
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
    
    def retrive_scanning_scheme(self, Nest_data_directory):
        """
        Return lists that contain round sequence and coordinates strings, like ['Coords1_R0C0', 'Coords2_R0C1500']

        Parameters
        ----------
        Nest_data_directory : string.
            The directory to folder where the screening data is stored.
    
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
            if 'PMT_0Zmax' in file:
                fileNameList.append(file)
        
        RoundNumberList = []
        CoordinatesList = []
        for eachfilename in fileNameList:
            # Get how many rounds are there
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
    def DetectionOnImage(self, Rawimage, axis = None, show_mask=True, show_bbox=True):
        """ 
        Convert image pixel values to unit8 to run on MaskRCNN, and then run MaskRCNN on it.
        """        
        # image = ProcessImage.convert_for_MaskRCNN(Rawimage)
        
        # Run the detection on input image.
        results        = self.Detector.detect([Rawimage])
        
        MLresults      = results[0]
        
        if axis != None:
            # If axis is given, draw on axis.
            visualize.display_instances(image, MLresults['rois'], MLresults['masks'], MLresults['class_ids'],
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
        round_num : int.
            The target round number of analysis.
        save_mask: bool.
            Whether to save segmentation masks.
            
        Returns
        -------
        cell_Data : pd.DataFrame.
            Sum of return from func: retrieveDataFromML, for whole round.
        """
        RoundNumberList, CoordinatesList, fileNameList = self.retrive_scanning_scheme(folder)
        os.mkdir(os.path.join(folder, 'MLimages_{}'.format(round_num))) # Create the folder
        
        for EachRound in RoundNumberList:
            
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
                            ImgNameInfor = Eachfilename[1][0:len(Eachfilename[1])-14] # get rid of '_PMT_0Zmax.tif' in the name.
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
                        cell_Data, self.cell_counted_inRound = ProcessImage.retrieveDataFromML(Rawimage, MLresults, str(ImgNameInfor), self.cell_counted_inRound)
                    else:                       
                        Cell_Data_new, self.cell_counted_inRound = ProcessImage.retrieveDataFromML(Rawimage, MLresults, str(ImgNameInfor), self.cell_counted_inRound)
                        if len(Cell_Data_new) > 0:
                            cell_Data = cell_Data.append(Cell_Data_new)
                    
        return cell_Data
                
    
    #%%
    def MergeDataFrames(self, cell_Data_1, cell_Data_2, method = 'TagLib'):
        """ Merge Data frames based on input methods.
        
        #   'TagLib': Merge tag protein screening round with library screening round.
        #   -In this mode, for each bounding box in the tag round, it will search through every bounding box in library round and find the best match with
        #   -with the most intersection, and then treat them as images from the same cell and merge the two input dataframes.
        
        Parameters
        ----------
        cell_Data_1, cell_Data_2 : pd.DataFrame.
            Input data from two rounds.
        method : String.
            Merge method. 
            'TagLib': for brightness screening.
            
        Returns
        -------
        Cell_DataFrame_Merged : pd.DataFrame.
            Detail information from merging two input dataframe, with bounding boxes from different rounds overlapping above 60% seen as same cell.
        """
        if method == 'TagLib':
            cell_Data_1 = cell_Data_1.add_suffix('_Tag')
            cell_Data_2 = cell_Data_2.add_suffix('_Lib')
            cell_merged_num = 0
            
            print("Start linking cells...")
            # Assume that cell_Data_1 is the tag protein dataframe, for each of the cell bounding box, find the one with the most intersection from library dataframe.
            for index_Data_1, row_Data_1 in cell_Data_1.iterrows():
                # For each flat cell in round
                bounding_box_str_Data_1 = row_Data_1['BoundingBox_Tag']
                ImgNameInforString_Data1 = row_Data_1['ImgNameInfor_Tag']
                # Retrieve boundingbox information
                minr_Data_1 = int(bounding_box_str_Data_1[bounding_box_str_Data_1.index('minr')+4:bounding_box_str_Data_1.index('_maxr')])
                maxr_Data_1 = int(bounding_box_str_Data_1[bounding_box_str_Data_1.index('maxr')+4:bounding_box_str_Data_1.index('_minc')])        
                minc_Data_1 = int(bounding_box_str_Data_1[bounding_box_str_Data_1.index('minc')+4:bounding_box_str_Data_1.index('_maxc')])
                maxc_Data_1 = int(bounding_box_str_Data_1[bounding_box_str_Data_1.index('maxc')+4:len(bounding_box_str_Data_1)])
                
                Area_cell_1 = (maxr_Data_1 - minr_Data_1) * (maxc_Data_1 - minc_Data_1)
                
                intersection_Area_percentage_list = []
                index_list_Data_2 = []
                # Iterate through DataFrame 2 calculating intersection area
                for index_2, row_Data_2 in cell_Data_2.iterrows():
                    ImgNameInforString_Data2 = row_Data_2['ImgNameInfor_Lib']
                    # Search in the same coordinates.
                    if ImgNameInforString_Data2[ImgNameInforString_Data2.index('_R')+1:len(ImgNameInforString_Data2)] == \
                    ImgNameInforString_Data1[ImgNameInforString_Data1.index('_R')+1:len(ImgNameInforString_Data1)]:
                        bounding_box_str_Data_2 = row_Data_2['BoundingBox_Lib']
                        # Retrieve boundingbox information
                        minr_Data_2 = int(bounding_box_str_Data_2[bounding_box_str_Data_2.index('minr')+4:bounding_box_str_Data_2.index('_maxr')])
                        maxr_Data_2 = int(bounding_box_str_Data_2[bounding_box_str_Data_2.index('maxr')+4:bounding_box_str_Data_2.index('_minc')])        
                        minc_Data_2 = int(bounding_box_str_Data_2[bounding_box_str_Data_2.index('minc')+4:bounding_box_str_Data_2.index('_maxc')])
                        maxc_Data_2 = int(bounding_box_str_Data_2[bounding_box_str_Data_2.index('maxc')+4:len(bounding_box_str_Data_2)])                
                        
                        Area_cell_2 = (maxr_Data_2 - minr_Data_2) * (maxc_Data_2 - minc_Data_2)
                        
                        # Overlapping row
                        if minr_Data_2 < maxr_Data_1 and maxr_Data_2 > minr_Data_1:
                            intersection_rowNumber = min((abs(minr_Data_2 - maxr_Data_1), maxr_Data_1 - minr_Data_1)) - max(maxr_Data_1 - maxr_Data_2, 0)
                        else:
                            intersection_rowNumber = 0
                        # Overlapping column
                        if minc_Data_2 < maxc_Data_1 and maxc_Data_2 > minc_Data_1:
                            intersection_colNumber = min((abs(minc_Data_2 - maxc_Data_1), maxc_Data_1 - minc_Data_1)) - max(maxc_Data_1 - maxc_Data_2, 0)
                        else:
                            intersection_colNumber = 0                
            
                        intersection_Area = intersection_rowNumber * intersection_colNumber
                        # Calculate the percentage based on smaller number of intersection over the two.
                        intersection_Area_percentage = min([(intersection_Area / Area_cell_1), (intersection_Area / Area_cell_2)])

                        intersection_Area_percentage_list.append(intersection_Area_percentage)
                        index_list_Data_2.append(index_2)
                
                if len(intersection_Area_percentage_list) > 0:
                    # Link back cells based on intersection area
                    if max(intersection_Area_percentage_list) > 0.6:
                        # If in DataFrame_2 there's a cell that has a overlapping bounding box, merge and generate a new dataframe.
                        Merge_data2_index = index_list_Data_2[intersection_Area_percentage_list.index(max(intersection_Area_percentage_list))]
      
                        Merged_identifiedCell = pd.concat((cell_Data_1.loc[index_Data_1], cell_Data_2.loc[Merge_data2_index]), axis = 0)
                        
                        # Add the lib/tag brightness ratio
                        Lib_Tag_ratio = pd.DataFrame([Merged_identifiedCell.loc['Mean_intensity_in_contour_Lib'] / Merged_identifiedCell.loc['Mean_intensity_in_contour_Tag']],
                                                     index = ['Lib_Tag_contour_ratio'])
                        
                        Merged_identifiedCell = pd.concat((Merged_identifiedCell, Lib_Tag_ratio), axis = 0)
                        Merged_identifiedCell.rename(columns={0:'Cell {}'.format(cell_merged_num)}, inplace=True) # Rename the column name, which is the index name after T.
                         
                        if cell_merged_num == 0:
                            Cell_DataFrame_Merged = Merged_identifiedCell
                        else:
                            Cell_DataFrame_Merged = pd.concat((Cell_DataFrame_Merged, Merged_identifiedCell), axis = 1)
                        cell_merged_num += 1
              
            Cell_DataFrame_Merged = Cell_DataFrame_Merged.T
            print("Cell_DataFrame_Merged.")
            
        return Cell_DataFrame_Merged
    
    
    def FilterDataFrames(self, DataFrame, Mean_intensity_in_contour_thres, Contour_soma_ratio_thres, *args, **kwargs):
        """
        Filter the dataframe based on input numbers.
        
        Parameters
        ----------
        DataFrame : pd.DataFrame.
            Input data.
        Mean_intensity_in_contour_thres : Float.
            Threshold for eliminating dim cells.
        Contour_soma_ratio_thres : Float.
            Threshold for contour soma ratio.
            
        Returns
        -------
        DataFrames_filtered : pd.DataFrame.
            Filtered dataframe.
        """
        DataFrames_filtered = DataFrame[(DataFrame['Mean_intensity_in_contour_Lib'] > Mean_intensity_in_contour_thres) & 
                                        (DataFrame['Contour_soma_ratio_Lib'] > Contour_soma_ratio_thres)]
                
        return DataFrames_filtered
    
    def Sorting_onTwoaxes(self, DataFrame, axis_1, axis_2, weight_1, weight_2):
        """
        Sort the dataframe based on normalized distance calculated from two given axes.
        """
        if axis_1 == "Lib_Tag_contour_ratio" and axis_2 == "Contour_soma_ratio_Lib":
            # Get the min and max on two axes, prepare for next step.
            Contour_soma_ratio_min, Contour_soma_ratio_max = DataFrame.Contour_soma_ratio_Lib.min(), DataFrame.Contour_soma_ratio_Lib.max()
            Lib_Tag_contour_ratio_min, Lib_Tag_contour_ratio_max = DataFrame.Lib_Tag_contour_ratio.min(), DataFrame.Lib_Tag_contour_ratio.max()
            
            DataFrame_sorted = DataFrame.loc[(((DataFrame.Contour_soma_ratio_Lib - Contour_soma_ratio_min) / (Contour_soma_ratio_max - Contour_soma_ratio_min)) ** 2 * weight_2
            + ((DataFrame.Lib_Tag_contour_ratio - Lib_Tag_contour_ratio_min) / (Lib_Tag_contour_ratio_max - Lib_Tag_contour_ratio_min)) **2 * weight_1) \
            .sort_values(ascending=False).index]   
    
        return DataFrame_sorted
    
    def showPlotlyScatter(self, DataFrame, x_axis, y_axis, saving_directory):
        
        fig = px.scatter(DataFrame, x = x_axis, y=y_axis, hover_name= DataFrame.index, color= 'Lib_Tag_contour_ratio',
                         hover_data= ['Contour_soma_ratio_Lib', 'Lib_Tag_contour_ratio', 'ImgNameInfor_Lib'], width=1050, height=950)
#        fig.update_layout(hovermode="x")
        fig.write_html(saving_directory, auto_open=True)

    
if __name__ == "__main__":
    
    import time
    import skimage
    # from skimage.io import imread
    # =============================================================================
    tag_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-4-08 Archon citrine library 100FOVs\trial_3_library_cellspicked'
    lib_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-7-30 Archon1 comparision 100 FOV\code_test'
  #   tag_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-3-6 Archon brightness screening\NovArch library'

    tag_round = 'Round1'
    lib_round = 'Round2'
    
    
    ProcessML = ProcessImageML()

#    cell_Data_1 = ProcessML.FluorescenceAnalysis(lib_folder, tag_round)
    # cell_Data_2 = ProcessML.FluorescenceAnalysis(lib_folder, tag_round)
    
    # 5.6s for each detection
    img = skimage.io.imread(r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-7-30 Archon1 comparision 100 FOV\code_test\Round1_Coords1_R0C0_PMT_0Zmax.tif")
    for _ in range(3):    
        starttime = time.time()
        ProcessML.DetectionOnImage(img)
        endtime = time.time()
        print(starttime-endtime)
    