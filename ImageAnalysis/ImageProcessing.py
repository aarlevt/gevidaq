# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:46:47 2020

@author: xinmeng
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import time

from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters.rank import entropy
from skimage.segmentation import clear_border
from skimage.measure import label, perimeter, find_contours
from skimage.morphology import closing, square, opening, reconstruction, skeletonize, convex_hull_image, dilation, thin, binary_erosion, disk, binary_dilation
from skimage.measure import regionprops, moments, moments_central, moments_hu
from skimage.draw import line, polygon2mask, polygon_perimeter
from skimage.color import label2rgb, gray2rgb
from skimage.restoration import denoise_tv_chambolle
from skimage.io import imread
from skimage.transform import rotate, resize
from scipy.signal import convolve2d, medfilt
import scipy.interpolate as interpolate
from scipy.ndimage.filters import gaussian_filter1d
import numpy.lib.recfunctions as rfn
import copy
import os
import pandas as pd
# import plotly.express as px

#================================================================ProcessImage============================================================
class ProcessImage():
    #%%
    """
    # ==========================================================================================================================================================
    # ************************************  Retrive scanning scheme and read in images. ************************************ 
    # ==========================================================================================================================================================
    """

    def ReadinImgs_Roundstack(Nest_data_directory, rowIndex, colIndex):
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
            temp_loaded_image = imread(img_fileName, as_gray=True)
            temp_loaded_image = temp_loaded_image[np.newaxis, :, :]
            if ImgSequenceNum == 1:
                PMT_image_wholetrace_stack = temp_loaded_image
            else:
                PMT_image_wholetrace_stack = np.concatenate((PMT_image_wholetrace_stack, temp_loaded_image), axis=0)
                    
        return PMT_image_wholetrace_stack
    
    def retrive_scanning_scheme(Nest_data_directory):
        """
        # =============================================================================
        # Return lists that contain round sequence and coordinates strings, like ['Coords1_R0C0', 'Coords2_R0C1500']
        # =============================================================================
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
            
#        print(CoordinatesList)
        return RoundNumberList, CoordinatesList, fileNameList

    #%%
    """           
    # ==========================================================================================================================================================
    # ************************************    Individual image processing    ************************************    
    # ==========================================================================================================================================================
    """
    
    def generate_mask(imagestack, openingfactor, closingfactor, binary_adaptive_block_size):
        """
        Return a rough binary mask generated from single image or first image of the stack using adaptive thresholding.
        
        Parameters
        ----------
        imagestack : stack of 2-D ndarray.
            The directory to folder where the screening data is stored.
        openingfactor:
            Degree of morphology opening operation on adaptive thresholded input image.
        closingfactor:
            Degree of morphology closing operation on adaptive thresholded input image.
        binary_adaptive_block_size:
            Block size applied for adaptive thresholding.
            
        Returns
        -------
        RegionProposal_Mask : 2-D ndarray.
            Binary mask for cells.
        RegionProposal_ImgInMask : 2-D ndarray.
            Denoised image with cell binary mask imposed.
        """
        if imagestack.ndim == 3:
            template_image = imagestack[0,:,:] # Get the first image of the stack to generate the mask for Region Proposal
        elif imagestack.ndim == 2:
            template_image = imagestack
        
        template_image = denoise_tv_chambolle(template_image, weight=0.01) # Denoise the image.
        # -----------------------------------------------Adaptive thresholding-----------------------------------------------
#        block_size = binary_adaptive_block_size#335
        AdaptiveThresholding = threshold_local(template_image, binary_adaptive_block_size, offset=0)
        BinaryMask = template_image >= AdaptiveThresholding
        OpeningBinaryMask = opening(BinaryMask, square(int(openingfactor)))
        RegionProposal_Mask = closing(OpeningBinaryMask, square(int(closingfactor)))
        
        RegionProposal_ImgInMask = RegionProposal_Mask*template_image
        
        return RegionProposal_Mask, RegionProposal_ImgInMask
    
    
    def Region_Proposal(image, RegionProposalMask, smallest_size, biggest_size, lowest_region_intensity, Roundness_thres, DeadPixelPercentageThreshold,
                        contour_thres, contour_dilationparameter, cell_region_opening_factor, cell_region_closing_factor):
        """
        # =============================================================================
        # Based on tag fluorescence image, generate region proposal bounding box.
        # -- RegionProposalMask: the binary mask for region iterative analysis.
        # -- smallest_size/biggest_size: cells size out of this range are ignored.
        # -- lowest_region_intensity: cells with mean region intensity below this are ignored.
        # -- contour_thres: threshold for contour recognizition.
        # -- Roundness_thres: Roundness above this are ignored.
        # -- DeadPixelPercentageThreshold: Percentage of saturated pixels.
        # -- contour_dilationparameter: the dilation degree applied when doing inward contour dilation for thicker menbrane area.
        # -- cell_region_opening_factor: degree of opening operation on individual cell mask.
        # -- cell_region_closing_factor: degree of closing operation on individual cell mask.
        # =============================================================================
        """
        cleared = RegionProposalMask.copy()
        clear_border(cleared)
        # label image regions, prepare for regionprops
        label_image = label(cleared)       
        dtype = [('BoundingBox', 'U32'), ('Mean intensity', float), ('Mean intensity in contour', float), ('Contour soma ratio', float), ('Roundness', float)]
        CellSequenceInRegion = 0
        dirforcellprp = {}
        show_img = False
        if show_img == True:
            plt.figure()
            fig_showlabel, ax_showlabel = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax_showlabel.imshow(image)#Show the first image
        for region in regionprops(label_image,intensity_image = image): 
            
            # skip small images
            if region.area > smallest_size and region.mean_intensity > lowest_region_intensity and region.area < biggest_size:

                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                boundingbox_info = 'minr{}_minc{}_maxr{}_maxc{}'.format(minr, minc, maxr, maxc)
                bbox_area = (maxr-minr)*(maxc-minc)
                # Based on the boundingbox for each cell from first image in the stack, raw image of slightly larger region is extracted from each round.
                RawRegionImg = image[max(minr-4,0):min(maxr+4, image[0].shape[0]), max(minc-4,0):min(maxc+4, image[0].shape[0])] # Raw region image 
        
                RawRegionImg_for_contour = RawRegionImg.copy()
                
                #---------Get the cell filled mask-------------
                filled_mask_bef, MeanIntensity_Background = ProcessImage.get_cell_filled_mask(RawRegionImg = RawRegionImg, region_area = bbox_area*0.2, 
                                                                                                      cell_region_opening_factor = cell_region_opening_factor, 
                                                                                                      cell_region_closing_factor = cell_region_closing_factor)

                filled_mask_convolve2d = ProcessImage.smoothing_filled_mask(RawRegionImg, filled_mask_bef = filled_mask_bef, region_area = bbox_area*0.2, threshold_factor = 1.1)

                # Find contour along filled image
                contour_mask_thin_line = ProcessImage.contour(filled_mask_convolve2d, RawRegionImg_for_contour.copy(), contour_thres) 

                # after here intensityimage_intensity is changed from contour labeled with number 5 to binary image
                contour_mask_of_cell = ProcessImage.inward_mask_dilation(contour_mask_thin_line.copy() ,filled_mask_convolve2d, contour_dilationparameter)
                
                #                    Calculate Roundness
                #--------------------------------------------------------------
                filled_mask_area = len(np.where(filled_mask_convolve2d == 1)[0])
                contour_mask_perimeter = len(np.where(contour_mask_thin_line == 1)[0])
                Roundness = 4*3.1415*filled_mask_area/contour_mask_perimeter**2
#                print('Roundness: {}'.format(4*3.1415*filled_mask_area/contour_mask_perimeter**2))
                
                #                    Calculate central moments
                #--------------------------------------------------------------
#                M = moments(filled_mask_convolve2d)
#                centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
#                Img_moments_central = moments_central(filled_mask_convolve2d, centroid, order=4)
##                print(Img_moments_central)
#                Img_moments_hu = moments_hu(Img_moments_central/np.amax(Img_moments_central))
#                
#                # Log scale hu moments
#                for EachMoment in range(len(Img_moments_hu)):
#                    Img_moments_hu[EachMoment] = -1* np.copysign(1.0, Img_moments_hu[EachMoment]) * np.log10(abs(Img_moments_hu[EachMoment]))
                
#                print(sum(Img_moments_hu[0:4]))
#                print('Img_moments_hu is {}'.format(Img_moments_hu))
                
                #--------------------------------------------------------------
                # Roundness Threshold
                if Roundness < Roundness_thres:
                    MeanIntensity_FilledArea = np.mean(RawRegionImg[np.where(filled_mask_bef == 1)]) - MeanIntensity_Background # Mean pixel value of filled raw cell area
                                    
                    MeanIntensity_Contour = np.mean(RawRegionImg[np.where(contour_mask_of_cell == 1)]) - MeanIntensity_Background
                    
                    soma_mask_of_cell = filled_mask_convolve2d - contour_mask_of_cell
                    MeanIntensity_Soma = np.mean(RawRegionImg[np.where(soma_mask_of_cell == 1)]) - MeanIntensity_Background#Mean pixel value of soma area                
                    contour_soma_ratio = MeanIntensity_Contour/MeanIntensity_Soma
                    
                    Cell_Area_Img = filled_mask_convolve2d * RawRegionImg
                    # Calculate the entrophy of the image.
    #                entr_img = entropy(Cell_Area_Img/np.amax(Cell_Area_Img), disk(5))
    #                print(np.mean(entr_img))
    
                    #---------------------Calculate dead pixels----------------
                    DeadPixelNum = len(np.where(Cell_Area_Img >= 3.86)[0])
                    filled_mask_convolve2d_area = len(np.where(filled_mask_convolve2d >= 0)[0])
                    DeadPixelPercentage = round(DeadPixelNum / filled_mask_convolve2d_area, 3)
#                    print('Dead Pixel percentage: {}'.format(DeadPixelPercentage)) # b[np.where(aa==16)]=2
                    
                    if str(MeanIntensity_FilledArea) == 'nan':
                        MeanIntensity_FilledArea = 0
                    if str(MeanIntensity_Contour) == 'nan':
                        MeanIntensity_Contour = 0
                    if str(contour_soma_ratio) == 'nan':
                        contour_soma_ratio = 0
                        
                    if DeadPixelPercentage <= DeadPixelPercentageThreshold:
                    
                        dirforcellprp[CellSequenceInRegion] = (boundingbox_info, MeanIntensity_FilledArea, MeanIntensity_Contour, contour_soma_ratio, Roundness)    
                    
    #                    plt.figure()
    #                    plt.imshow(RawRegionImg)
    #                    plt.show()
    #    # #    
    #                    plt.figure()
    #                    plt.imshow(filled_mask_convolve2d)
    #                    plt.show()
                    
                        #--------------------------------------------------Add red boundingbox to axis----------------------------------------------
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                        contour_mean_bef_rounded = str(round(MeanIntensity_Contour, 3))[0:5]
                        
                        if show_img == True:
                            ax_showlabel.add_patch(rect)
                            ax_showlabel.text((maxc + minc)/2, (maxr + minr)/2, 'Cell-{}, {}: {}'.format(CellSequenceInRegion, 'c_m', contour_mean_bef_rounded),
                                              fontsize=8, color='yellow', style='italic')#,bbox={'facecolor':'red', 'alpha':0.3, 'pad':8})
        
                        CellSequenceInRegion += 1
        if show_img == True:
            ax_showlabel.set_axis_off()
            plt.show()
            
        TagFluorescenceLookupBook = np.zeros(CellSequenceInRegion, dtype = dtype)
        for p in range(CellSequenceInRegion):
            TagFluorescenceLookupBook[p] = dirforcellprp[p]
            
        return TagFluorescenceLookupBook
    
    def extract_information_from_bbox(image, bbox_list, DeadPixelPercentageThreshold, contour_thres, contour_dilationparameter, cell_region_opening_factor, cell_region_closing_factor):
        """
        Extract information from bounding box in image.
        
        Parameters
        ----------
        image : 2-D ndarray.
            Input image.
        bbox_list: List of strings.
            List of each cell's bounding box.
        DeadPixelPercentageThreshold:
            Threshold of percentage of saturated pixels in input image.
        contour_thres:
            Threshold used when finding cell contour.
        contour_dilationparameter:
            Degree of morphology dilation operation for generating contour mask from contour line.
        cell_region_opening_factor:
             Degree of morphology opening operation when generating the filled mask for cell.
        cell_region_closing_factor:
             Degree of morphology closing operation when generating the filled mask for cell.
             
        Returns
        -------
        LibFluorescenceLookupBook : Dictionary.
            Dictionary with all cells' information in dtype.
        
        """
        
        dtype = [('BoundingBox', 'U32'), ('Mean intensity', float), ('Mean intensity in contour', float), ('Contour soma ratio', float)]
        CellSequenceInRegion = 0
        dirforcellprp = {}
        
        show_img = False
        if show_img == True:
            plt.figure()
            fig_showlabel, ax_showlabel = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax_showlabel.imshow(image)#Show the first image
            
        for Each_bounding_box in bbox_list:

            # Retrieve boundingbox information
            minr = int(Each_bounding_box[Each_bounding_box.index('minr')+4:Each_bounding_box.index('_minc')])
            maxr = int(Each_bounding_box[Each_bounding_box.index('maxr')+4:Each_bounding_box.index('_maxc')])        
            minc = int(Each_bounding_box[Each_bounding_box.index('minc')+4:Each_bounding_box.index('_maxr')])
            maxc = int(Each_bounding_box[Each_bounding_box.index('maxc')+4:len(Each_bounding_box)])
                
            # Based on the boundingbox for each cell from first image in the stack, raw image of slightly larger region is extracted from each round.
            RawRegionImg = image[max(minr-4,0):min(maxr+4, image[0].shape[0]), max(minc-4,0):min(maxc+4, image[0].shape[0])] # Raw region image 
            
            RawRegionImg_for_contour = RawRegionImg.copy()
            
            #---------Get the cell filled mask-------------
            bbox_area = (maxr-minr)*(maxc-minc)

            filled_mask_bef, MeanIntensity_Background = ProcessImage.get_cell_filled_mask(RawRegionImg = RawRegionImg, region_area = bbox_area*0.2, 
                                                                                                  cell_region_opening_factor = cell_region_opening_factor, 
                                                                                                  cell_region_closing_factor = cell_region_closing_factor)

            filled_mask_convolve2d = ProcessImage.smoothing_filled_mask(RawRegionImg, filled_mask_bef = filled_mask_bef, region_area = bbox_area*0.2, threshold_factor = 1.1)

            # Find contour along filled image
            contour_mask_thin_line = ProcessImage.findContour(filled_mask_convolve2d, RawRegionImg_for_contour.copy(), contour_thres) 

            # after here intensityimage_intensity is changed from contour labeled with number 5 to binary image
            contour_mask_of_cell = ProcessImage.inward_mask_dilation(contour_mask_thin_line.copy() ,filled_mask_convolve2d, contour_dilationparameter)
            
            # Calculate mean values.
            #--------------------------------------------------------------
            MeanIntensity_FilledArea = np.mean(RawRegionImg[np.where(filled_mask_bef == 1)]) - MeanIntensity_Background # Mean pixel value of filled raw cell area
                            
            MeanIntensity_Contour = np.mean(RawRegionImg[np.where(contour_mask_of_cell == 1)]) - MeanIntensity_Background
            
            soma_mask_of_cell = filled_mask_convolve2d - contour_mask_of_cell
            MeanIntensity_Soma = np.mean(RawRegionImg[np.where(soma_mask_of_cell == 1)]) - MeanIntensity_Background#Mean pixel value of soma area                
            contour_soma_ratio = MeanIntensity_Contour/MeanIntensity_Soma

            Cell_Area_Img = filled_mask_convolve2d * RawRegionImg
            
            #---------------------Calculate dead pixels----------------
            DeadPixelNum = len(np.where(Cell_Area_Img >= 3.86)[0])
            filled_mask_convolve2d_area = len(np.where(filled_mask_convolve2d >= 0)[0])
            DeadPixelPercentage = round(DeadPixelNum / filled_mask_convolve2d_area, 3)
            
            if str(MeanIntensity_FilledArea) == 'nan':
                MeanIntensity_FilledArea = 0
            if str(MeanIntensity_Contour) == 'nan':
                MeanIntensity_Contour = 0
            if str(contour_soma_ratio) == 'nan':
                contour_soma_ratio = 0
            
            if DeadPixelPercentage <= DeadPixelPercentageThreshold:
                dirforcellprp[CellSequenceInRegion] = (Each_bounding_box, MeanIntensity_FilledArea, MeanIntensity_Contour, contour_soma_ratio, )
                
                # plt.figure()
                # plt.imshow(RawRegionImg)
                # plt.show()
    
                # plt.figure()
                # plt.imshow(contour_mask_of_cell)
                # plt.show()
            
                #--------------------------------------------------Add red boundingbox to axis----------------------------------------------
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                contour_mean_bef_rounded = str(round(MeanIntensity_Contour, 3))[0:5]
                
                if show_img == True:
                    ax_showlabel.add_patch(rect)
                    ax_showlabel.text((maxc + minc)/2, (maxr + minr)/2, 'Cell-{}, {}: {}'.format(CellSequenceInRegion, 'c_m', contour_mean_bef_rounded),fontsize=8, color='yellow', style='italic')
    
                CellSequenceInRegion += 1
                
        if show_img == True:
            ax_showlabel.set_axis_off()
            plt.show()
            
        LibFluorescenceLookupBook = np.zeros(CellSequenceInRegion, dtype = dtype)
        for p in range(CellSequenceInRegion):
            LibFluorescenceLookupBook[p] = dirforcellprp[p]
            
        return LibFluorescenceLookupBook        
    #%%
    """           
    # ==========================================================================================================================================================
    # ************************************    Contour scanning processing    ************************************    
    # ==========================================================================================================================================================
    """
    def findContour(imagewithouthole, image, threshold):
        """
        Return contour mask by eroding inward from filled cell mask.
        
        Parameters
        ----------
        imagewithouthole : 2-D ndarray.
            Input filled image.
        image : 2-D ndarray.
            Raw image.
        threshold : Float.
            Threshold for finding contour.
            
        Returns
        -------
        binarycontour : 2-D ndarray.
            Binary contour mask.
        """      
        contours = find_contours(imagewithouthole, threshold) # Find iso-valued contours in a 2D array for a given level value.
                
        for n, contour in enumerate(contours):
            #print(contour[1,0])
            col = contour[:, 1]
            row = contour[:, 0]
            col1 = [int(round(i)) for i in col]
            row1 = [int(round(i)) for i in row]
                    
            for m in range(len(col1)):
                image[row1[m], col1[m]] = 5
                #filledimg[contour[:, 0], contour[:, 1]] = 2
            #ax.plot(contour[:, 1]+minc, contour[:, 0]+minr, linewidth=3, color='yellow')
        binarycontour = np.where(image == 5, 1, 0)
        
        return binarycontour
    
    def inward_mask_dilation(binarycontour, imagewithouthole, dilationparameter):
        
        dilationimg = dilation(binarycontour, square(dilationparameter))
        
        contour_mask = dilationimg*imagewithouthole
        
        return contour_mask
    
    def get_cell_filled_mask(RawRegionImg, region_area, cell_region_opening_factor, cell_region_closing_factor):
        '''
        RawRegionImg: Original region image.
        region_area: Area of binary whole cell mask.
        cell_region_opening_factor: Number used for opening.
        cell_region_closing_factor: Number used for closing.
        '''
        #---------------------------------------------------Get binary cell image baseed on expanded current region image-------------------------------------------------
        RawRegionImg = denoise_tv_chambolle(RawRegionImg, weight=0.01)
        binary_adaptive_block_size = region_area*0.3
        if (binary_adaptive_block_size % 2) == 0:
            binary_adaptive_block_size += 1
#        thresh_regionbef = threshold_otsu(RawRegionImg)
        thresh_regionbef = threshold_local(RawRegionImg, binary_adaptive_block_size, offset=0)
        expanded_binary_region_bef = np.where(RawRegionImg >= thresh_regionbef, 1, 0)
        
        binarymask_bef = opening(expanded_binary_region_bef, square(int(cell_region_opening_factor)))
        expanded_binary_region_bef = closing(binarymask_bef, square(int(cell_region_closing_factor)))

        #---------------------------------------------------fill in the holes, prepare for contour recognition-----------------------------------------------------------
        seed_bef = np.copy(expanded_binary_region_bef)
        seed_bef[1:-1, 1:-1] = expanded_binary_region_bef.max()
        mask_bef = expanded_binary_region_bef

        filled_mask_bef = reconstruction(seed_bef, mask_bef, method='erosion')# The binary mask with filling holes
        
        # Calculate the background
        MeanIntensity_Background = np.mean(RawRegionImg[np.where(filled_mask_bef == 0)])
        """ MeanIntensity_Background is not accurate!!!
        """
        MeanIntensity_Background = 0 
        #----------------------------------------------------Clean up parts that don't belong to cell of interest---------------------------------------
        SubCellClearUpSize = int(region_area*0.35) # Assume that trash parts won't take up 35% of the whole cell boundbox area
#        print(region_area)
        IndividualCellCleared = filled_mask_bef.copy()

        clear_border(IndividualCellCleared)
        # label image regions, prepare for regionprops
        IndividualCell_label_image = label(IndividualCellCleared)
        
        for subcellregion in regionprops(IndividualCell_label_image,intensity_image = RawRegionImg.copy()):
            
            if subcellregion.area < SubCellClearUpSize: # Clean parts that are smaller than SubCellClearUpSize, which should result in only one main part left.

                for EachsubcellregionCoords in subcellregion.coords:
#                                print(EachsubcellregionCoords.shape)
                    filled_mask_bef[EachsubcellregionCoords[0], EachsubcellregionCoords[1]] = 0
        #------------------------------------------------------------------------------------------------------------------------------------------------         
     
        return filled_mask_bef, MeanIntensity_Background
    
    
    def smoothing_filled_mask(RawRegionImg, filled_mask_bef, region_area, threshold_factor):
        """
        Given the cell filled mask, smooth the egde by convolution.
        '"""
        # Shrink the image a bit.
#        filled_mask_bef = binary_erosion(filled_mask_bef, square(1))
        # Try to smooth the boundary.
        kernel = np.ones((5,5))
        filled_mask_convolve2d = convolve2d(filled_mask_bef, kernel, mode='same')                
        try:
            filled_mask_convolve2d = np.where(filled_mask_convolve2d >= threshold_otsu(filled_mask_convolve2d)*threshold_factor, 1, 0) # Here higher the threshold a bit to shrink the mask, make sure generated contour doesn't exceed.
        except:
            pass
        # Get rid of little patches.
#                self.filled_mask_convolve2d = opening(self.filled_mask_convolve2d, square(int(1)))
        
        #---------------------------------------------------fill in the holes, prepare for contour recognition-----------------------------------------------------------
        seed_bef = np.copy(filled_mask_convolve2d)
        seed_bef[1:-1, 1:-1] = filled_mask_convolve2d.max()
        mask_bef = filled_mask_convolve2d

        filled_mask_reconstructed = reconstruction(seed_bef, mask_bef, method='erosion')# The binary mask with filling holes        
        #----------------------------------------------------Clean up parts that don't belong to cell of interest---------------------------------------
        SubCellClearUpSize = int(region_area*0.30) # Assume that trash parts won't take up 35% of the whole cell boundbox area
#                    print('minsize: '+str(SubCellClearUpSize))
        IndividualCellCleared = filled_mask_reconstructed.copy()

        clear_border(IndividualCellCleared)
        # label image regions, prepare for regionprops
        IndividualCell_label_image = label(IndividualCellCleared)
        
        for subcellregion_convolve2d in regionprops(IndividualCell_label_image,intensity_image = RawRegionImg.copy()):
            
            if subcellregion_convolve2d.area < SubCellClearUpSize:

                for EachsubcellregionCoords in subcellregion_convolve2d.coords:
#                                print(EachsubcellregionCoords.shape)
                    filled_mask_reconstructed[EachsubcellregionCoords[0], EachsubcellregionCoords[1]] = 0
        #------------------------------------------------------------------------------------------------------------------------------------------------
        return filled_mask_reconstructed

    
    def get_Skeletonized_contour(image, RegionProposalMask, smallest_size, contour_thres, contour_dilationparameter, cell_region_opening_factor, 
                                 cell_region_closing_factor, scanning_voltage, points_per_contour, sampling_rate):
        """
        # =============================================================================
        #         Get the skeletonized contour of the cell for automated contour scanning.
        # -- RegionProposalMask: the binary mask for region iterative analysis.
        # -- smallest_size: cells size below this number are ignored.
        # -- lowest_region_intensity: cells with mean region intensity below this are ignored.
        # -- contour_thres: threshold for contour recognizition.
        # -- contour_dilationparameter: the dilation degree applied when doing inward contour dilation for thicker menbrane area.
        # -- cell_region_opening_factor: degree of opening operation on individual cell mask.
        # -- cell_region_closing_factor: degree of closing operation on individual cell mask.
        # -- scanning_voltage: The scanning voltage of input image.
        # -- points_per_contour: desired number of points in contour routine.
        # -- sampling_rate: sampling rate for contour scanning.
        # =============================================================================
        """
        cleared = RegionProposalMask.copy()
        clear_border(cleared)
        # label image regions, prepare for regionprops
        label_image = label(cleared)
        
        CellSequenceInRegion = 0
        CellSkeletonizedContourDict = {}
#        dtype = [('No.', int), ('Mean intensity', float), ('Mean intensity in contour', float), ('Contour soma ratio', float)]
        
        for region in regionprops(label_image,intensity_image = image): # USE first image in stack before perfusion as template 
            
            # skip small images
            if region.area > smallest_size:
         
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                
                #region_mean_intensity = region.mean_intensity #mean intensity of the region, 0 pixels in label are omitted.
                
                # Based on the boundingbox for each cell from first image in the stack, raw image of slightly larger region is extracted from each round.
                RawRegionImg = image[max(minr-4,0):min(maxr+4, image[0].shape[0]), max(minc-4,0):min(maxc+4, image[0].shape[0])] # Raw region image 
                
                RawRegionImg_for_contour = RawRegionImg.copy()
                
                #---------Get the cell filled mask-------------
                filled_mask_bef, MeanIntensity_Background = ProcessImage.get_cell_filled_mask(RawRegionImg = RawRegionImg, region_area = region.area, 
                                                                            cell_region_opening_factor = cell_region_opening_factor, 
                                                                            cell_region_closing_factor = cell_region_closing_factor)
                
                filled_mask_convolve2d = ProcessImage.smoothing_filled_mask(RawRegionImg, filled_mask_bef = filled_mask_bef, region_area = region.area, threshold_factor = 2)
                
                # Set the edge lines to zero so that we don't have the risk of unclosed contour at the edge of image.
                if minr == 0 or minc == 0:
                    filled_mask_convolve2d[0,:] = False
                    filled_mask_convolve2d[:,0] = False
                if maxr == image[0].shape[0] or maxc == image[0].shape[0]:
                    filled_mask_convolve2d[filled_mask_convolve2d.shape[0]-1, :] = False
                    filled_mask_convolve2d[:, filled_mask_convolve2d.shape[1]-1] = False
                    
                # Find contour along filled image
                contour_mask_thin_line = ProcessImage.findContour(filled_mask_convolve2d, RawRegionImg_for_contour.copy(), contour_thres) 
#                plt.figure()
#                plt.imshow(contour_mask_thin_line)
#                plt.show()
                # after here intensityimage_intensity is changed from contour labeled with number 5 to binary image
#                contour_mask_of_cell = imageanalysistoolbox.inward_mask_dilation(contour_mask_thin_line.copy() ,filled_mask_convolve2d, contour_dilationparameter)
                #--------------------------------------------------------------
#                print(len(np.where(contour_mask_thin_line == 1)[0]))
                if len(np.where(contour_mask_thin_line == 1)[0]) > 0:
                    #-------------------Sorting and filtering----------------------
                    clockwise_sorted_raw_trace = ProcessImage.sort_index_clockwise(contour_mask_thin_line)
                    [X_routine, Y_routine], filtered_cellmap = ProcessImage.tune_contour_routine(contour_mask_thin_line, clockwise_sorted_raw_trace, filtering_kernel = 1.5)
                    #--------------------------------------------------------------
                    
                    #----------Put contour image back to original image.-----------
                    ContourFullFOV = np.zeros((image.shape[0], image.shape[1]))
                    ContourFullFOV[max(minr-4,0):min(maxr+4, image[0].shape[0]), max(minc-4,0):min(maxc+4, image[0].shape[0])] = filtered_cellmap.copy()
    
                    X_routine = X_routine + max(minr-4,0)
                    Y_routine = Y_routine + max(minc-4,0)
                    #--------------------------------------------------------------
                    
                    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
                    ax1.imshow(ContourFullFOV, cmap = plt.cm.gray)
                    ax2.imshow(filtered_cellmap*2+RawRegionImg, cmap = plt.cm.gray)
    #                ax2.imshow(ContourFullFOV*2+image, cmap = plt.cm.gray)
    #                ax2.imshow(filled_mask_convolve2d, cmap = plt.cm.gray)           
    #                figure.tight_layout()
                    plt.show()
                    
                    #------------Organize for Ni-daq execution---------------------
                    voltage_contour_routine_X = (X_routine/ContourFullFOV.shape[0])*scanning_voltage*2-scanning_voltage
                    voltage_contour_routine_Y = (Y_routine/ContourFullFOV.shape[1])*scanning_voltage*2-scanning_voltage
                    
                    #--------------interpolate to get 500 points-------------------
                    x_axis = np.arange(0,len(voltage_contour_routine_X))
                    f_x = interpolate.interp1d(x_axis, voltage_contour_routine_X, kind='cubic')
                    newx = np.linspace(x_axis.min(), x_axis.max(), num=points_per_contour)
                    X_interpolated = f_x(newx)
                    
                    y_axis = np.arange(0,len(voltage_contour_routine_Y))
                    f_y = interpolate.interp1d(y_axis, voltage_contour_routine_Y, kind='cubic')
                    newy = np.linspace(y_axis.min(), y_axis.max(), num=points_per_contour)
                    Y_interpolated = f_y(newy)
                    
                    #-----------speed and accelation check-------------------------
    #                contour_x_speed = np.diff(X_interpolated)/time_gap
    #                contour_y_speed = np.diff(Y_interpolated)/time_gap
                    time_gap = 1/sampling_rate
                    contour_x_acceleration = np.diff(X_interpolated, n=2)/time_gap**2
                    contour_y_acceleration = np.diff(Y_interpolated, n=2)/time_gap**2
                    
                    if AccelerationGalvo < np.amax(abs(contour_x_acceleration)):
                        print(np.amax(abs(contour_x_acceleration)))
                    if AccelerationGalvo < np.amax(abs(contour_y_acceleration)):
                        print(np.amax(abs(contour_y_acceleration)))
                    
                    X_interpolated = np.around(X_interpolated, decimals=3)
                    Y_interpolated = np.around(Y_interpolated, decimals=3)
                    
                    ContourArray_forDaq = np.vstack((X_interpolated,Y_interpolated))
                    
                    CellSkeletonizedContourDict['DaqArray_cell{}'.format(CellSequenceInRegion)] = ContourArray_forDaq
                    CellSkeletonizedContourDict['ContourMap_cell{}'.format(CellSequenceInRegion)] = ContourFullFOV
                    CellSequenceInRegion += 1
                    #--------------------------------------------------------------
                                    
                
        return CellSkeletonizedContourDict
    
    def sort_index_clockwise(cellmap):
        """
        # =============================================================================
        #  Given the binary contour, sort the index so that they are in clockwise sequence for further contour scanning.
        # =============================================================================
        """

        rawindexlist = list(zip(np.where(cellmap == 1)[0], np.where(cellmap == 1)[1]))
        rawindexlist.sort()
        
        
        cclockwiselist = rawindexlist[0:1] # first point in clockwise direction
        clockwiselist = rawindexlist[1:2] # first point in counter clockwise direction
        # reverse the above assignment depending on how first 2 points relate
        if rawindexlist[1][1] > rawindexlist[0][1]: 
            clockwiselist = rawindexlist[1:2]
            cclockwiselist = rawindexlist[0:1]
        
        coordstorage = rawindexlist[2:]
#        print(len(rawindexlist))
        timeout = time.time()
        while len(clockwiselist+cclockwiselist) != len(rawindexlist):
            for p in coordstorage:#Try one by one from coords dump until find one that is right next to existing clockwise or counter clockwise liste.
                # append to the list to which the next point is closest
                x_last_clockwise = clockwiselist[-1][0]
                y_last_clockwise = clockwiselist[-1][1]
                x_last_cclockwise = cclockwiselist[-1][0]
                y_last_cclockwise = cclockwiselist[-1][1]
#                if (x_last_clockwise-p[0])**2+(y_last_clockwise-p[1])**2 == 1 and \
#                ((x_last_clockwise-p[0])**2+(y_last_clockwise-p[1])**2) < ((x_last_cclockwise-p[0])**2+(y_last_cclockwise-p[1])**2):
#                    clockwiselist.append(p)
#                    coordstorage.remove(p)                    
                if (x_last_clockwise-p[0])**2+(y_last_clockwise-p[1])**2 <= 2 and \
                ((x_last_clockwise-p[0])**2+(y_last_clockwise-p[1])**2) <= ((x_last_cclockwise-p[0])**2+(y_last_cclockwise-p[1])**2):
                    clockwiselist.append(p)
                    coordstorage.remove(p)
                    break
                elif (x_last_cclockwise-p[0])**2+(y_last_cclockwise-p[1])**2 <= 2 and \
                ((x_last_clockwise-p[0])**2+(y_last_clockwise-p[1])**2) > ((x_last_cclockwise-p[0])**2+(y_last_cclockwise-p[1])**2):
#                    print((cclockwiselist[-1][0]-p[0])**2+(cclockwiselist[-1][1]-p[1])**2)
#                    print('cc')
                    cclockwiselist.append(p)
                    coordstorage.remove(p)
                    break
            # If clockwise and counter clockwise meet each other
            if len(clockwiselist+cclockwiselist) > 10 and (x_last_clockwise-x_last_cclockwise)**2+(y_last_clockwise-y_last_cclockwise)**2 <= 2:
                break
            # If we have a situation like this at the end of enclosure:
            #  0  0  1
            #  0  1  1
            #  1  0  0
            if len(clockwiselist+cclockwiselist) > 10 and (x_last_clockwise-x_last_cclockwise)**2+(y_last_clockwise-y_last_cclockwise)**2 == 5:
                if (cclockwiselist[-2][0]-clockwiselist[-1][0])**2 + (cclockwiselist[-2][1]-clockwiselist[-1][1])**2 == 2:
                    cclockwiselist.remove(cclockwiselist[-1])
                    break
                
            if time.time() > timeout+2:
                print('timeout')
                break
#        print(clockwiselist[-1])
#        print(cclockwiselist[-1])
#        print(p)
        print(coordstorage)
        cclockwiselist.reverse()
        result = clockwiselist + cclockwiselist
        
        return result
    
    def tune_contour_routine(cellmap, clockwise_sorted_raw_trace, filtering_kernel):
        """
        # =============================================================================
        #  Given the clockwise sorted binary contour, interploate and filter for further contour scanning.
        # =============================================================================
        """
        Unfiltered_contour_routine_X = np.array([])
        Unfiltered_contour_routine_Y = np.array([])
        for rawcoord in clockwise_sorted_raw_trace:
            Unfiltered_contour_routine_X = np.append(Unfiltered_contour_routine_X, rawcoord[0])
            Unfiltered_contour_routine_Y = np.append(Unfiltered_contour_routine_Y, rawcoord[1])
        
        # filtering and show filtered contour
#        X_routine = medfilt(Unfiltered_contour_routine_X, kernel_size=filtering_kernel)
#        Y_routine = medfilt(Unfiltered_contour_routine_Y, kernel_size=filtering_kernel)
        X_routine = gaussian_filter1d(Unfiltered_contour_routine_X, sigma = filtering_kernel)
        Y_routine = gaussian_filter1d(Unfiltered_contour_routine_Y, sigma = filtering_kernel)
        
        filtered_cellmap = np.zeros((cellmap.shape[0], cellmap.shape[1]))
        for i in range(len(X_routine)):
            filtered_cellmap[int(X_routine[i]), int(Y_routine[i])] = 1

        
        return [X_routine, Y_routine], filtered_cellmap
    
    def mask_to_contourScanning_DAQsignals(filled_mask, OriginalImage, scanning_voltage, points_per_contour, sampling_rate, repeats = 1):
        """
        # =============================================================================
        #  Given the binary mask which ONLY covers cell of interest and original image,
        #  generate the voltage signals to NI-DAQ for one contour scanning.
        #
        #  -- filled_mask: The filled binary mask which ONLY covers cell of interest.
        #     Refer to outputs from func:get_cell_filled_mask and func:smoothing_filled_mask.
        #  -- OriginalImage: Raw image.
        #  -- scanning_voltage: The scanning voltage of input image.
        #  -- points_per_contour: desired number of points in contour routine.
        #  -- sampling_rate: sampling rate for contour scanning.
        # =============================================================================        
        """
        AccelerationGalvo = 1.54*10**8 # Maximum acceleration of galvo mirror in volt/s^2
        
        # Find contour along filled image
        contour_mask_thin_line = ProcessImage.findContour(filled_mask, OriginalImage.copy(), threshold=0.001) 
        #--------------------------------------------------------------
        if len(np.where(contour_mask_thin_line == 1)[0]) > 0:
            #-------------------Sorting and filtering----------------------
            clockwise_sorted_raw_trace = ProcessImage.sort_index_clockwise(contour_mask_thin_line)
            [X_routine, Y_routine], filtered_cellmap = ProcessImage.tune_contour_routine(contour_mask_thin_line, clockwise_sorted_raw_trace, filtering_kernel = 1.5)
            #--------------------------------------------------------------
            
            #------------Organize for Ni-daq execution---------------------
            voltage_contour_routine_X = (X_routine/OriginalImage.shape[0])*scanning_voltage*2-scanning_voltage
            voltage_contour_routine_Y = (Y_routine/OriginalImage.shape[1])*scanning_voltage*2-scanning_voltage
            
            #-----interpolate to get desired number of points in one contour---
            x_axis = np.arange(0,len(voltage_contour_routine_X))
            f_x = interpolate.interp1d(x_axis, voltage_contour_routine_X, kind='cubic')
            newx = np.linspace(x_axis.min(), x_axis.max(), num=points_per_contour)
            X_interpolated = f_x(newx)
            
            y_axis = np.arange(0,len(voltage_contour_routine_Y))
            f_y = interpolate.interp1d(y_axis, voltage_contour_routine_Y, kind='cubic')
            newy = np.linspace(y_axis.min(), y_axis.max(), num=points_per_contour)
            Y_interpolated = f_y(newy)
            
            #---------------speed and accelation check-------------------------
            time_gap = 1/sampling_rate
            contour_x_acceleration = np.diff(X_interpolated, n=2)/time_gap**2
            contour_y_acceleration = np.diff(Y_interpolated, n=2)/time_gap**2
            
            if AccelerationGalvo < np.amax(abs(contour_x_acceleration)):
                print('Danger! Xmax: {}'.format(np.amax(abs(contour_x_acceleration))))
            if AccelerationGalvo < np.amax(abs(contour_y_acceleration)):
                print('Danger! Ymax: {}'.format(np.amax(abs(contour_y_acceleration))))
            
            X_interpolated = np.tile(np.around(X_interpolated, decimals=3), repeats)
            Y_interpolated = np.tile(np.around(Y_interpolated, decimals=3), repeats)
            
            # Pure numerical np arrays need to be converted to structured array, with 'Sepcification' field being the channel name.
            tp_analog = np.dtype([('Waveform', float, (len(X_interpolated),)), ('Sepcification', 'U20')])
            ContourArray_forDaq = np.zeros(2, dtype =tp_analog)
            ContourArray_forDaq[0] = np.array([(X_interpolated, 'galvos_X_contour')], dtype =tp_analog)
            ContourArray_forDaq[1] = np.array([(Y_interpolated, 'galvos_Y_contour')], dtype =tp_analog)
            
            # ContourArray_forDaq = np.vstack((X_interpolated,Y_interpolated))
        else:
            print('Error: no contour found')
            return
            
        return ContourArray_forDaq
    
    #--------------------------------------------------------------------------
    
    def get_cell_properties_Roundstack(imagestack, RegionProposalMask, smallest_size, contour_thres, contour_dilationparameter, cell_region_opening_factor, cell_region_closing_factor):
        
        cleared = RegionProposalMask.copy()
        clear_border(cleared)
        # label image regions, prepare for regionprops
        label_image = label(cleared)
        
        dtype = [('Mean intensity', float), ('Mean intensity in contour', float), ('Circularity', float), ('Contour soma ratio', float)]
        
        CellPropDictEachRound = {}
        
        for EachRound in range(len(imagestack)):
            
            region_mean_intensity_list = []
            region_circularit_list = []
            region_meanintensity_contour_list = []        
            RegionLoopNumber = 0
            dirforcellprp={}
            
            if EachRound == 0: # Plot the label figure in the first round. 

                plt.figure()
                fig_showlabel, ax_showlabel = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                ax_showlabel.imshow(imagestack[0])#Show the first image
            
            for region in regionprops(label_image,intensity_image = imagestack[0]): # USE first image in stack before perfusion as template 
                
                # skip small images
                if region.area > smallest_size:
             
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    
                    #region_mean_intensity = region.mean_intensity #mean intensity of the region, 0 pixels in label are omitted.
                    #allpixelnum = region.bbox_area
                    #labeledpixelnum = region.area #number of pixels in region omitting 0.
                    filledimg = region.filled_image 
                    filledperimeter = perimeter(filledimg)
                    regioncircularity = (4 * math.pi * region.filled_area) / (filledperimeter * filledperimeter) # region.perimeter will count in perimeters from the holes inside
                    #Sliced_binary_region_image = region.image
#                    intensityimage_intensity = region.intensity_image # need a copy of this cause region will be altered by s.contour
                    
                    # Based on the boundingbox for each cell from first image in the stack, raw image of slightly larger region is extracted from each round.
                    RawRegionImg = imagestack[EachRound][max(minr-4,0):min(maxr+4, imagestack[0].shape[0]), max(minc-4,0):min(maxc+4, imagestack[0].shape[0])] # Raw region image 
                    
                    RawRegionImg_for_contour = RawRegionImg.copy()

                    #---------Get the cell filled mask-------------
                    filled_mask_bef = ProcessImage.get_cell_filled_mask(RawRegionImg = RawRegionImg, region_area = region.area, 
                                                                                cell_region_opening_factor = cell_region_opening_factor, 
                                                                                cell_region_closing_factor = cell_region_closing_factor)

                    filled_origin_image_intensity = RawRegionImg*filled_mask_bef # Intensity image of cell with hole filled
                    filled_mean_bef = np.mean(RawRegionImg[np.where(filled_mask_bef == 1)]) # Mean pixel value of filled raw cell area
                    
                    filled_mask_convolve2d = ProcessImage.smoothing_filled_mask(RawRegionImg, filled_mask_bef = filled_mask_bef, region_area = region.area, threshold_factor = 1.1)
                    
                    # Find contour along filled image
#                    imageanalysistoolbox_instacne=imageanalysistoolbox()
                    contour_mask_bef = ProcessImage.findContour(filled_mask_convolve2d, RawRegionImg_for_contour.copy(), contour_thres)
                    # after here intensityimage_intensity is changed from contour labeled with number 5 to binary image
                    contour_mask_of_cell = ProcessImage.inward_mask_dilation(contour_mask_bef ,filled_mask_bef, contour_dilationparameter)   
                
#                    contourimage_intensity_aft = s.contour(filled_mask_aft, regionimage_after_for_contour.copy(), contour_thres) 
#                    after here intensityimage_intensity is changed with contour labeled with number 5
#                    contour_mask_of_intensity_aft = s.inward_mask_dilation(contourimage_intensity_aft ,filled_mask_aft, contour_dilationparameter)
    
                    contour_mean_bef = np.mean(RawRegionImg[np.where(contour_mask_of_cell == 1)])
#                    contour_mean_aft = np.mean(regionimage_after[np.where(contour_mask_of_intensity_aft == 1)])  
                    
                    cell_soma_mask_bef = filled_mask_bef - contour_mask_of_cell
                    
                    contour_origin_image_intensity = RawRegionImg*contour_mask_of_cell # Intensity image of cell contour
                    soma_origin_image_intensity = RawRegionImg*cell_soma_mask_bef # Intensity image of cell soma part
    
                    soma_mean_bef = np.mean(RawRegionImg[np.where(cell_soma_mask_bef == 1)])#Mean pixel value of soma area
                    
                    contour_soma_ratio = contour_mean_bef/soma_mean_bef

                    region_mean_intensity_list.append(filled_mean_bef)# Mean intensity of filled image
                    
                    region_circularit_list.append(regioncircularity)
                    region_meanintensity_contour_list.append(contour_mean_bef)                
                    dirforcellprp[RegionLoopNumber] = (filled_mean_bef, contour_mean_bef, regioncircularity, contour_soma_ratio)
                    
                    RegionLoopNumber = RegionLoopNumber+1
                    
                    #--------------------------------------------------Add red boundingbox to axis----------------------------------------------
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)

                    contour_mean_bef_rounded = str(round(contour_mean_bef, 3))[0:5]
                    if EachRound == 0:
                        ax_showlabel.add_patch(rect)
                        ax_showlabel.text((maxc + minc)/2, (maxr + minr)/2, 'Cell-{}, {}: {}'.format \
                                          (RegionLoopNumber, 'c_m', contour_mean_bef_rounded),fontsize=8, color='yellow', style='italic')
                                       
#            print('Total region number: {}'.format(RegionLoopNumber))
            cell_properties = np.zeros(len(region_mean_intensity_list), dtype = dtype)
            for p in range(RegionLoopNumber):
                cell_properties[p] = dirforcellprp[p]
            
            CellPropDictEachRound['RoundSequence{}'.format(EachRound+1)] = cell_properties
            
            if EachRound == 0:
                ax_showlabel.set_axis_off()
    #            if EachRound == 0:
                plt.show()
            
        return CellPropDictEachRound
    
    """
    # ==========================================================================================================================================================
    # ************************************   Organize cell properties dictionary  ************************************ 
    # ==========================================================================================================================================================
    """

    def TagFluorescenceAnalysis(tag_folder, tag_round, Roundness_threshold):
        """
        # =============================================================================
        # Given the folder and round number of tag fluorescence, return a dictionary 
        # that contains each scanning position as key and structured array of detailed 
        # information about each identified cell as content.
        # =============================================================================
        """
        tagprotein_cell_properties_dict = {}
        RoundNumberList, CoordinatesList, fileNameList = ProcessImage.retrive_scanning_scheme(tag_folder)

        for EachRound in RoundNumberList:
            
            if EachRound == tag_round:
                
                for EachCoord in CoordinatesList:
                    
                # =============================================================================
                #             For tag fluorescence:
                # =============================================================================    
                    print(EachCoord)
                    #-------------- readin image---------------
                    for Eachfilename in enumerate(fileNameList):
                        if EachCoord in Eachfilename[1] and EachRound in Eachfilename[1]:                        
                            tag_imagefilename = os.path.join(tag_folder, Eachfilename[1])
#                    print(tag_imagefilename)
                    loaded_tag_image = imread(tag_imagefilename, as_gray=True)
                    #------------------------------------------
    
                    RegionProposalMask, RegionProposalOriginalImage = ProcessImage.generate_mask(loaded_tag_image, openingfactor=2, 
                                                                                                                closingfactor=3, binary_adaptive_block_size=335)#256(151) 500(335)
                    
                    TagFluorescenceLookupBook = ProcessImage.Region_Proposal(loaded_tag_image, RegionProposalMask, smallest_size=800, biggest_size=3500, Roundness_thres = Roundness_threshold,
                                                                             DeadPixelPercentageThreshold = 0.14, lowest_region_intensity=0.16, contour_thres=0.001, contour_dilationparameter=11,
                                                                             cell_region_opening_factor=1, cell_region_closing_factor=2)
                    
                    tagprotein_cell_properties_dict['{}_{}'.format(EachRound, EachCoord)] = TagFluorescenceLookupBook
                    
    #            for Eachpos in tagprotein_cell_properties_dict:
    #                cellnum = len(tagprotein_cell_properties_dict[Eachpos])
    #                for eachcell in range(cellnum):
    #                    if str(tagprotein_cell_properties_dict[Eachpos][eachcell]['Mean intensity']) != 'nan':
    #                        tag_cell_mean_intensity.append(tagprotein_cell_properties_dict[Eachpos][eachcell]['Mean intensity'])
    #                        trace_back.append('{}_{}'.format(Eachpos, eachcell))
    
        return tagprotein_cell_properties_dict
    
    def LibFluorescenceAnalysis(lib_folder, tag_round, lib_round, tagprotein_cell_properties_dict):
        """
        # =============================================================================
        # Delivering cell properties from tag screening round, return a dictionary 
        # that contains each scanning position as key and structured array of detailed 
        # information about each identified cell as content.
        # =============================================================================
        """
        lib_cell_properties_dict = {}
        RoundNumberList, CoordinatesList, fileNameList = ProcessImage.retrive_scanning_scheme(lib_folder)

        for EachRound in RoundNumberList:
            
            if EachRound == lib_round:
                
                for EachCoord in CoordinatesList:
                # =============================================================================
                #             For library fluorescence:
                # =============================================================================
                    print(EachCoord)
                    bbox_list = []
                    #-------------- readin image---------------
                    for Eachfilename in enumerate(fileNameList):
                        if EachCoord in Eachfilename[1] and EachRound in Eachfilename[1]:            
                            lib_imagefilename = os.path.join(lib_folder, Eachfilename[1])
#                    print(lib_imagefilename)
                    loaded_lib_image = imread(lib_imagefilename, as_gray=True)
                    #------------------------------------------                    
                    
                    #------------unpack bbox infor----------------------
                    for eachcellregion in range(len(tagprotein_cell_properties_dict['{}_{}'.format(tag_round, EachCoord)])):
                        bbox_list.append(tagprotein_cell_properties_dict['{}_{}'.format(tag_round, EachCoord)][eachcellregion]['BoundingBox'])                    

                    LibFluorescenceLookupBook = ProcessImage.extract_information_from_bbox(image = loaded_lib_image, bbox_list = bbox_list, DeadPixelPercentageThreshold = 0.54, 
                                                                                           contour_thres=0.001, contour_dilationparameter=11, cell_region_opening_factor=1, cell_region_closing_factor=2)
                    
                    lib_cell_properties_dict['{}_{}'.format(EachRound, EachCoord)] = LibFluorescenceLookupBook
                    
        
        return lib_cell_properties_dict
    
    def CorrectForFusionProtein(tagprotein_cell_properties_dict, lib_cell_properties_dict, tagprotein_laserpower, lib_laserpower):
        """
        # =============================================================================
        # Assume that there's one tag protein scanning round, add 'Mean intensity devided by tag' field to the structured array in the dictionary.
        # =============================================================================
        """
        for coordinate_tag in tagprotein_cell_properties_dict:
            
            current_coord_string = coordinate_tag[coordinate_tag.index('Coord'):len(coordinate_tag)]
            cell_number = len(tagprotein_cell_properties_dict[coordinate_tag])
            
            for coordinate_lib in lib_cell_properties_dict:
                
                if current_coord_string in coordinate_lib: # No matter what the round number is, look for the same coorinate.
                    
                    divisionlist = []
                    for eachcell in range(cell_number):
                        tag_fluo = tagprotein_cell_properties_dict[coordinate_tag][eachcell]['Mean intensity']
                        lib_fluo = lib_cell_properties_dict[coordinate_lib][eachcell]['Mean intensity']
                        if tag_fluo == 0:
                            division =0
                        else:
                            division = (lib_fluo/lib_laserpower)/(tag_fluo/tagprotein_laserpower)

#                        print(coordinate_lib+', cell'+str(eachcell)+','+str(lib_fluo))
#                        print(division)
                        divisionlist.append(division)
                    
                    lib_cell_properties_dict[coordinate_lib] = rfn.append_fields(lib_cell_properties_dict[coordinate_lib], 'Mean intensity divided by tag', divisionlist, usemask=False)
                        
        return lib_cell_properties_dict
    
    def OrganizeOverview(lib_cell_properties_dict, CutOffThresList, EvaluatingPara_1, WeightPara_1, EvaluatingPara_2, WeightPara_2):
        """
        # =============================================================================
        # Add 'ID' field which indicates position, round information to the structured array in the dictionary.
        # Delete cells that are dimmer than threshold.
        # Add 'Normalized distance' field which indicates the distance to the scatter plots origin in EvaluatingPara_1 and EvaluatingPara_2 dimensions.
        # ** EvaluatingPara: Indicates the field along which calculation of distance takes place.
        # =============================================================================
        """        
        Overview_dtype = [('BoundingBox', 'U32'), ('Mean intensity', float), ('Mean intensity in contour', float), 
                          ('Contour soma ratio', float), ('Mean intensity divided by tag', float)]
        counting = 0
        
        totalcellnum = 0
        for EachCoord in lib_cell_properties_dict:
            totalcellnum += len(lib_cell_properties_dict[EachCoord])
            
        Overview_LookupBook = np.zeros(totalcellnum, dtype = Overview_dtype)
        IDlist = []
        for EachCoord in lib_cell_properties_dict:
            for EachCellArray in lib_cell_properties_dict[EachCoord]:
                Overview_LookupBook[counting] = EachCellArray
#                ID_infor = EachCoord + "_No" + str(counting)
                IDlist.append(EachCoord)
                counting += 1
        # Append the id field
        Overview_LookupBook = rfn.append_fields(Overview_LookupBook, 'ID', IDlist, usemask=False)  
        
        # Trimming according to thresholds
        DeleteIndexList = []
        # Delete cells that are dimmer than threshold.
        for EachCell in range(len(Overview_LookupBook)):
            if 'Mean intensity in contour' in CutOffThresList and 'Contour soma ratio' in CutOffThresList:
                if Overview_LookupBook[EachCell]['Mean intensity in contour'] < CutOffThresList[CutOffThresList.index('Mean intensity in contour') + 1]:
                    DeleteIndexList.append(EachCell)

                elif Overview_LookupBook[EachCell]['Contour soma ratio'] < CutOffThresList[CutOffThresList.index('Contour soma ratio') + 1]:
                    DeleteIndexList.append(EachCell)
        
        Overview_LookupBook = np.delete(Overview_LookupBook, DeleteIndexList, 0)
        
        # Append the number field.
        NoList = []
        for EachCell in range(len(Overview_LookupBook)):
            NoList.append(EachCell)
        Overview_LookupBook = rfn.append_fields(Overview_LookupBook, 'IDNumber', NoList, usemask=False)
            
        # Add 'Normalized distance' field
        NormalizedDistanceArray = np.array([])
        for EachCellIndex in range(len(Overview_LookupBook)):
            # Get the original values of two axes.
            value_Para_1 = Overview_LookupBook[EachCellIndex][EvaluatingPara_1]/np.amax(Overview_LookupBook[EvaluatingPara_1])
            value_Para_2 = Overview_LookupBook[EachCellIndex][EvaluatingPara_2]/np.amax(Overview_LookupBook[EvaluatingPara_2])
            
            Distance = ((value_Para_1 * WeightPara_1)**2 \
                        + (value_Para_2 * WeightPara_2)**2)**0.5

            NormalizedDistanceArray = np.append(NormalizedDistanceArray, Distance)
                
        Overview_LookupBook = rfn.append_fields(Overview_LookupBook, 'Normalized distance', NormalizedDistanceArray, usemask=False)
        
        return Overview_LookupBook
    
    def WeightedSorting(cell_properties, property_1, property_2, property_3, weight_1, weight_2, weight_3):
        """
        # Sorting using linear combination of weights.
        """
        max_p1 = np.amax(cell_properties[property_1])
        max_p2 = np.amax(cell_properties[property_2])
        max_p3 = np.amax(cell_properties[property_3])
        
        weights = cell_properties[property_1]/max_p1*weight_1 + \
                  cell_properties[property_2]/max_p2*weight_2 + \
                  cell_properties[property_3]/max_p3*weight_3
        
        if str(weights) == 'nan':
            weights = 0
            
        cell_properties = rfn.append_fields(cell_properties, 'Weighted ranking', weights, usemask=False)
        
        cell_properties = np.flip(np.sort(cell_properties, order='Weighted ranking'), 0)

        return cell_properties
    
    
    def DistanceSelecting(cell_properties, selectionRadiusPercent):
        """
        # Sorting using field 'Normalized distance'.
        # ** selectionRadiusPercent: Threshold for distance to be selected.
        """
        max_Normalized_distance = np.amax(cell_properties['Normalized distance'])
        min_Normalized_distance = np.amin(cell_properties['Normalized distance'])
        selectionRadiusThres = ((max_Normalized_distance - min_Normalized_distance) * selectionRadiusPercent/100) + min_Normalized_distance
        
        Selected_cell_number = 0
        
        if selectionRadiusPercent == 100:
            Selected_LookupBook = cell_properties
            Selected_cell_number = len(Selected_LookupBook)
        else:
            Selected_LookupBook = np.array([])
            for EachCellIndex in range(len(cell_properties)):
                if cell_properties[EachCellIndex]['Normalized distance'] > selectionRadiusThres:
                    if Selected_cell_number == 0:
                        Selected_LookupBook = np.array([cell_properties[EachCellIndex]])
                    else:
                        Selected_LookupBook = np.append(Selected_LookupBook, cell_properties[EachCellIndex])
                    Selected_cell_number += 1
        
        if Selected_cell_number == 1:
            Selected_LookupBook = Selected_LookupBook
        else:
            Selected_LookupBook = np.flip(np.sort(Selected_LookupBook, order='Normalized distance'), 0)

        return Selected_LookupBook        
    #%%
    # =============================================================================
    #     ROI and mask generation
    # =============================================================================
    def CreateBinaryMaskFromRoiCoordinates(list_of_rois, fill_contour = False, contour_thickness = 1, mask_resolution = (2048, 2048), invert_mask = False):
        """
        Creating a binary mask using a set of vertices defining roi's. This function 
        is written for the purpose of creating DMD masks. 
        
        param list_of_rois: list of numpy arrays, that contain vertices coordinates
                            defining a roi. Example with 2 square roi's:
                                list_of_rois = [np.array((0,0), (1,0), (0,1), (1,1)), np.array((2,2), (2,3), (3,2), (3,3))]
        param fill_contour: if False create contour mask, if True create filled shape mask
        param contour_thickness: only applies when fill_contour is False
        param mask_resolution: size of the mask, default is camera output resolution, for DMD use (768,1024)
        param invert_mask: invert binary mask, meaning 1 and 0 interchange
        """
        
        mask = np.zeros(mask_resolution)

        for roi in list_of_rois:
            if fill_contour:
                mask += polygon2mask(mask_resolution, roi)
            else:
                mask[polygon_perimeter(roi[:,0], roi[:,1], mask_resolution)] = 1
                
                for _ in range(contour_thickness):
                    mask += binary_dilation(binary_dilation(mask))
        
        # Make sure the mask is binary
        mask = (mask > 0).astype(int)
        
        if invert_mask:
            mask = 1 - mask
        
        return mask
    
    def ROIitem2Mask(roi_list, fill_contour = True, contour_thickness = 1, mask_resolution = (1024, 768), invert_mask = False):
        """
        Create binary mask from roi items from pyqtgraph
        
        Parameters
        ----------
        image_shape : tuple of size 2.
            The shape of the mask.
        polygon : array_like.
            The polygon coordinates of shape (N, 2) where N is
            the number of points.
    
        Returns
        -------
        mask : 2-D ndarray of type 'bool'.
            The mask that corresponds to the input polygon.

        """
        list_of_rois = []
        Width = int(mask_resolution[0])
        Height = int(mask_resolution[1])
        
        mask = np.zeros((Width, Height))
                
        if type(roi_list) is list:
            for roi in roi_list:
                roi_handle_positions = roi.getLocalHandlePositions()
#                print(roi.getLocalHandlePositions())
                num_vertices = len(roi_handle_positions)
                vertices = np.zeros([num_vertices,2])
    
                for idx, vertex in enumerate(roi_handle_positions):
                    vertices[idx,:] = np.array([vertex[1].y(), vertex[1].x()])
    #            
                list_of_rois.append(vertices)     
                
                for roi in list_of_rois:
                    if fill_contour:
                        mask += polygon2mask((Width, Height), roi)
                    else:
                        mask[polygon_perimeter(roi[:,0], roi[:,1], mask_resolution)] = 1
                        
                        for _ in range(contour_thickness):
                            mask += binary_dilation(binary_dilation(mask))
                            
        elif type(roi_list) is dict:
            for roikey in roi_list:
                roi = roi_list[roikey]
                roi_handle_positions = roi.getLocalHandlePositions()
#                print(roi.getLocalHandlePositions())
                num_vertices = len(roi_handle_positions)
                vertices = np.zeros([num_vertices,2])
    
                for idx, vertex in enumerate(roi_handle_positions):
                    vertices[idx,:] = np.array([vertex[1].y(), vertex[1].x()])
    #            
                list_of_rois.append(vertices)     
                
                for roi in list_of_rois:
                    if fill_contour:
                        mask += polygon2mask((Width, Height), roi)
                    else:
                        mask[polygon_perimeter(roi[:,0], roi[:,1], mask_resolution)] = 1
                        
                        for _ in range(contour_thickness):
                            mask += binary_dilation(binary_dilation(mask))           
                        
        return mask
    
    def ROIitem2Vertices(roi_items_list):
        """
        Return vertices from input pyqtgraph roi items
        
        Parameters
        ----------
        roi_items_list: list of roi items from pyqtgraph.
    
        Returns
        -------
        list_of_rois: List of vertices np array
            e.g. [np.array([[1,1],[1,2],[2,1]]), #from first roi item
                  np.array([[1,1],[1,2],[2,1]])  #from second roi item
                    ]
        """
        list_of_rois = []

        for roi in roi_items_list:
            roi_handle_positions = roi.getLocalHandlePositions()
            
            num_vertices = len(roi_handle_positions)
            vertices = np.zeros([num_vertices,2])

            for idx, vertex in enumerate(roi_handle_positions):
                vertices[idx,:] = np.array([vertex[1].y(), vertex[1].x()])
           
            list_of_rois.append(vertices)
            
        return list_of_rois
    
    def vertices_to_DMD_mask(vertices_assemble, laser, dict_transformations, \
                             flag_fill_contour = True, contour_thickness = 1, flag_invert_mode = False, mask_resolution = (1024, 768)):
        """
        Create binary DMD transformed mask from input vertices
        
        Parameters
        ----------
        vertices_assemble : np.array of size (n, 2), e.g. np.array([[1,1], [1, 2], [1,3]]),
                            or list of np.array of size 2, e.g. [np.array([1,1]), np.array([2,1])]
            The vertices of the input mask contour.
        laser :
            To which laser the returned DMD mask belongs.
    
        Returns
        -------
        mask_transformed : 2-D ndarray.
            The transformed mask that corresponds to the input vertices group.
        """
        mask_transformed = {}
        list_of_rois_transformed = [] # list of np.array
        
        if type(vertices_assemble) is list or type(vertices_assemble) is np.ndarray:
            
            vertices_assemble = np.asarray(vertices_assemble)

            if 'camera-dmd-'+laser in dict_transformations.keys():

                vertices_transformed = ProcessImage.transform(vertices_assemble, dict_transformations['camera-dmd-'+laser])
                list_of_rois_transformed.append(vertices_transformed)
            else:
                list_of_rois_transformed.append(vertices_assemble)
                print('Warning: not registered')

            mask_transformed[laser] = ProcessImage.CreateBinaryMaskFromRoiCoordinates(list_of_rois_transformed, \
                                             fill_contour = flag_fill_contour, contour_thickness = contour_thickness, invert_mask = flag_invert_mode)
            print(mask_transformed[laser].shape)

        return mask_transformed       
    
    def binarymask_to_DMD_mask(binary_mask, laser, dict_transformations, flag_fill_contour = True, contour_thickness = 1, flag_invert_mode = False, mask_resolution = (1024, 768)):
        """
        First binart mask to contour vertices, then from vertices to transformed vertices then to DMD mask.
        """
        mask_transformed_final = np.zeros((mask_resolution[1], mask_resolution[0]))
        
        contours = find_contours(binary_mask, 0.5) # Find iso-valued contours in a 2D array for a given level value.
                
        for n, contour in enumerate(contours):

            mask_transformed = ProcessImage.vertices_to_DMD_mask(contour, laser, dict_transformations, flag_fill_contour = True, contour_thickness = 1, \
                                              flag_invert_mode = False, mask_resolution = (1024, 768))
            print(mask_transformed_final.shape)
            print(mask_transformed[laser].shape)
            mask_transformed_final += mask_transformed[laser]
            
        return mask_transformed_final
        
    
    def transform(r, A): 
        """
        This function takes points as input and returns the 
        transformed points. 
            
        r = np.array([[1,1], [1, 2], [1,3]])
        
        """
        
        if r.ndim == 1:
            Q = ProcessImage.createTransformationMatrix(r)
            
            if Q is None:
                return
            
            return np.squeeze(np.reshape(np.dot(Q, A), (-1,2), order='F'))
            
        else:
            num_points = r.shape[0]
        
        transformed_points = np.zeros([num_points, 2])
        
        for i in range(num_points):
            
            Q = ProcessImage.createTransformationMatrix(r[i,:])
            
            if Q is None:
                return
            
            transformed_points[i,:] = np.squeeze(np.dot(Q, A))
        return np.reshape(transformed_points, (-1,2), order='F')
        
    def createTransformationMatrix(q, order = 1):
        if len(q.shape) == 1:
            Qx = np.array([1, 0, q[0], q[1]])
            Qy = np.hstack((0, 1, np.zeros(2*order), q[0], q[1]))
    
            for i in range(2,order+1):
                Qx = np.hstack((Qx, q[0]**i, q[1]**i))
                Qy = np.hstack((Qy, q[0]**i, q[1]**i))
            
            Qx = np.hstack((Qx, np.zeros(2*order)))
        else:
            print("Function takes only one point at a time")
            return 
            
        return np.vstack((Qx, Qy))
    
    #%%
    # =============================================================================
    #     MaskRCNN related
    # =============================================================================
    def convert_for_MaskRCNN(input_img):
        """Convert the image size and bit-depth to make it suitable for MaskRCNN detection.
        
        Parameters
        ----------
        input_img : 2-D ndarray.
            Input image.        
            
        Returns
        -------
        output_image : 2-D ndarray.
            Converted image after resizing and bit-size adjustment.
        """
        if input_img.shape[0] > 1024 or input_img.shape[1] > 1024:
            resized_img = resize(input_img,[1024,1024],preserve_range=True).astype(input_img.dtype)
        else:
            resized_img = input_img
            
        minval = np.min(resized_img)
        maxval = np.max(resized_img)
        
        output_image = ((resized_img-minval)/(maxval-minval)*255).astype(np.uint8)+1
        
        if len(np.shape(output_image)) == 2:
            output_image = gray2rgb(output_image)

        return output_image    
    
    def retrieveDataFromML(image, MLresults, ImgNameInfor, cell_counted_number):
        """ Given the raw image and ML returned result dictionary, calculate interested parameters from it.
        
        class_ids = 3: Flat cell
        class_ids = 2: Round cell
        class_ids = 1: Dead cell
        
        #   Returned structured array fields:
        #   - BoundingBox of cell ROI
        #   - Mean intensity of whole cell area
        #   - Mean intensity of cell membrane part
        #   - Contour soma ratio
        
        Parameters
        ----------
        image : 2-D ndarray.
            Input image.
        MLresults : Dictionary.
            The returned dictionary from MaskRCNN.
        ImgNameInfor : String.
            Information of input image.
        cell_counted_number: int.
            Number of cells already counted in round.
        Returns
        -------
        Cell_DataFrame : pd.DataFrame.
            Detail information extracted from MaskRCNN mask from the image, in pandas dataframe format.
        cell_counted_number: int.
            Number of cells counted together with number from this image.
        """
        ROInumber = len(MLresults['scores']) 
        cell_counted_inImage = 0
        
        for eachROI in range(ROInumber):
            if MLresults['class_ids'][eachROI] == 3:
                ROIlist = MLresults['rois'][eachROI]
                CellMask = MLresults['masks'][:,:,eachROI]
                
                # If image size is larger than 1024X1024, it is resized before processed by MaskRCNN.
                # Here we need to resize the image to match the output mask from MaskRCNN.
                if image.shape[0] != CellMask.shape[0] or image.shape[1] != CellMask.shape[1]:
                    resized_img = resize(image,[CellMask.shape[0], CellMask.shape[1]],preserve_range=True).astype(image.dtype)
                    RawImg_roi = resized_img[ROIlist[0]:ROIlist[2], ROIlist[1]:ROIlist[3]] # Raw image in each bounding box
                else:
                    RawImg_roi = image[ROIlist[0]:ROIlist[2], ROIlist[1]:ROIlist[3]] # Raw image in each bounding box
                    
                CellMask_roi = CellMask[ROIlist[0]:ROIlist[2], ROIlist[1]:ROIlist[3]] # Individual cell mask in each bounding box
    
                # =============================================================
                #             # Find contour along cell mask
                # =============================================================
                cell_contour_mask = ProcessImage.findContour(CellMask_roi, RawImg_roi.copy(), 0.001) # Return the binary contour mask in bounding box.
                # after here intensityimage_intensity is changed from contour labeled with number 5 to binary image.
                cell_contour_mask_dilated = ProcessImage.inward_mask_dilation(cell_contour_mask, CellMask_roi, dilationparameter = 11)   
                
                #-------------Calculate intensity based on masks---------------
                cell_contour_meanIntensity = np.mean(RawImg_roi[np.where(cell_contour_mask_dilated == 1)]) # Mean pixel value of cell membrane.
                cell_area_meanIntensity = np.mean(RawImg_roi[np.where(CellMask_roi == 1)]) # Mean pixel value of whole cell area.
                
                cell_soma_mask = CellMask_roi - cell_contour_mask_dilated
                cell_soma_meanIntensity = np.mean(RawImg_roi[np.where(cell_soma_mask == 1)]) # Mean pixel value of soma area.         
                cell_contourSoma_ratio = round(cell_contour_meanIntensity/cell_soma_meanIntensity, 5) # Calculate the contour/soma intensity ratio.
                
                boundingbox_info = 'minr{}_maxr{}_minc{}_maxc{}'.format(ROIlist[0], ROIlist[2], ROIlist[1], ROIlist[3])

                if cell_counted_inImage == 0:
                    Cell_DataFrame = pd.DataFrame([[ImgNameInfor, boundingbox_info, cell_area_meanIntensity, cell_contour_meanIntensity, cell_contourSoma_ratio]], 
                                      columns = ['ImgNameInfor', 'BoundingBox', 'Mean_intensity', 'Mean_intensity_in_contour', 'Contour_soma_ratio'],
                                      index = ['Cell {}'.format(cell_counted_number)])
                else:
                    Cell_DataFrame_new = pd.DataFrame([[ImgNameInfor, boundingbox_info, cell_area_meanIntensity, cell_contour_meanIntensity, cell_contourSoma_ratio]], 
                                      columns = ['ImgNameInfor', 'BoundingBox', 'Mean_intensity', 'Mean_intensity_in_contour', 'Contour_soma_ratio'],
                                      index = ['Cell {}'.format(cell_counted_number)])                    
                    Cell_DataFrame = Cell_DataFrame.append(Cell_DataFrame_new)
                    
                cell_counted_number += 1
                cell_counted_inImage += 1
        
        if cell_counted_inImage == 0:
            return pd.DataFrame(), cell_counted_number
        else:
            return Cell_DataFrame, cell_counted_number
        
        
    def Convert2Unit8(Imagepath, Rawimage):
        """ Convert image pixel values to unit8 to run on MaskRCNN.
        """
        if Imagepath[len(Imagepath)-3:len(Imagepath)] == 'tif':
            """ set image data type to unit8
            """
            image = Rawimage * (255.0/Rawimage.max())
            image=image.astype(int)+1
        
            if len(np.shape(image)) == 2:
                image = gray2rgb(image)
    
            return image
        
        else:
            return Rawimage
        
    #%%
if __name__ == "__main__":
    
    from skimage.io import imread
    import time
    from IPython import get_ipython

#    speedGalvo = 20000.0 #Volt/s
#    AccelerationGalvo = 1.54*10**8 #Acceleration galvo in volt/s^2
#    #--------------------------------------------------------------------------
#    PMT_image = imread(r'D:\XinMeng\imageCollection\Round2_Coord3_R1500C1500_PMT_2.tif', as_gray=True)
##    time_gap = 1/50000
#     
#    RegionProposalMask, RegionProposalOriginalImage = ProcessImage.generate_mask(PMT_image, openingfactor=2, 
#                                                                                                closingfactor=4, binary_adaptive_block_size=335)#256(151) 500(335)
#
#    CellSkeletonizedContourDict= ProcessImage.get_Skeletonized_contour(PMT_image, RegionProposalMask, smallest_size=400, contour_thres=0.001, 
#                                                                                       contour_dilationparameter=11, cell_region_opening_factor=1, cell_region_closing_factor=2,
#                                                                                       scanning_voltage=5, points_per_contour=500, sampling_rate = 50000)
            
        
# =============================================================================
    tag_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-05-12 Archon lib 400FOVs 4 grid\trial_1'
    lib_folder = r'D:\XinMeng\imageCollection\Fov3\New folder (3)'
  #   tag_folder = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-3-6 Archon brightness screening\NovArch library'

    tag_round = 'Round1'
    lib_round = 'Round4'
    
    EvaluatingPara_1 = 'Mean intensity divided by tag'
    EvaluatingPara_2 = 'Contour soma ratio'
    
    MeanIntensityThreshold = 0.16
    
    starttime = time.time()
    
    tagprotein_cell_properties_dict = ProcessImage.TagFluorescenceAnalysis(tag_folder, tag_round, Roundness_threshold = 2.1)
    print('tag done.')
    
    tagprotein_cell_properties_dict_meanIntensity_list = []
    for eachpos in tagprotein_cell_properties_dict:
        for i in range(len(tagprotein_cell_properties_dict[eachpos])):
            tagprotein_cell_properties_dict_meanIntensity_list.append(tagprotein_cell_properties_dict[eachpos]['Mean intensity'][i])
            
        
#    lib_cell_properties_dict = ProcessImage.LibFluorescenceAnalysis(lib_folder, tag_round, lib_round, tagprotein_cell_properties_dict)
#    print('lib done.')
#    
#    # Devided by fusion protein brightness.
#    lib_cell_properties_dict = ProcessImage.CorrectForFusionProtein(tagprotein_cell_properties_dict, lib_cell_properties_dict, tagprotein_laserpower=1, lib_laserpower=30)
#    # Organize and add 'ranking' and 'boundingbox' fields to the structured array.
#    Overview_LookupBook = ProcessImage.OrganizeOverview(lib_cell_properties_dict, ['Mean intensity in contour', 0.16, 'Contour soma ratio', 0.8], EvaluatingPara_1, 1, EvaluatingPara_2, 0.5)
#    #--------------------------------------------------------------------------
##    Overview_LookupBook_sorted = ProcessImage.WeightedSorting(Overview_LookupBook, 'Mean intensity divided by tag', 'Mean intensity in contour', 'Contour soma ratio', 
##                                                              weight_1 = 0.4, weight_2 = 0.4, weight_3 = 0.2)
#    #--------------------------------------------------------------------------
#    selectionRadius = 'circle'
#    selectionPercent = 100
#    
#    Overview_LookupBook_filtered = ProcessImage.DistanceSelecting(Overview_LookupBook, selectionPercent)
#
#    totalselectnum = 5
    
    
#    fig = px.scatter(Overview_LookupBook, x=EvaluatingPara_1, y=EvaluatingPara_2, 
#               hover_name= 'ID', color= 'Normalized distance', 
#               hover_data= ['Sequence', 'Mean intensity'], width=1050, height=950)
#    fig.write_html('Screening scatters.html', auto_open=True)
    # Display scatters
#    get_ipython().run_line_magic('matplotlib', 'qt')
#    plt.scatter(Overview_LookupBook[EvaluatingPara_1], Overview_LookupBook[EvaluatingPara_2], s=np.pi*3, c='blue', alpha=0.5)
#    plt.scatter(Overview_LookupBook_filtered[EvaluatingPara_1], Overview_LookupBook_filtered[EvaluatingPara_2], s=np.pi*3, c='red', alpha=0.5)
#    plt.title('Screening scatter plot')
#    plt.xlabel(EvaluatingPara_1)
#    plt.ylabel(EvaluatingPara_2)
#    plt.show()
#    
##    get_ipython().run_line_magic('matplotlib', 'inline')
#
##    Top_from_Overview_LookupBook = Overview_LookupBook_sorted[0:20]
#    ranking = 1
#    for EachCell in range(len(Overview_LookupBook_filtered)):
#        spec = Overview_LookupBook_filtered[EachCell]['ID']
##        #-------------- readin image---------------
##        for file in os.walk(lib_folder):
##            if spec and 'Zmax' in file:
#        lib_imagefilename = os.path.join(lib_folder, spec+'_PMT_0Zmax.tif')
##                break
##            break
#        print(lib_imagefilename)
#        loaded_lib_image_display = imread(lib_imagefilename, as_gray=True)
#        # Retrieve boundingbox information
#        Each_bounding_box = Overview_LookupBook_filtered[EachCell]['BoundingBox']
#        minr = int(Each_bounding_box[Each_bounding_box.index('minr')+4:Each_bounding_box.index('_minc')])
#        maxr = int(Each_bounding_box[Each_bounding_box.index('maxr')+4:Each_bounding_box.index('_maxc')])        
#        minc = int(Each_bounding_box[Each_bounding_box.index('minc')+4:Each_bounding_box.index('_maxr')])
#        maxc = int(Each_bounding_box[Each_bounding_box.index('maxc')+4:len(Each_bounding_box)])
#        
##        plt.figure()
#        fig_showlabel, ax_showlabel = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#        ax_showlabel.imshow(loaded_lib_image_display)#Show the first image
#        #--------------------------------------------------Add red boundingbox to axis----------------------------------------------
#        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='cyan', linewidth=2)
##        contour_mean_bef_rounded = str(round(MeanIntensity_Contour, 3))[0:5]
#        ax_showlabel.add_patch(rect)
#        ax_showlabel.text(maxc, minr, 'NO_{}'.format(ranking),fontsize=8, color='orange', style='italic')
#        ranking += 1
#        # Based on the boundingbox for each cell from first image in the stack, raw image of slightly larger region is extracted from each round.
##        RankdisplayImg = loaded_lib_image_display[minr:maxr, minc:maxc] # Raw region image 
#        
#        ax_showlabel.set_axis_off()
#        plt.show()
# =============================================================================


                