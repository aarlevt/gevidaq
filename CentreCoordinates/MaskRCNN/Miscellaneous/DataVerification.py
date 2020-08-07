# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:00:51 2020

@author: meppenga
"""

import json
import logging
import os
import MaskRCNN.Miscellaneous.Misc as Misc


class dataVerification():
    def __init__(self,config):
        """Class to verify if data sets are complete and use the correct format
        Author: Martijn Eppenga"""
        self.ValidLabels = config.ValidLabels
        self.config      = config
    
    def CheckExistImage(self,file, PrintError=False,LogError=False,ReturnImagePath=False):
        """This function verifies if the image specified within a given json 
        file exist
        Input:
            file:       str, File to vefiy, must end with .json
            PrintError: boolean (optional, default False), when true prints error message when image file does not exist
            LogError, boolean (optional, default False), when true logs error message when image file does not exist
        Return:
            boolean, True when image file exist"""
        file = Misc.fixPathName(file)
        with open(file) as datafile:
            data = json.load(datafile)
        try:
            # Verify if json file contains an image path
            Image = data['imagePath']
        except:
            if PrintError:
                Misc.printColor('File '+file+'is not a readable file')
            if LogError:
                logging.warning('File '+file+'is not a readable file')
            if ReturnImagePath:
                return False, ''
            else:
                return False
        directory,_ = file.rsplit('\\',1)
        if os.path.isfile(directory+'\\'+Image):
            # search for image
            if ReturnImagePath:
                return True, directory+'\\'+Image
            else:
                return True
        else:
            if PrintError:
                Misc.printColor('warning','Cannot find image file: '+directory+'\\'+Image)
                Misc.printColor('warning','Json file is: '+file)
            if LogError:
                logging.warning('Cannot find image file: '+directory+'\\'+Image)
                logging.warning('Json file is: '+file)
            if ReturnImagePath:
                return False, ''
            else:
                return False

    def CheckLabels(self,file, PrintError=False,LogError=False,ReturnIndex=False):
        """This function verifies if the labels within a given json file are
        valid labels according to the config file
        Input:
            file:       str, File to vefiy, must end with .json
            PrintError: boolean (optional, default False), when true prints error message when encoutered invalidd image
            LogError, boolean (optional, default False), when true logs error message when encoutered invalidd image
            ReturnIndex: boolean, when true, returns list with booleans for each found label 
        Return:
            AllDataValid: boolean, True when image file exist
            ValidIndex:   list of booleans (only if ReturnIndex= True), True for valid labels, False for invalid labels"""
        with open(file) as datafile:
            data = json.load(datafile)
        try:
            # Verify if json file contains labels
            polygons = data['shapes']
        except:
            if PrintError:
                Misc.printColor('File '+file+'is not a readable file')
            if LogError:
                logging.warning('File '+file+'is not a readable file')
            if ReturnIndex: 
                return False, False
            else:
                return False
            
        numpolygons  = len(polygons)
        AllDataValid = True
        ValidIndex   =  [True for tel in range(numpolygons)] 
        for ii in range(numpolygons):
            # Check each label
            label = polygons[ii]['label']
            if not label in self.ValidLabels:
                if PrintError:
                    if AllDataValid:
                        # First error, specify file
                        print('Error in file: '+file)
                    Misc.printColor('warning','Invalid label. Label is: '+label)
                if LogError:
                    if AllDataValid:
                        # First error, specify file
                        logging.warning('Error in file: '+file)
                        logging.warning('Invalid label. Label is: '+label)
                AllDataValid   = False
                ValidIndex[ii] = False
        if ReturnIndex:
            return AllDataValid, ValidIndex
        else:
            return AllDataValid

                    
    def CheckJsonFile(self,file,PrintError=False,LogError=False,ReturnIndex=False):
        """Verfies if it can find the image specified in a json file and if it 
        all labels are valid according to the config file
        Input:
            file:       str, File to vefiy, must end with .json
            PrintError: boolean (optional, default False), when true prints error message when file has invalid label and or image
            LogError, boolean (optional, default False), when true logs error message when file has invalid label and or image
            ReturnIndex: boolean, when true, returns list with booleans for each found label 
        Return:
            ImageExist: boolean, True when json file contains valid path to image and all labels are valid
            ValidIndex: list of booleans (only if ReturnIndex= True), True for valid labels, False for invalid labels"""
        ImageExist = self.CheckExistImage(file,PrintError,LogError)
        if ReturnIndex:
            LabelValid, Index = self.CheckLabels(self,file, PrintError,LogError,ReturnIndex)
            return ImageExist and LabelValid, Index
        else:
            LabelValid = self.CheckLabels(file, PrintError,LogError,ReturnIndex)
            return ImageExist and LabelValid

 
    def CheckDirectory(self,directories,CheckSubdirectories=[False],PrintError=True,LogError=False):
        """Verfies for each json file in a directory if it contains a valid image path and
        if it contains valid labels according to config file
        Input:
            directories: list of str, list with paths to directories to verfify
            CheckSubdirectories: list of booleans (optional default [False]), When true at index ii  will check all files in the subdirectory
                                    of directories at index ii. If list has only one boolean it will use this boolean for all directories in directories
            PrintError: boolean (optional, default True), when true prints error message when file has invalid label and or image
            LogError, boolean (optional, default False), when true logs error message when file has invalid label and or image"""
        if not isinstance(directories, list):
            directories = [directories]
        if not isinstance(CheckSubdirectories, list):
            CheckSubdirectories = [CheckSubdirectories]    
        if len(CheckSubdirectories) != len(directories):
            if len(CheckSubdirectories) != 1:
                raise Exception('CheckSubdirectories and directories must have the same length:'+
                                'length CheckSubdirectories: %2d length directories' % (len(CheckSubdirectories), len(directories)))
            CheckSubdirectories = [CheckSubdirectories[0] for ii in range(len(directories))]
        
        for directory, CheckSubdirectory in zip(directories,CheckSubdirectories):
            AllFilesValid   = True
            numFilesValid   = 0
            numFilesInvalid = 0
            # Check if directory exist
            if not os.path.isdir(directory):
                Misc.printColor('warning','Directory '+directory+' does not exist')
            else:
                for file in os.listdir(directory):
                    FileFullPath = directory+'/'+file
                    if os.path.isdir(FileFullPath):
                        # Check if file is a directory
                        if CheckSubdirectory:
                            # loop over subdirectories
                            self.CheckDirectory([FileFullPath],CheckSubdirectories=[True],PrintError=PrintError,LogError=LogError)
                    elif file.endswith('.json'):
                        filevalid = self.CheckJsonFile(FileFullPath,PrintError,LogError,ReturnIndex=False)
                        if filevalid:
                            numFilesValid += 1
                        else:
                            numFilesInvalid += 1
                        AllFilesValid = AllFilesValid and filevalid                
            if AllFilesValid: 
                msg = 'All %2d json files valid in directory ' % (numFilesValid)
                print(msg+directory )
            else:
                msg = 'Directory contains %2d invalid json files ' % (numFilesInvalid)
                Misc.printColor('warning',msg+directory )

   