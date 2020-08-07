# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:26:16 2020

@author: meppenga
"""
import os  
from MaskRCNN.Engine.MaskRCNN import MaskRCNN as modellib
import logging
import numpy as np
import skimage.io
import MaskRCNN.Miscellaneous.visualize as visualize
import matplotlib.pyplot as plt
from CustomDataLoader import CellDataSetLoader
from Timer import Timer
from MaskRCNN.Miscellaneous.Misc import printColor, updateProgressBar, fixPathName
import MaskRCNN.Miscellaneous.utils as ut

class Detect(CellDataSetLoader):
    
    def __init__(self,config, SaveConfig):
        """This class can run the MaskRCNN software on an image file to detect
        instances within the image"""
        CellDataSetLoader.__init__(self,config)
        self.config     = config
        self.ChangeSaveConfig(SaveConfig)             
        
        # Constants
        self.Images       = [] # Image paths stored as [directory,imagename]
        self.NumImages    = 0 # Number of images loaded in to the data holder
        self.numDetection = 0 # Number of detections run
        self.numSave      = 0 # Number of images saved
        self.DetectionTimeEachImage = [] # Log the detection time for each detection
        
    
        
        self.RemoveDoubleDetectedCells  = self.config.RemoveDoubleDetectedCells
        self.DoubleDetectedThersholdIoU = self.config.DoubleDetectedThersholdIoU
        
        self.loadModel()
        self.TimeObj = Timer()
        
        #Remove
        self.figsize       = (20,10) # figure size used to show and save images in inches
    
   
    def loadModel(self):
        """Function wich will load a MaskRCNN model. All information will be taken
        form the config file"""
        self.model = modellib(mode="inference", model_dir=self.config.LogDir, config=self.config)
        self.model.LoadWeigths(self.config.WeigthPath, by_name=True)
        
    def setFigureSize(self,width, height):
        """Set the scale of the figures used to display and save images
        Input:
            width: float, width of the images in inches
            height: float, height of the images in inches"""
        self.SaveConfig.FigSize = (width,height)
        self._InitFigures()
    
    def ChangeSaveConfig(self,SaveConfig):
        self.SaveConfig = SaveConfig
        self.SetupSaveDir()
        self._InitFigures()
    
    def SetupSaveDir(self):
        """Create save directory"""
        if self.SaveConfig.CreateSaveDir:
            if not os.path.isdir(self.SaveConfig.SaveDir):
                os.mkdir(self.SaveConfig.SaveDir)
            print('Detected Images save directory: '+self.SaveConfig.SaveDir)
            logging.info('Detected Images save directory: '+self.SaveConfig.SaveDir)
            self._saveConfig()
            self._CreateInfoFile()
       
    def endDetection(self):
        """Call this function when you are done with detection. This function
        will save the info stored in the _saveInfoDict"""
        logging.info('Detecion run on %3d images' % (self.numDetection))
        logging.info('Number of saved images: ' + str(self.numSave))
        logging.info('Detecion finished')
        
    def _fixPathName(self, Path,endwith=''):
        """This function eensures that the path name for a file or directory has
        a uniform format. It will makes sure that the path is always displayed
        as name1\\name2\\name3... We use here \\ but it will be become a single 
        one.
        Input: 
            Path:    str, a string which represent the path to a directory or 
                     file
            endwith: str, optional, if given the funtion will ensure that the 
                     input path ends with these charachters
        Return:
            Path: str, a string with the path to a file or directory using a 
                  uniform format"""
        return fixPathName(Path,endwith='')  
    
    def _InitFigures(self):
        self.fig1,  self.ax11                        = plt.subplots(1, 1, figsize=self.SaveConfig.FigSize) 
        self.fig2, (self.ax21, self.ax22)            = plt.subplots(1, 2, figsize=self.SaveConfig.FigSize) 
        self.fig3, (self.ax31, self.ax32, self.ax33) = plt.subplots(1, 3, figsize=self.SaveConfig.FigSize) 
        plt.close(self.fig1)
        plt.close(self.fig2)
        plt.close(self.fig3)
        self.ax11.axis('off')
        self.ax21.axis('off')
        self.ax22.axis('off')
        self.ax31.axis('off')
        self.ax32.axis('off')
        self.ax33.axis('off')
        
    
    ################ Info save methods ###############################
    def getImageNameAndDirectory(self,ImagePath):
        ImagePath = self._fixPathName(ImagePath)
        try:
            Directory,ImageName = os.path.split(ImagePath)
            _, Directory = os.path.split(Directory)
            if Directory == 'Training':
                Directory = 'Train'
            elif Directory == 'Validation':
                Directory = 'Val'
            else:
                Directory = 'None'
        except:
             _,ImageName  = os.path.split(ImagePath)
             Directory    = 'None'

        ImageName, _ = ImageName.rsplit('.',1)
        return ImageName, Directory   
    
    def _saveConfig(self):
        """Create a file called Config.txt in the default save directory
        This file contains all the setting of the configuration file"""
        if not self.SaveConfig.CreateSaveDir:
            return
        ConfigSavepath = os.path.join(self.SaveConfig.SaveDir,'Config.txt')
        with open(ConfigSavepath,'a') as fileobj:
            for setting in dir(self.config):
                if not setting.startswith("__") and not callable(getattr(self.config, setting)):
                    fileobj.write("{:30} {}".format(setting, getattr(self.config, setting))+'\n')
        logging.info('Saved conifg file: '+ConfigSavepath)



        
        
        
    def _CreateInfoFile(self):
        if not self.SaveConfig.CreateSaveDir:
            return
        self.infoFilePath = os.path.join(self.SaveConfig.SaveDir,'InfoFile.txt')
        with open(self.infoFilePath ,'a') as fileobj:
            fileobj.write('Info file with information about each detection')
            fileobj.write('\nSave directory: '+self.SaveConfig.SaveDir)
            fileobj.write('\n\n')
    
    def LogInfoDetection(self,ClassIds,SaveName='Not saved',FilePath='No file path',NumRemovedObjects=0):
        if not self.SaveConfig.CreateSaveDir:
            return
        try:
            _, filenamesave = os.path.split(SaveName)
        except:
            filenamesave = SaveName
        with open(self.infoFilePath,'a') as fileobj:
            fileobj.write('File: '+filenamesave)
            fileobj.write('\nOriginal file path: '+FilePath)
            fileobj.write('\nDetection Time (ms): '+str(self.DetectionTimeEachImage[-1]))
            if self.RemoveDoubleDetectedCells:
                fileobj.write('\nRemoved double detections. Threshold IoU: '+str(self.DoubleDetectedThersholdIoU))
                fileobj.write('\nNumber of removed double detections: '+str(NumRemovedObjects))
            fileobj.write('\nNumber of objects:   '+str(len(ClassIds)))
            labelcounter = np.zeros(len(self.config.ValidLabels)+1,dtype=np.int32)
            for ClassId in ClassIds:
                labelcounter[ClassId] += 1
            for ii in range(len(self.config.ValidLabels)):
                fileobj.write('\nNumber of objects with label '+self.config.ValidLabels[ii]+': '+str(labelcounter[ii+1]))
            fileobj.write('\n\n')
                
    

    def CreateSavePathImage(self,ImageName, Directory='',SavePath=None, FileFormat='.png'):
        if SavePath == None:
            if not self.SaveConfig.CreateSaveDir:
                raise Exception('There is no directory to save the results.\n'+
                       'Use the "CreateSaveDirectory" method to create a save directory.\n'+
                       'You can specify the save directory in the config file')
            SavePath = self.SaveConfig.SaveDir
        num = '%03d'%(int(self.numSave))
        self.numSave += 1
        if '.' in ImageName:
            ImageName,_ = ImageName.rsplit('.',1)
            try:
                _, ImageName = os.path.split(ImageName)
            except: 
                pass
        if not '.' in FileFormat:
            FileFormat = '.'+FileFormat
        SaveFileName = num + Directory + ImageName + FileFormat
        return os.path.join(SavePath, SaveFileName)    
    
    def runDetectionAllImages(self):
        """This function will run the MaskRCNN software an all the images
        stored in the self.Images variable. One can use the addImagePaths method
        to add iamges to the self.Images variable.
        Input:
            SaveImage: boolean (optional, default = True), if true, the function
                       will save the result of each detection in the default 
                       save folder (see _CreateSavePath
            ShowImage: boolean (optional, default = False), if true, the function
                       will create an image window and show the result of a each
                       detection. Not recommended when using many images
            SaveWithOrgImage: boolean (optional default = True), if true the 
                       function will add the origional image to the saved and or
                       displayed image
            SaveWithOrgMask: boolean (optional default = False), if true will 
                       include image with the masks and bounding boxes in the
                       saved image
           NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object"""
        if not self.SaveConfig.CreateSaveDir:
            raise Exception('There is no directory to save the results.\n'+
                       'Use the "CreateSaveDirectory" method to create a save directory.\n'+
                       'You can specify the save directory in the config file')
            
        print('Run detection on %2d Images' % (self.NumImages))
        logging.info('Run detection on %2d Images' % (self.NumImages))
        self.TimeObj.tic()

        for ii in range(len(self.Images)):
            directory = self.Images[ii][0]
            for Image in self.Images[ii][1]:
                ImagePath = os.path.join(directory, Image)
                if os.path.isfile(ImagePath):
                    if self._ExistJsonForImagePath(ImagePath):
                        self.DetectFromImagePathAndSave(ImagePath, ShowImage=False)
                    else:
                        self.DetectFromImagePathAndSave(ImagePath, ShowImage=False)
                updateProgressBar(self.numDetection/self.NumImages, message = 'Running detection on '+str(self.NumImages)+' images. Progress')
        logging.info('Detection Done')
        logging.info('Time to Run detection: '+self.TimeObj.toc())
        print('Time to Run detection: '+self.TimeObj.toc())
                    
    
    
        
    
    ################ Detect and show ###############################
    
    def DetectAndShow(self,Imagepath):
        """This function will run the maskRCNN software on a single image and
        will create a figure to display the result
        Input: 
            Imagepath: str, path to an image file to run the detection on
            ShowOrgImage: Displays the orginal input image next to the detection
            NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object"""
        Results, Image = self.RunDetectionOnImageFile(self._fixPathName(Imagepath))
        print(len(Results['class_ids']))
        self.ShowResultDetection(Results,Image)
        
    
    def DetectAndShowImage(self,Image):
        """This function will run the maskRCNN software on a single image and
        will create a figure to display the result
        Input: 
            Imagepath: str, path to an image file to run the detection on
            ShowOrgImage: Displays the orginal input image next to the detection
            NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object"""
        Results,Image = self.RunDetectionOnImage(Image)
        self.ShowResultDetection(Results,Image)

    def ShowResultDetection(self,Result,Image):
        """This function displays the Results of an detection
        Input: 
            Results: list, results of a detection use RunDetectionOnImageFile or
                      RunDetectionOnImage method to create the result and take only 
                      first item of the returned result: 
                          Results = RunDetectionOnImage(Image)
                          ShowResultDetection(Results[0],Image)
             Image: image on which the detection has run
             ShowOrgImage: Displays the orginal input image next to the detection
             NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object""" 
        
        if len(np.shape(Image)) != 3:
            raise Exception('Image should be of RGB type. Use the image returned by the detection as input for the image of this function')
        if not self.SaveConfig.Captions:
            captions = ['' for ii in range(len(Result['class_ids']))]
        else:
            captions = None
        if self.SaveConfig.dispCentreCoor:
            centreCoor = Result['Centre_coor']
        else:
            centreCoor = []
        if self.SaveConfig.Add_orgIm:
            self.ax21.clear()
            self.ax22.clear()
            self.ax21.axis('off')
            self.ax22.axis('off')
            visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'], 
                                            ['BG'] + self.config.ValidLabels, Result['scores'], 
                                            centre_coors= centreCoor, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                            ax= self.ax21,captions=captions)
            self.ax22.imshow(Image)
            self.fig2.show()
        else:
            self.ax11.clear()
            self.ax11.axis('off')
            visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'], 
                                            ['BG'] + self.config.ValidLabels, Result['scores'], 
                                            centre_coors= centreCoor, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                            ax= self.ax1,captions=captions)
            self.fig1.show()

        
        
        
    ############## Detect and save ###########################
    
    def DetectFromImageAndSave(self,Image,SaveName,ShowImage=False):
        if not self.SaveDirectory:
            raise Exception('warning','There is no directory to save the results.\n'+
                       'Use the "CreateSaveDirectory" method to create a save directory.\n'+
                       'You can specify the save directory in the config file')
 
        ImageName,_   = self.getImageNameAndDirectory(SaveName)
        Results,Image = self.RunDetectionOnImage(Image,filename=SaveName)

        self.SaveImageResult(Results,Image,SaveName,ShowImage=ShowImage)
        
    
    
    def DetectFromImagePathAndSave(self,ImagePath,ShowImage=False,SaveName=None):
        """This function will run the MaskRCNN software on a single image and
        saves the result in the default direcory (see _CreateSavePath), You can 
        change the directory for one image by uisng the optional SavePath input
        Inputs:
            ImagePath: str, path to the image to run the detection on
            ShowImage: boolean (optional, default = False), if true, the function
                       will create an image window and show the result of a each
                       detection. Not recommended when using many images
            SavePath:  str, (optional default = None), If specified, the function 
                       will save the result in this directory. The directory must
                       exist. If not specified it will use the default direcory
                       created using the _CreateSavePath method
            SaveWithOrgImage: boolean (optional default = True), if true the 
                       function will add the origional image to the saved and or
                       displayed image
           SaveWithOrgMask: boolean (optional default = False), if true will 
                       include image with the masks and bounding boxes in the
                       save image
           NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object
           
            """
        if not self.SaveConfig.CreateSaveDir:
            if not SaveName:
                raise Exception('warning','There is no directory to save the results.\n'+
                       'Use the "CreateSaveDirectory" method to create a save directory.\n'+
                       'You can specify the save directory in the config file')

        ImagePath = self._fixPathName(ImagePath)

        
        if SaveName == None:
            ImageName, Directory = self.getImageNameAndDirectory(ImagePath)
            SaveName  = self.CreateSavePathImage(ImageName,Directory)
        _,ImageName = SaveName.rsplit('\\',1)
        ImageName,_ = ImageName.rsplit('.',1)
        
        Results, Image = self.RunDetectionOnImageFile(ImagePath,SaveName=SaveName)
 
        if self.SaveConfig.dispMask and self._ExistJsonForImagePath(ImagePath):
            JsonPath, _ = ImagePath.rsplit('.',1)
            JsonPath += '.json'
            Mask, Label, CentreCoor = self.ReturnMaskAndLabel(JsonPath)
            coor = []
            for ii in range(len(Label)):
                coor.append([CentreCoor[ii,0],CentreCoor[ii,1]])
            MaskAndLabel = [Mask, Label, coor]
        else:
            MaskAndLabel=[None]
        self.SaveImageResult(Results,Image,SaveName,ShowImage=ShowImage, MaskAndLabel=MaskAndLabel)

    

        
        
  
    
    
    def SaveImageResult(self,Result,Image,SaveName,ShowImage=False, MaskAndLabel=[None], CustomCaptions=None):
        """
        Results: The results of a detection
        ShowImage: boolean (optional, default = False), if true, the function
                       will create an image window and show the result of a each
                       detection. Not recommended when using many images
            SaveName:  str, name of file where to save the result
            SaveWithOrgImage: boolean (optional default = True), if true the 
                       function will add the origional image to the saved and or
                       displayed image
           SaveWithOrgMask: boolean (optional default = False), if true will 
                       include image with the masks and bounding boxes in the
                       save image
        NoCaptions: boolean (optional default = False), if True then the result 
                       of a detection will be shown with labels and scores for each
                       detected object"""
        if not self.SaveConfig.CreateSaveDir:
            raise Exception('warning','There is no directory to save the results.\n'+
                       'Use the "CreateSaveDirectory" method to create a save directory.\n'+
                       'You can specify the save directory in the config file')
        if len(MaskAndLabel) == 1:
            SaveWithMask = False
            numImages = 1+int(self.SaveConfig.Add_orgIm)
        else:
            SaveWithMask = True
            MaskAndLabel[0] = self.ResziseMask(MaskAndLabel[0], np.shape(Image))
            numImages       = 2+int(self.SaveConfig.Add_orgIm)

        
        if not self.SaveConfig.Captions:
            captions1 = ['' for ii in range(len(Result['class_ids']))]
            if SaveWithMask:
                captions2 = ['' for ii in range(len(MaskAndLabel[1]))]
            else:
                captions2 = None
        else:
            captions1 = None
            captions2 = None
        if CustomCaptions:
            captions1 = CustomCaptions
        if self.SaveConfig.dispCentreCoor:
            CentreCoors1 = Result['Centre_coor']
            if SaveWithMask:
                CentreCoors2 = MaskAndLabel[2]
            else:
                CentreCoors2 = []
        else:
            CentreCoors1 = []
            CentreCoors2 = []
                

              
        if numImages == 2:
            self.ax21.clear()
            self.ax22.clear()
            self.ax21.axis('off')
            self.ax22.axis('off')
            visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'],
                                        ['BG'] + self.config.ValidLabels, Result['scores'],
                                        centre_coors = CentreCoors1, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                        captions = captions1,
                                        ax=self.ax21)
            if SaveWithMask:
                visualize.display_instances(Image, self.CreateBBoxes(MaskAndLabel[0]), MaskAndLabel[0],MaskAndLabel[1],
                                        ['BG'] + self.config.ValidLabels, 
                                        centre_coors = CentreCoors2, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                        captions = captions2, ax=self.ax22)
            else: 
                self.ax22.imshow(Image)
            self.ax21.title.set_text('Detection')
            if SaveWithMask:
                self.ax22.title.set_text('Ground truth')
            else:
                self.ax22.title.set_text('Input image')
            self.fig2.savefig(SaveName)
            if ShowImage:
                self.fig2.show()
            else:
                plt.close(self.fig2)
        elif numImages == 3:
            self.ax31.clear()
            self.ax32.clear()
            self.ax33.clear()
            self.ax31.axis('off')
            self.ax32.axis('off')
            self.ax33.axis('off')
            visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'],
                                        ['BG'] + self.config.ValidLabels, Result['scores'],
                                        centre_coors = CentreCoors1, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                        captions = captions1, ax=self.ax31)
            visualize.display_instances(Image, self.CreateBBoxes(MaskAndLabel[0]), MaskAndLabel[0],MaskAndLabel[1],
                                        ['BG'] + self.config.ValidLabels, 
                                        centre_coors = CentreCoors2, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                        captions = captions2, ax=self.ax32)
            self.ax33.imshow(Image)
            self.ax31.title.set_text('Detection')
            self.ax32.title.set_text('Ground truth')
            self.ax33.title.set_text('Input image')
            self.fig3.savefig(SaveName)
            if ShowImage:
                self.fig3.show()
            else:
                plt.close(self.fig3)
        else:
            self.ax11.clear()
            self.ax11.axis('off')
            self.ax11, fig = visualize.display_instances(Image, Result['rois'], Result['masks'], Result['class_ids'],
                                        ['BG'] + self.config.ValidLabels, Result['scores'],
                                        centre_coors = CentreCoors1, Centre_coor_radius = self.SaveConfig.CenterCoorRadius,
                                        captions = captions1, ax=self.ax11)
            self.fig1.savefig(SaveName)
            if ShowImage:
                self.fig1.show()
            else:
                plt.close(self.fig1)
        logging.info('Saved detection as: '+SaveName) 



####################################################################################################



    ############### Detection Functions #################

        
    def RunDetectionOnImageFile(self,ImagePath,SaveName='Not saved'):
        """This function will run the MaskRCNN software on an single image to detect 
        insatnces. The function will return the detections and the orgional image
        Input:
            ImagePath: str, string to the image to run the detection on
        Return:
            Results: list containing hte results of the detection
            image: image on which the detection is run"""
        if not os.path.isfile(ImagePath):
            raise Exception('Cannot run detection since file '+ImagePath+' does not exist')
        ImagePath = self._fixPathName(ImagePath)
        return self.RunDetectionOnImage(skimage.io.imread(ImagePath),SaveName=SaveName,FilePath=ImagePath)
    
    
    
    def RunDetectionOnImage(self,Image,SaveName='Not saved',FilePath='No file path'):
        """This function will run the MaskRCNN software on an single image to detect 
        insatnces. The function will return the detections and the orgional image
        Input:
            image: image to run detection on
        Return:
            Results: list containing hte results of the detection"""
        if self.config.IMAGE_RESIZE_MODE == "pad64":
            Image = self.PrepairImageForDetection(Image)
        else:
            Image = ut.ConvertImage2RGB(Image)
        self.numDetection += 1
        Tstart = self.TimeObj.toc_ms()
        Result = self.model.detect([Image], verbose=0)[0]
        if self.RemoveDoubleDetectedCells:
            Result, NumRemoved = self._RemoveDuplicateDetections(Result)
        else:
            NumRemoved = 0
        self.DetectionTimeEachImage.append(self.TimeObj.toc_ms() - Tstart)
        self.LogInfoDetection(Result['class_ids'],SaveName=SaveName,FilePath=FilePath,NumRemovedObjects=NumRemoved)
        return Result, Image
    
     ############### Helper methods #################
    
    def PrepairImageForDetection(self,Image):
        """Ensure that an image has the correct size and color format for a detection
        Input:
            Image: array like, Image to run the detection on"""
        return ut.ConvertImage2RGB(self._resizeImage(Image))
    
    
    def _resizeImage(self,Image):
        """Resize an image such that it format is less then or equal to the image
        size specified in the config file"""
        shape = np.shape(Image)
        if shape[0] > self.config.IMAGE_MAX_DIM or shape[1] > self.config.IMAGE_MAX_DIM:
            logging.info('Rescaled image')
            return skimage.transform.resize(Image,[self.config.IMAGE_MAX_DIM,self.config.IMAGE_MAX_DIM],preserve_range=True).astype(Image.dtype)
        else:
            return Image
    
    def ResziseMask(self,mask,imageshape):
        shapeMask = np.shape(mask)
        if len(shapeMask)  == 2: 
            shapeMask.append(1)
        if shapeMask[0] != imageshape[0] or shapeMask[1] != imageshape[1]:
            maskResize = np.zeros((imageshape[0], imageshape[1], shapeMask[2]),dtype=bool)
            for ii in range(shapeMask[2]):
                maskResize[:,:,ii] = skimage.transform.resize(mask[:,:,ii],[imageshape[0],imageshape[1]],preserve_range=True).astype(bool)
            return maskResize
        return mask
    
    
    
    
    def _correctImageScale(self,Image):
        minval = np.min(Image)
        maxval = np.max(Image)
        return ((Image-minval)/(maxval-minval)*255).astype(np.uint8)
    
        
    def addImagePaths(self,directories,usesubdirectories=[False],specifier=['png'],Exlude=[None]):
        """This function will add all images from a directory and optional from 
        it subdirectories to the self.Images list
        Input: 
            directories: list of str, List containing the path to directories from
                         which to search for images
            usesubdirectories: list of booleans (optional, default is [False])
                         List containing booleans to specify if the function can 
                         search in the subdirectory for each directory specified
                         in the directories input. 
                         If only one boolean is given it will use this value for
                         all directoriesin the directories input
            specifier:   list of str (optional deffault = ['png']), list containing 
                         the image types to search for
                         """
        if not isinstance(specifier, list):
            specifier = [specifier]
        if not isinstance(directories, list):
            directories = [directories]
        if not isinstance(usesubdirectories, list):
            usesubdirectories = [usesubdirectories]
        if len(usesubdirectories) != len(directories):
            if len(usesubdirectories) != 1:
                raise Exception('CheckSubdirectories and directories must have the same length:'+
                                    'length usesubdirectories: %2d length directories' % (len(usesubdirectories), len(directories)))
            usesubdirectories = [usesubdirectories[0] for ii in range(len(directories))]
        for directory, CheckSubdirectory in zip(directories,usesubdirectories):
            directory = self._fixPathName(directory)
            ImageList = []
            for file in os.listdir(directory):
                fileFullpath = os.path.join(directory,file)
                if os.path.isdir(fileFullpath) and CheckSubdirectory:
                    if not file in Exlude:
                        self.addImagePaths([fileFullpath],[CheckSubdirectory],specifier)  
                else:
                    for spec in specifier:
                        if file.endswith(spec):
                            ImageList.append(file)
                            self.NumImages += 1
            if len(ImageList) > 0:
                ImageHolder = [directory, ImageList]
                self.Images.append(ImageHolder)



    def _ExistJsonForImagePath(self,ImagePath):
        """Checks if a json file exist for given image path
        if true one can use the _GetOrgImageWithMaskAndBBOX method
        Input: 
              ImagePath: str, path to an image
        Return:
            boolean, true if json file exist"""
        JsonPath, _ = ImagePath.rsplit('.',1)
        JsonPath += '.json'
        return os.path.isfile(JsonPath)
    
    def _GetOrigImageBBoxAndMask(self,ImagePath):
        """Returns the bounding boxes, masks and labels for an image determined
        from a json file"""
        JsonPath, _ = ImagePath.rsplit('.',1)
        JsonPath += '.json'
        Mask, Label = self.ReturnMaskAndLabel(JsonPath)
        return self.CreateBBoxes(Mask), Mask, Label

    
    def _GetOrgImageWithMaskAndBBoxHandle(self,Image,ImagePath,ax):
        """Returns an image handle of with mask and bounding boxes"""
        BBox, Mask, label = self._GetOrigImageBBoxAndMask(ImagePath)
        return visualize.display_instances(Image, BBox, Mask,label,
                                        ['BG'] + self.config.ValidLabels,
                                        ReturnImageHandle=True, ax=ax)
    ############### Post processing ##########################
    def FindDuplicateDetection(self,boxDetect,ThersholdIoU=0.6):
        """Determens the intersection over union for the Bounding boxes and
        the mask:
            Input:
                boxDetect: list with length number of detection. Each list elemetns 
                               contains the coordinates of a bbox bounding boxes 
                               of a detection
                boxGT: list with length number of ground truth boxes, same format 
                                as boxDetectground truth boxes
                MaskDetect: array of size [height, width, num detections] masks the a detection
                MaskGT: array of size [height, width, num ground truth] ground truth masks
            Returns:
                  IoUBBox: array of length len(boxDetect)  """
        NumObjects = len(boxDetect)
        index      = np.ones(NumObjects,dtype=bool)
        for ii in range(NumObjects):
            D_y1, D_x1, D_y2, D_x2 = boxDetect[ii]
            BBoxArea = (D_x2 - D_x1 + 1) * (D_y2 - D_y1 + 1)
            for jj in range(ii+1,NumObjects):
                GT_y1, GT_x1, GT_y2, GT_x2 = boxDetect[jj]
                xA = max(D_x1, GT_x1)
                yA = max(D_y1, GT_y1)
                xB = min(D_x2, GT_x2)
                yB = min(D_y2, GT_y2)
                intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                if intersection > 0:
                    GTArea = (GT_x2 - GT_x1 + 1) * (GT_y2 - GT_y1 + 1)
                    if (intersection/float(BBoxArea + GTArea - intersection)) > ThersholdIoU:
                        index[jj] = False
        return index

    def _RemoveDuplicateDetections(self,Results):
        return self.RemoveDuplicateDetections(Results,ThersholdIoU=self.DoubleDetectedThersholdIoU)
    
    def RemoveDuplicateDetections(self,Results,ThersholdIoU=0.6):
        if len(Results['class_ids']) == 0:
            return Results, 0
        index = self.FindDuplicateDetection(Results['rois'],ThersholdIoU=ThersholdIoU)
        if False in index:
            Results['rois']      = Results['rois'][index]
            Results['masks']     = Results['masks'][:,:,index]
            Results['class_ids'] = Results['class_ids'][index]
            Results['scores']    = Results['scores'][index]
        return Results, np.sum(np.logical_not(index))





if __name__ == "__main__":     
    from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig   
    from MaskRCNN.Configurations.ConfigSave import ConfigSave
    config = cellConfig()


    SaveConfig = ConfigSave()
    SaveConfig.dispCentreCoor = True
    SaveConfig.Captions       = False
    config.CentreExist = False
    Detector = Detect(config,SaveConfig)   

#    Detector.DetectFromImagePathAndSave(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\Octoscope2020-4-23 Archon library 400FOVs 4gridtrial_1\Training\Round1_Coords6_R0C8250_PMT_0Zmax.png')
    # Detector.addImagePaths(r'M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/ML images/NewAnnotationDaanPart2',usesubdirectories=True)
    Detector.addImagePaths(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaan\PMT\Critine\Tag\Training',usesubdirectories=True)
    Detector.runDetectionAllImages()
    # Detector.DetectAndShow(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\Octoscope2020-4-23 Archon library 400FOVs 4gridtrial_1\Training\Round1_Coords6_R0C8250_PMT_0Zmax.png')
    print('Detection time each image:')
    print(Detector.DetectionTimeEachImage)
    Detector.endDetection()





















