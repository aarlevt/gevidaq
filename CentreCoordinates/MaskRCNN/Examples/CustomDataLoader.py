# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:30:18 2020

@author: meppenga
"""

import os
import json
import numpy as np
import skimage.io
from matplotlib.path import Path as CreatePolygon
from MaskRCNN.DataGenerators.DataSetHolder import Dataset
import MaskRCNN.Miscellaneous.utils as ut

class dataVerification():
    def __init__(self,config):
        """Class to verify if data sets are complete and use the correct format
        Author: Martijn Eppenga"""
        self.ValidLabels = config.ValidLabels
        self.config      = config
    
    
    def getImageFile(self,file):
        maindir ,_ = os.path.split(file)
        with open(file) as datafile:
            data = json.load(datafile)
        ImagePath = os.path.join(maindir,data['imagePath'])
        if os.path.isfile(ImagePath):
            return ImagePath
        ImagePath, _  = ImagePath.rsplit('.')
        ImagePath += '.png'
        if os.path.isfile(ImagePath):
            return ImagePath
        raise Exception('Could not find image file')
        
    def ExistImage(self,file):
        """This function verifies if the image specified within a given json 
        file exist
        Input:
            file:       str, File to vefiy, must end with .json
            PrintError: boolean (optional, default False), when true prints error message when image file does not exist
            LogError, boolean (optional, default False), when true logs error message when image file does not exist
        Return:
            boolean, True when image file exist"""
        maindir ,_ = os.path.split(file)
        with open(file) as datafile:
            data = json.load(datafile)
        ImagePath1 = os.path.join(maindir, data['imagePath'])
        ImagePath2, _  = ImagePath1.rsplit('.')
        return os.path.isfile(ImagePath1) or os.path.isfile(ImagePath2+'.png')


    def CheckLabels(self,file):
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
        polygons = data['shapes']           
        numpolygons  = len(polygons)
        AllDataValid = True
        ValidIndex   =  [True for tel in range(numpolygons)] 
        for ii in range(numpolygons):
            # Check each label
            label = polygons[ii]['label']
            if not label in self.ValidLabels:
                AllDataValid   = False
                ValidIndex[ii] = False
        return AllDataValid, ValidIndex


                    
    def CheckJsonFile(self,file):
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
        ImageExist = self.ExistImage(file)
        LabelValid, _ = self.CheckLabels(file)
        return ImageExist and LabelValid


 







class CellDataSetLoader(Dataset, dataVerification):
    
    def __init__(self,config, CentreExist=False, UseRotation=False, UseMirror=False, ValTrain = ''):
        """Class to store information on images
        Author: Martijn Eppenga"""
        Dataset.__init__(self, config, UseRotation=UseRotation, UseMirror=UseMirror, ValTrain = ValTrain)
        self.ValidLabels = config.ValidLabels
        self.configfile  = config
        self.numImage    = 0
        self.TotalImagesAdded = 0
        self._SetupDataLoader()
        self.image_idInUse = -1 
        self.numRotImages = -1
        self.CentreExist = CentreExist
        dataVerification.__init__(self,config)

    
    def load_image(self,image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        return ut.ConvertImage2RGB(image)

    
    def addCellImage(self,directories, ValTrain, UseSubFolders=False,Exclude=[None]):
        """This function will add all json files within a directory to the data
        holder used for training of the network.
        This function will only add files from a directory Training or validation
         The funtion verifies if a given jason file contains information about 
         an image and if it can find the specified image within the json file
         and if all the labels are valid according to the config file, otherwise 
         it will not add the image file
         
         input:
             directories: list of str, list with the directories containing json 
                         files
             ValTrain:      str, Use "Validation" for validation data set,
                         "Training" for training data set
             UseSubFolders: boolean, When true the progam will seach all directories
                             within a directory for json files"""     
        if not ValTrain in ['Validation','Training']:
            # logging.warning('Not loading images from Training or validation folder but from '+ValTrain+' . Proceed with fingers crossed')
            raise Exception('Not loading images from Training or validation folder but from '+ValTrain+' . Proceed with fingers crossed')
        
        if not isinstance(directories, list):
            directories = [directories]
        allImageFiles = []
        allImageInfo  = []
        for directory in directories:
            directory = fixPathName(directory)
            if directory[-1] == '\\':
                directory = directory[:-1]
            image,info = self._getAllImages(directory, ValTrain,subdirectory= UseSubFolders,Exclude=Exclude)
            allImageFiles += image
            allImageInfo  += info
        NumLoaded = 0 

        for imagefile, infofile in zip(allImageFiles, allImageInfo):
            self._addCellImage(imagefile,infofile)
            NumLoaded += 1

        print('Number of image added: '+str(len(self.image_info)))

    
    
        
    
    
    
    def _addCellImage(self,ImagePath,imageJsonPath):
        """This function adds an image, the json file with the mask and label
        and the rotation angle with respect to the mask to the data holder
        The rotation agle tells how much degree an image is rotate counter
        clockwise with respect to the ground truth masks defined in a json file """
        self.image_idInUse    += 1
        self.TotalImagesAdded += 1
        self.add_image( 'Cell',
                       image_id  = self.image_idInUse,
                       path      = ImagePath,
                       imageInfo = imageJsonPath)
    
    
    def addSingleImage(self,file):
        """This function adds a single image to the data holder
        Input: 
            file: str, string to the json file containg the information of an 
            image (file can be made with labelme software. This will ensure 
            correct format of json file. See the CheckJsonFile method of the
            dataVerification class for format json file"""
        file = fixPathName(file)
        if not file.endswith('.json'):
            JsonPath, _ = file.rsplit('.',1)
            JsonPath   += '.json'
            if not os.path.isfile(JsonPath):
                # logging.error('Input must be a json file or json file with same image name must be saved in the directory. File given is '+file)
                raise Exception('Input must be a json file or json file with same image name must be saved in the directory. File given is '+file)
            else:
                file = JsonPath
        isImage = self._isimagefile(file)
        if isImage[0]:
            self._addCellImage(isImage[1],file)
        else:
            # logging.warning('Cannot find image file '+file)
            raise Exception('Cannot find image file '+file)
                
            

        
    def _SetupDataLoader(self):
        """
        This function addes all labels of instances possible within an image
        to the data loader class
        """
        if not isinstance(self.ValidLabels, list):
            raise Exception('Config file variable "ValidLabels" must be a list of labels')
        ii = 0
        if self.configfile.NUM_CLASSES != (len(self.ValidLabels)+1):
            msg = 'Number of classes is not equal to number of labels number of classes: '+ \
            str(self.configfile.NUM_CLASSES)+'Number of labels (+ background): '+ \
            str(len(self.ValidLabels)+1+'\nChange the NUM_CLASSES or ValidLabels '+\
                'variable in the config file')
            # logging.error(msg)
            raise Exception(msg)
        for Label in self.ValidLabels:
            ii += 1
            self.add_class("Cell", ii, Label)

            
    def _getAllImages(self,directory,ValTrain,subdirectory=True,Exclude=[None],FirstCall=True):
        """Returns two list. One with the path to an image, and a list with the path 
        to the json file with information about an image for a given input directory
        The function cheaks if the found images files are json files and if it can
        return the path to the image from the json file
        Input:
            directory: str, directroy in which to search for image files
            ValTrain:  str, Use "Validation" for validation data set,
                         "Training" for training data set
            subdirectory: boolean, Seaches sub directories for image files when
                            true
            FirstCall: boolean, used for recurssion, do not change its default
        Returns
            allImageFiles: list of str, list with paths to the image files
            allImageInfo: list of str, list with paths to the json files"""
        allImageFiles = []
        allImageInfo  = []
        
        if  FirstCall:  
            self.numImage = 0
        for file in os.listdir(directory):
            filePath = os.path.join(directory,file)
            if os.path.isdir(filePath) and subdirectory:
                if not file in Exclude:
                    files, info = self._getAllImages(filePath, ValTrain, subdirectory, FirstCall=False)
                    allImageFiles += files
                    allImageInfo  += info
            elif directory.endswith(ValTrain):
                # Check if the files are within the Validation or Training folder
                isImage = self._isimagefile(filePath)
                if isImage[0]:
                    self.numImage += 1
                    allImageFiles.append(isImage[1])
                    allImageInfo.append(filePath)
        if FirstCall:
            # logging.info('Number of loaded images for '+ValTrain+': '+str(self.numImage))
            print('Number of loaded images for '+ValTrain+': '+str(self.numImage))
        return allImageFiles, allImageInfo
    

    def load_mask(self, image_id):
        """This function returns the binary mask of a given image with the label
        for each mask"""
        path = self.image_info[image_id]['imageInfo']
        return self._createBinaryMask(path)
        

         
    
    def ReturnMaskAndLabel(self,file):
        """Function to create binary image masks from json file
        Function retuns a numpy array as [height, width, instances]
        Input: 
            file: str, path to the json file
        Returns:
            Mask, numpy array boolean, binary mask as [height, width, instances]
            labels: list, label for each mask"""
        if not file.endswith('.json'):
            raise Exception('Input file must be a json file')
        return self._createBinaryMask(file)
    
    #######
        ###################
    def _createBinaryMask(self,file):
        """Function to create binary image masks from json file
        Function retuns a numpy array as [height, width, instances]
        Input: 
            file: str, path to the json file
        Returns:
            Mask, numpy array boolean, binary mask as [height, width, instances]
            labels: list of int32, label for each mask
        """
        with open(file) as datafile:
            data = json.load(datafile)
        # Get image information
        imageHeight = data['imageHeight']
        imageWidth  = data['imageWidth']
        polygons    = data['shapes']
        numpolygons = len(polygons)
        # Allocated memory for mask instance
        mask = np.zeros((imageHeight, imageWidth, numpolygons), dtype='bool')
        Centre_coors = np.zeros((numpolygons,2), dtype=np.float32)
        # Create image points
        labels     = np.zeros(numpolygons,dtype='int32')
        ValidIndex = [False for tel in range(numpolygons)]
        numlabels  = len(self.ValidLabels)
        for ii in range(numpolygons):
            if self.CentreExist:
                centre_coor = polygons[ii]['points'][0]
                xmax, xmin, ymax, ymin, polymask = self._CreateMaskFromPolygon(polygons[ii]['points'][1:], imageHeight, imageWidth)
            else: 
                xmax, xmin, ymax, ymin, polymask = self._CreateMaskFromPolygon(polygons[ii]['points'], imageHeight, imageWidth)
                centre_coor = [int((xmax-xmin)/2+xmin), int((ymax-ymin)/2+ymin)]
                
            mask[ymin:ymax,xmin:xmax,ii]     = polymask
            Centre_coors[ii,0] = centre_coor[1]
            Centre_coors[ii,1] = centre_coor[0]
            for jj in range(numlabels):
                if polygons[ii]['label'] == self.ValidLabels[jj]:
                    labels[ii]     = jj + 1
                    ValidIndex[ii] = True
                    break
        if False in ValidIndex:
            # speeds up this if statement, otherwise everyime memory will be
            # realocated wich can make it an order of magnitude slower
            return mask[:,:,ValidIndex], labels[:,:,ValidIndex].astype(np.int32), Centre_coors[:,ValidIndex].astype(np.float32)
        else:
            return mask, labels.astype(np.int32), Centre_coors.astype(np.float32)

    def _CreateMaskFromPolygon(self, polygonPoints, imageHeight, imageWidth):
        """Creates a grid containg True on all points within a polygon spaned by the
        input polygonPoints, and False for all points outside the polygon for points 
        in an image.
        Futhermore it finds the corner coordinates of the box enclosing the polygon
        (+ 1 pixel for ymax and xmax, and -1 for xmin ymin)
        Input: 
            polygonPoints: list of coorinates as [[x1,y1],[x2,y2],....] of a polygon
            imageHeight: int, height of the image where the polygon will fit in
            imageWidth:, int widt of the the image where the polygon will fit in
        Return:
            xmax: int, x coordinate plus 1 pixel of the right corner of a box enclosing
                    the polygon
            xmin: int, x coordinate minus 1 pixel of the left corner of a box enclosing
                    the polygon
            ymax: int, y coordinate plus 1 pixel of the lower corner of a box enclosing
                    the polygon
            ymin: int, y coordinate minus 1 pixel of the upper corner of a box enclosing
                    the polygon
            grid: array size of ((ymax-ymin),(xmax-xmin)) dtype boolean, grid containing 
                    True on points inside a polygon and False for points outside polygon
                    image coordinates coresponding to the grid:  image[ymin:ymax,xmin:xmax]
        """
        xmax = polygonPoints[0][0]
        xmin = polygonPoints[0][0]
        ymax = polygonPoints[0][1]
        ymin = polygonPoints[0][1]
        
        # find the coorners of the box enclosing the polygon
        for ii in range(1,len(polygonPoints)):
            if polygonPoints[ii][0] > xmax:
                xmax = polygonPoints[ii][0]
            elif polygonPoints[ii][0] < xmin:
                xmin = polygonPoints[ii][0]
            if polygonPoints[ii][1] > ymax:
                ymax = polygonPoints[ii][1]
            elif polygonPoints[ii][1] < ymin:
                ymin = polygonPoints[ii][1]
        
        # get the coordinates and add or subtract one pixel to padded it        
        xmax = int(min(np.ceil(xmax+1),imageWidth))
        xmin = int(max(np.floor(xmin-1),0))
        ymax = int(min(np.ceil(ymax+1),imageHeight))
        ymin = int(max(np.floor(ymin-1),0))
        
        # Create the grid only for the part that contains the polygons
        # Otherwise we are calculating many times a zero which makes the algorithm slow
        x, y   = np.meshgrid(np.arange(xmin,xmax), np.arange(ymin,ymax))
        x, y   = x.flatten(), y.flatten()
        p = CreatePolygon(polygonPoints)
        grid = p.contains_points(np.vstack((x,y)).T)
        return  xmax, xmin, ymax, ymin, grid.reshape(ymax-ymin, xmax-xmin).astype(np.bool)

    
    ################
    
    
    
    def CreateBBoxes(self,mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        if mask.dtype != bool:
            raise Exception('Data type of mask is '+str(mask.dtype)+' but must be bool')
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for ii in range(mask.shape[-1]):
            m = mask[:, :, ii]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies   = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[ii] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)
    
    def image_reference(self, image_id):
        """This function returns the path to the json file of annotated image
        This json file contains all the information of the image"""
        return self.image_info[image_id]['imageInfo']
    
        
    def _isimagefile(self,file):
        """Check if input file is a json file and if it can find the specified
        image in the json file
        Input:
            file: str, path to a json file
        Returns:
            isimage: list, 
                first entry boolean which is true if the json file
                    exist, contains the right information and if it contains a path to 
                    to an image
            second entry (only if first entry is true, str with path to the image file"""
        isimage = [False]
        if file.endswith('.json'):
            if self.CheckJsonFile(file):
                isimage[0] = True
                isimage.append(self.getImageFile(file))
        return isimage
    
   
    
def fixPathName(Path,endwith=''):
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
    
    if endwith != '' and not Path.endswith(endwith):
        Path += endwith
    return (Path.replace('/','\\')).replace('\\\\','\\')   


def _ConvertImage2RGB(Image):   
    """Converst an image to RGB 
    Output data is uint8 if input is not a uint8 then values are rescaled 
    to [0-255] and then cast to a uint8"""
    if len(np.shape(Image)) == 2:
        Image = skimage.color.gray2rgb(Image).astype(Image.dtype)
        
    elif len(np.shape(Image)) == 4:
        Image = Image[:,:,:]
    if Image.dtype == np.uint8:
        return Image
    else:
        return _correctImageScale(Image)

def _correctImageScale(Image):
    minval = np.min(Image)
    maxval = np.max(Image)
    return ((Image-minval)/(maxval-minval)*255).astype(np.uint8)