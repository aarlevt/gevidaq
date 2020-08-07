# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:01:19 2020

@author: meppenga
"""
import sys
import os
import logging
import skimage
import numpy as np

def setuplogger(logdir=None):    
    # setup logger  
    if not logdir:
        path, _= os.path.split( os.path.abspath(__file__))
        path, _= os.path.split(path)
        path = os.path.join(path, 'Data')
        if not os.path.isdir(path):
            os.mkdir(path)
        logdir = path
    logging.basicConfig(filename=os.path.join(logdir,'LogFile.log'), level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    with open(os.path.join(logdir,'LogFile.log'),'a') as fileobj:
        fileobj.write('\n')        
    logging.info('Started')

def printColor(color,msg):
    """Function to print text using a specific style
    Author: Martijn Eppenga
    Supported colors: Black, Red Green, Yellow, Blue, Purple, Cyan, White
    Underline use underline or UColor Replace Color with the requested color
    Bold: use bold or BColor Replace Color with the requested color
    Input:
        color: str, requested color example 'Red'
        msg:   str, text to print"""
    colordict ={'None': '',
    'error':       "\033[0;31m",
    'Error':       "\033[0;31m",
    'warning':     "\033[0;33m",
    'Warning':     "\033[0;33m",
    'underline':   "\033[4;30m",
    'Underline':   "\033[4;30m",
    'bold':        "\033[0;30m",
    'Bold':        "\033[0;30m",
    'Black':       "\033[0;30m",      # Black
    'Red':         "\033[0;31m",      # Red
    'Green':       "\033[0;32m",      # Green
    'Yellow':      "\033[0;33m",      # Yellow
    'Blue':        "\033[0;34m",      # Blue
    'Purple':      "\033[0;35m",      # Purple
    'Cyan':        "\033[0;36m",      # Cyan
    'White':       "\033[0;37m",      # White
    'UBlack':      "\033[4;30m",      # Underline Black
    'URed':        "\033[4;31m",      # Underline Red
    'UGreen':      "\033[4;32m",      # Underline Green
    'UYellow':     "\033[4;33m",      # Underline Yellow
    'UBlue':       "\033[4;34m",      # Underline Blue
    'UPurple':     "\033[4;35m",      # Underline Purple
    'UCyan':       "\033[4;36m",      # Underline Cyan
    'UWhite':      "\033[4;37m",      # Underline White
    'BBlack':      "\033[1;30m",      # Bold Black
    'BRed':        "\033[1;31m",      # Bold Red
    'BGreen':      "\033[1;32m",      # Bold Green
    'BYellow':     "\033[1;33m",      # Bold Yellow
    'BBlue':       "\033[1;34m",      # Bold Blue
    'BPurple':     "\033[1;35m",      # Bold Purple
    'BCyan':       "\033[1;36m",      # Bold Cyan
    'BWhite':      "\033[1;37m",      # Bold White
    }
    print(colordict[color]+msg+'\033[0m')

def updateProgressBar(progress, barLength = 30, message = 'Progress',decimals=1,padding=5):
    """Creates progress bar
    Input:
        process: float (0-1) status of the process
        barLength: int (optional default 30), length of the progress bar
        message: str (optional default Progress), Message to print at front of 
                     progress bar
        decimals: int (optional default =1), number of decimal places for percentage
        padding: int (default=5), number of white spaces at the end, can be used
                   to ensure that the message looks clean"""
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    barLength = int(barLength)
    block     = int(barLength*progress)
    text      = "\r{0}: [{1}] {2}% {3} {4}".format(message, "#"*block +
                   "-"*(barLength-block), round(progress*100,decimals), status, ' '*padding)
    sys.stdout.write(text)
    sys.stdout.flush()



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

def GetAllImagePaths(directory,FileType='png',UseSubDir = True):
    """Function returns the path to each png file int the directory "directory"
    and it sub directories
    Input:
        directory: str, directory in which to search for png files"""
    files = []
    for file in os.listdir(directory):
        if os.path.isdir(directory+'\\'+file) and UseSubDir:
            files += GetAllImagePaths(directory+'\\'+file, FileType, UseSubDir)
        elif os.path.isfile(directory+'\\'+file):
            if file.endswith('.'+FileType):
                files.append(directory+'\\'+file)
    return files

def convert2png(ImagePath,Tempdir,ImageName='',Overwrite=False):
    """This function converts an image to a png image. the default image format 
    used for the mask rcnn detector
    Input: 
        ImagePath: str, path to image to be converted
        Tempdir: str, directory to save the converted image
        ImageName: str (optional), if given this will be the name given to the 
                  new image. This name must NOT include a path NOR the .png specifier
        Overwrite: boolean (optional default = True), when true overwrites 
                    converted image file if it already exist
        Return:
            SavePath: str, path to converted image"""
    if not os.path.isfile(ImagePath):
        raise Exception('File '+ImagePath+' does not exist')
    if ImageName == '':
        _, ImageName = fixPathName(ImagePath).rsplit('\\',1)
        ImageName, _ = ImageName.rsplit('.',1)
    SavePath = os.path.join(Tempdir,ImageName+'.png')
    if (not Overwrite) and os.path.isfile(SavePath):
        raise Exception('File '+SavePath+' does already exist\n'+
                        'To overwrite set the option overwrite to True')
    if not os.path.isdir(Tempdir):
        os.mkdir(Tempdir)
        print('Çreated directory '+Tempdir+' to store converted file')
    Image = skimage.io.imread(ImagePath)
    Image = skimage.color.gray2rgb(Image).astype(Image.dtype)
    Image = skimage.img_as_ubyte(skimage.exposure.rescale_intensity(Image,out_range=(0, 255)))
    skimage.io.imsave(SavePath,Image)
    return SavePath




##### Image format converter     ######
def _resizeImage(self,Image):
    """Resize an image such that it format is less then or equal to the image
    size specified in the config file"""
    shape = np.shape(Image)
    if shape[0] > self.config.IMAGE_MAX_DIM or shape[1] > self.config.IMAGE_MAX_DIM:
        logging.info('Rescaled image')
        return skimage.transform.resize(Image,[self.config.IMAGE_MAX_DIM,self.config.IMAGE_MAX_DIM],preserve_range=True).astype(Image.dtype)
    else:
        return Image
    
    
    
def _ConvertImage2RGB(self,Image):   
    """Converst an image to RGB 
    Output data is uint8 if input is not a uint8 then values are rescaled 
    to [0-255] and then cast to a uint8"""
    if len(np.shape(Image)) == 2:
        Image = skimage.color.gray2rgb(Image).astype(Image.dtype)
        if self.config.ImageType != 'GrayScale':
            printColor('warning','Image to detect is gray scale, converting to color')
            logging.warning('Image to detect is gray scale, converting to color')
    elif len(np.shape(Image)) == 4:
        printColor('warning','Image to detect is RGBA, converting to color')
        Image = Image[:,:,:]
    if Image.dtype == np.uint8:
        return Image
    else:
        return self._correctImageScale(Image)
    
def _correctImageScale(self,Image):
    minval = np.min(Image)
    maxval = np.max(Image)
    return ((Image-minval)/(maxval-minval)*255).astype(np.uint8)