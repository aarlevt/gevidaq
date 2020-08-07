# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:49:56 2020

@author: meppenga
"""

import json
import os
import numpy as np
import skimage
from MaskRCNN.Miscellaneous.utils import ConvertImage2RGB
from MaskRCNN.Miscellaneous.Misc import updateProgressBar


class ImageTransform(object):
    
    def __init__(self,config):
        self.config = config
        self.numsaveRot    = -1
        self.numsaveMir    = -1
        
    def ClearCacheRotation(self, folder):
        if 'Cache' in self.config.ROTATION_CACHE and 'Rotation' in self.config.ROTATION_CACHE:
            for searchpath in [folder]:
                searhPathAll = os.path.join(self.config.ROTATION_CACHE, searchpath)
                for file in os.listdir(searhPathAll):
                    if file[:3].isnumeric() and file[3:7] == '_Rot':
                        if file.endswith('.json') or file.endswith('.png'):
                            os.remove(os.path.join(searhPathAll,file))
        else:
            raise Exception('Will not clear any other folder than the Rotation Cache folder\nFolder given is: '+self.config.ROTATION_CACHE)
    
    def ClearCacheMirror(self, folder):
        if 'Cache' in self.config.MIRROR_CACHE and 'Mirror' in self.config.MIRROR_CACHE:
            for searchpath in [folder]:
                searhPathAll = os.path.join(self.config.MIRROR_CACHE, searchpath)
                for file in os.listdir(searhPathAll):
                    if file[:3].isnumeric() and file[3:6] in ['_lr','_ud','_udlr','_lrud']:
                        if file.endswith('.json') or file.endswith('.png'):
                            os.remove(os.path.join(searhPathAll,file))
        else:
            raise Exception('Will not clear any other folder than the Rotation Cache folder\nFolder given is: '+self.config.MIRROR_CACHE)
        
    def RotateCoor(self, coordinates, theta, imageshape):
        theta *= (np.pi/180)
        h = imageshape[0]
        w = imageshape[1]
        rotated = []
        for coor in coordinates:
            xrot = ((coor[0]-w/2)*np.cos(theta) - (coor[1]-h/2)*np.sin(theta)) + (w/2)
            yrot = ((coor[0]-w/2)*np.sin(theta) + (coor[1]-h/2)*np.cos(theta)) + (h/2)
            rotated.append([xrot,yrot])
        return(rotated) 
    
    def RotateImage(self,Image,theta):
        return skimage.transform.rotate(Image, -theta, preserve_range=True).astype(np.uint8)
    
    def CreateSavePaths(self,ImagePath,SaveDir,TransformType, Folderdir):
        if TransformType[:3] == 'Rot':
            self.numsaveRot += 1
            num = '%03d'%(int(self.numsaveRot))
        elif TransformType[:2] in ['lr','ud','udlr','lrud']:
            self.numsaveMir += 1
            num = '%03d'%(int(self.numsaveMir))
        else:
            raise Exception('Unkown transfrom type')
            
        _, ImageName = os.path.split(ImagePath)
        ImageName, _ = ImageName.rsplit('.')
        jsonFileName  = num+'_'+TransformType+'_'+ImageName+'.json'
        imageFileName = num+'_'+TransformType+'_'+ImageName+'.png'
        SaveDir = os.path.join(SaveDir, Folderdir)
        if not os.path.isdir(SaveDir):
            os.mkdir(SaveDir)
        return os.path.join(SaveDir,imageFileName), os.path.join(SaveDir,jsonFileName)

    
    
    def CreateAndSaveRotatedImage(self, image, Mask, Label, CentreCoor, angle, Folderdir):
        ImageSavePath, JsonSavePath = self.CreateSavePaths('AutoRot', self.config.ROTATION_CACHE,'Rot%03d' % (int(angle)), Folderdir)
        image = self.RotateImage(image, angle)
        rotateMask = np.zeros(np.shape(Mask),dtype=bool)
        imageshape = np.shape(image)
        imageshape = [imageshape[0],imageshape[1]]
        for ii in range(np.shape(Mask)[-1]):
            rotateMask[:,:,ii] = self.RotateImage(Mask[:,:,ii],angle).astype(bool)
        for ii in range(np.shape(CentreCoor)[0]):  
            CentreCoor[ii,:] = self.RotateCoor(CentreCoor[ii,::-1],angle,imageshape)[::-1]
        data = {'image': image,
                'Mask': rotateMask,
                'Label': Label,
                'CentreCoor': CentreCoor}
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile)
        return JsonSavePath

    def CreateAndSaveMirroredImage(self, image, Mask, Label, CentreCoor, MirrorType, Folderdir):
        ImageSavePath, JsonSavePath = self.CreateSavePaths('AutoMir', self.config.MIRROR_CACHE, MirrorType, Folderdir)
        imageHeight,imageWidth = np.shape(image)[0:2] 
        if not MirrorType in ['lr','ud','udlr','lrud']:
            raise Exception('None supported mirror type\nSupported are: '+'lr ud udlr lrud')
        if MirrorType == 'lr' or MirrorType == 'udlr' or MirrorType == 'lrud':
            for ii in range(len(Label)):
                diff = imageWidth/2 - CentreCoor[ii,1]
                CentreCoor[ii,1] = CentreCoor[ii,1]+2*diff  
                Mask[:,:,ii] = Mask[:,::-1,ii]
            for ii in range(3):
                image[:,:,ii] = image[:,::-1,ii]
        if MirrorType == 'ud' or MirrorType == 'udlr' or MirrorType == 'lrud':
            for ii in range(len(Label)):

                diff = imageHeight/2 - CentreCoor[ii,0]
                CentreCoor[ii,0] = CentreCoor[ii,0]+2*diff 
                Mask[:,:,ii] = Mask[::-1,:,ii]
            for ii in range(3):
                image[:,:,ii] = image[::-1,:,ii]
        data = {'image': image,
                'Mask': Mask,
                'Label': Label,
                'CentreCoor': CentreCoor}
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile)
        return JsonSavePath
    
    
    
    def CreateRotatedImage(self,Imagepath,jsonPath, angle, Folderdir):
        ImageSavePath, JsonSavePath = self.CreateSavePaths(Imagepath, self.config.ROTATION_CACHE,'Rot%03d' % (int(angle)), Folderdir)
        with open(jsonPath) as datafile:
            data = json.load(datafile)
        # Get image information
        imageHeight = data['imageHeight']
        imageWidth  = data['imageWidth']
        polygons    = data['shapes']
        numpolygons = len(polygons)
        
        for ii in range(numpolygons):
            polygons[ii]['points'] =   self.RotateCoor(polygons[ii]['points'], angle, [imageHeight,imageWidth])
        data['shapes'] = polygons
        _, data['imagePath'] = os.path.split(ImageSavePath)
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile)
        image = ConvertImage2RGB(skimage.io.imread(Imagepath))
        skimage.io.imsave(ImageSavePath, self.RotateImage(image, angle))
        
    def CreateMirroredImage(self,Imagepath,jsonPath, MirrorType, Folderdir):
        if not MirrorType in ['lr','ud','udlr','lrud']:
            raise Exception('None supported mirror type\nSupported are: '+'lr ud udlr lrud')
        ImageSavePath, JsonSavePath = self.CreateSavePaths(Imagepath, self.config.MIRROR_CACHE, MirrorType, Folderdir)
        with open(jsonPath) as datafile:
            data = json.load(datafile)
        imageHeight = data['imageHeight']
        imageWidth  = data['imageWidth']
        polygons    = data['shapes']
        numpolygons = len(polygons)
        image = ConvertImage2RGB(skimage.io.imread(Imagepath))
        if MirrorType == 'lr' or MirrorType == 'udlr' or MirrorType == 'lrud':
            for ii in range(numpolygons):
                lrCoor = []
                for points in polygons[ii]['points']:
                    diff = imageWidth/2 - points[0]
                    lrCoor.append([points[0]+2*diff, points[1]])
                polygons[ii]['points'] = lrCoor
            data['shapes']  = polygons
            for ii in range(3):
                image[:,:,ii] = image[:,::-1,ii]
        if MirrorType == 'ud' or MirrorType == 'udlr' or MirrorType == 'lrud':
            for ii in range(numpolygons):
                udCoor = []
                for points in polygons[ii]['points']:
                    diff = imageHeight/2 - points[1]
                    udCoor.append([points[0], points[1]+2*diff])
                polygons[ii]['points'] = udCoor
            data['shapes'] = polygons
            for ii in range(3):
                image[:,:,ii] = image[::-1,:,ii]
        _, data['imagePath'] = os.path.split(ImageSavePath)
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile)
        skimage.io.imsave(ImageSavePath, image)
        
    def findFiles(self,searchdir):
        files = []
        for file in os.listdir(searchdir):
            filepath = os.path.join(searchdir,file)
            if os.path.isdir(filepath):
                files += self.findFiles(filepath)
            elif filepath.endswith('.json'):
                folder, _ = os.path.split(filepath)
                if folder.endswith('Training') or folder.endswith('Validation'):
                    filename, _ = filepath.rsplit('.')
                    filename += '.png'
                    if os.path.isfile(filename):
                        if folder.endswith('Training'):
                            files.append([filename, filepath,'Training'])
                        else:
                            files.append([filename, filepath,'Validation'])
        return files
                    
    def CreateMirror(self, searchdir,types=['ud','lr','udlr']):
        self.ClearCacheMirror()
        AllFiles = self.findFiles(searchdir)
        numfiles = max(len(AllFiles)*len(types),1)
        count = 0
        for files in AllFiles:
            for type_ in types:
                self.CreateMirroredImage(files[0], files[1], type_, files[2])
                count += 1
                updateProgressBar(count/numfiles, message = 'Mirroring %03d images, Progress' % (numfiles))
        
    
    def CreateRotations(self, searchdir, angles=[90,180,270]):
        self.ClearCacheRotation()
        AllFiles = self.findFiles(searchdir)
        numfiles = max(len(AllFiles)*len(angles),1)
        count = 0
        for files in AllFiles:
            for angle in angles:
                self.CreateRotatedImage(files[0], files[1], angle, files[2])
                count += 1
                updateProgressBar(count/numfiles, message = 'Rotating %03d images, Progress' % (numfiles))
                    
                
                
            
        
if __name__ == "__main__":     
    from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig
    config = cellConfig()
    jsonfile  = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\2020-4-15 Archon library 144FOVs 4grid__trial_2\Validation\Round1_Coords3_R0C3300_PMT_0Zmax.json'
    imagefile = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\2020-4-15 Archon library 144FOVs 4grid__trial_2\Validation\Round1_Coords3_R0C3300_PMT_0Zmax.png'
   
    Transformer = ImageTransform(config)
    
    if False:
        Transformer.CreateMirror(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2',['ud','lr','udlr'])
        Transformer.CreateRotations(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2',[90,180,270])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    