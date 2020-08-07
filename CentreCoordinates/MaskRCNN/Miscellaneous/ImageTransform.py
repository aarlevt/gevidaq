# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:49:56 2020

@author: meppenga
"""
from json import JSONEncoder
import json
import os
import numpy as np
import skimage



class ImageTransform(object):
    
    def __init__(self,config, Folder):
        self.config = config
        self.numsaveRot    = -1
        self.numsaveMir    = -1
        self.SaveDir, _ = os.path.split(self.config.ROTATION_CACHE)
        self.RotationFolder = 'Rotation'
        self.MirrorFolder   = 'Mirror'
        self.Folder = Folder

        if not os.path.isdir(os.path.join(self.config.ROTATION_CACHE,Folder)):
            os.makedirs(os.path.join(self.config.ROTATION_CACHE,Folder))
        if not os.path.isdir(os.path.join(self.config.MIRROR_CACHE,Folder)):
            os.makedirs(os.path.join(self.config.MIRROR_CACHE,Folder))
        
    def ClearCacheRotation(self):
        if 'Cache' in self.config.ROTATION_CACHE and 'Rotation' in self.config.ROTATION_CACHE:
            for searchpath in ['']:
                searhPathAll = os.path.join(self.config.ROTATION_CACHE, self.Folder)
                for file in os.listdir(searhPathAll):
                    if file[:3].isnumeric() and file[3:7] == '_Rot':
                        if file.endswith('.json') or file.endswith('.png'):
                            os.remove(os.path.join(searhPathAll,file))
        else:
            raise Exception('Will not clear any other folder than the Rotation Cache folder\nFolder given is: '+self.config.ROTATION_CACHE)
    
    def ClearCacheMirror(self):
        if 'Cache' in self.config.MIRROR_CACHE and 'Mirror' in self.config.MIRROR_CACHE:
            for searchpath in ['']:
                searhPathAll = os.path.join(self.config.MIRROR_CACHE, self.Folder)
                for file in os.listdir(searhPathAll):
                    if file[:3].isnumeric() and file[3:6] in ['_lr','_ud','_udlr','_lrud']:
                        if file.endswith('.json') or file.endswith('.png'):
                            os.remove(os.path.join(searhPathAll,file))
        else:
            raise Exception('Will not clear any other folder than the Rotation Cache folder\nFolder given is: '+self.config.MIRROR_CACHE)
        
    def RotateCoor(self, coordinates, theta, imageshape):
        dtype       = coordinates.dtype
        coordinates = coordinates.astype(np.float64)
        rotated = np.zeros(np.shape(coordinates),dtype=np.float64)
        
        # Cache for fast computation
        theta *= (np.pi/180)
        h = (imageshape[0]*-1)/2
        w = imageshape[1]/2
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Transform from image space to coordinate system (zero at image centre)
        coordinates[:,0] = coordinates[:,0] - w
        coordinates[:,1] = coordinates[:,1]*-1 - h
        
        # Rotated and transorfm back
        rotated[:,0] = coordinates[:,0] * cos_theta - coordinates[:,1] * sin_theta
        rotated[:,1] = coordinates[:,0] * sin_theta + coordinates[:,1] * cos_theta
        
        # Transfrom form coordinate system to image space
        rotated[:,0] = rotated[:,0] + w
        rotated[:,1] = (rotated[:,1] + h) * -1
        return rotated.astype(dtype)
    
    def RotateImage(self,Image,theta):
        return skimage.transform.rotate(Image, theta, preserve_range=True).astype(Image.dtype)
        
    
    def CreateSavePaths(self,ImageName,TransformType, Folderdir):
        if TransformType[:3] == 'Rot':
            self.numsaveRot += 1
            num = '%03d'%(int(self.numsaveRot))
        elif TransformType[:2] in ['lr','ud','udlr','lrud']:
            self.numsaveMir += 1
            num = '%03d'%(int(self.numsaveMir))
        else:
            raise Exception('Unkown transfrom type')                    
        jsonFileName  = num+'_'+TransformType+'_'+ImageName+'.json'
        SaveDir = os.path.join(os.path.join(self.SaveDir,Folderdir),self.Folder )
        if not os.path.isdir(SaveDir):
            os.mkdir(SaveDir)
        return os.path.join(os.path.join(Folderdir, self.Folder), jsonFileName), os.path.join(SaveDir, jsonFileName)

    
    
    def CreateAndSaveRotatedImage(self, image, Mask, Label, CentreCoor, angle):
        self.VerigyDataFormat(image, Mask, Label, CentreCoor)
        ImageName, JsonSavePath = self.CreateSavePaths('AutoRot','Rot%03d' % (int(angle)), self.RotationFolder)
        image = self.RotateImage(image, angle)
        rotateMask = np.zeros(np.shape(Mask),dtype=bool)
        imageshape = np.shape(image)
        imageshape = [imageshape[0],imageshape[1]]
        for ii in range(len(Label)):
            rotateMask[:,:,ii] = self.RotateImage(Mask[:,:,ii],angle).astype(bool)

        CentreCoor[:,:] = self.RotateCoor(CentreCoor[:,::-1],angle,imageshape)[:,::-1]
        Imsave,_ = JsonSavePath.rsplit('.')
        skimage.io.imsave(Imsave+'.png',image)
        MaskCoor  =  []
        SmallMask =  []
        for ii in range(len(Label)):
            MaskCoor.append(self.findMaskCoordinates(rotateMask[:,:,ii]))
            y1, x1, y2, x2 = MaskCoor[ii]
            SmallMask.append(rotateMask[y1:y2,x1:x2,ii])
        data = {'Mask': SmallMask,
                'MaskCoor': MaskCoor,
                'Label': Label,
                'CentreCoor': CentreCoor}
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile,cls=NumpyArrayEncoder)
        return ImageName

    def CreateAndSaveMirroredImage(self, image, Mask, Label, CentreCoor, MirrorType):
        self.VerigyDataFormat(image, Mask, Label, CentreCoor)
        ImageName, JsonSavePath = self.CreateSavePaths('AutoMir', MirrorType, self.MirrorFolder)
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

        Imsave,_ = JsonSavePath.rsplit('.')
        skimage.io.imsave(Imsave+'.png',image)
        MaskCoor  =  []
        SmallMask =  []
        for ii in range(len(Label)):
            MaskCoor.append(self.findMaskCoordinates(Mask[:,:,ii]))
            y1, x1, y2, x2 = MaskCoor[ii]
            SmallMask.append(Mask[y1:y2,x1:x2,ii])
        data = {'Mask': SmallMask,
                'MaskCoor': MaskCoor,
                'Label': Label,
                'CentreCoor': CentreCoor}
        with open(JsonSavePath, 'w') as outfile:
            json.dump(data, outfile,cls=NumpyArrayEncoder)
        return ImageName
    
    def findMaskCoordinates(self, Mask):
        # Bounding box.
        horizontal_indicies = np.where(np.any(Mask, axis=0))[0]
        vertical_indicies   = np.where(np.any(Mask, axis=1))[0]
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
        return np.array([y1, x1, y2, x2], dtype=np.int32)


    def load_Rot_Mir_data(self,ImageName):
        ImPath,_ = ImageName.rsplit('.')
        with open(os.path.join(self.SaveDir, ImageName)) as datafile:
            data = json.load(datafile)
        Image = skimage.io.imread(os.path.join(self.SaveDir, ImPath+'.png'))
        label = np.asarray(data['Label'])
        mask  = np.zeros((np.shape(Image)[0:2])+(len(label),),dtype=bool)
        SmallMask = data['Mask']
        
        MaskCoor = data['MaskCoor']
        for ii in range(len(np.asarray(data['Label']))):
            y1, x1, y2, x2 = MaskCoor[ii]
            mask[y1:y2,x1:x2,ii] = np.asarray(SmallMask[ii])
        return Image, mask, label, np.asarray(data['CentreCoor'])
  

    def VerigyDataFormat(self,image, Mask, Label, CentreCoor):
        shapeImage = np.shape(image)
        ShapeMask = np.shape(Mask)
        assert (shapeImage[0] == shapeImage[0] and shapeImage[1] == shapeImage[1]), \
            'Mask and Image should have the same shape. Shape mask: '+str(shapeImage[0:2])+' Shape image '+str(shapeImage[0:2])
        assert ShapeMask[2] == len(Label), 'The number of mask must be equal to the number of labels'
        assert len(Label) == np.shape(CentreCoor)[0], 'The number of centre coordinates must be equal to the number of labels'
        assert Mask.dtype == bool, 'Dtype mask must be boolean'
                                           
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)   