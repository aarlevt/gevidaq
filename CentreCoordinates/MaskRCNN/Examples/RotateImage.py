# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:26:03 2020

@author: meppenga
"""
from SearchPathsExamples import addSearchPaths
addSearchPaths()
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from matplotlib.path import Path as CreatePolygon
import json
import os
import visualize
from utils import ConvertImage2RGB
from ConfigFileInferenceOld import cellConfig
ValidLabels = ['DeadCell','RoundCell','FlatCell']


def _createBinaryMask(file,angle):
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
    Centre_coors = np.zeros((2, numpolygons), dtype=np.float32)
    # Create image points
    labels     = np.zeros(numpolygons,dtype='int32')
    ValidIndex = [False for tel in range(numpolygons)]
    numlabels  = len(ValidLabels)
    for ii in range(numpolygons):
        a = polygons[ii]['points']
        polygons[ii]['points'] =   Rotate(polygons[ii]['points'],angle,[imageHeight,imageWidth])
        centre_coor = polygons[ii]['points'][0]
        xmax, xmin, ymax, ymin, polymask = _CreateMaskFromPolygon(polygons[ii]['points'], imageHeight, imageWidth)
        mask[ymin:ymax,xmin:xmax,ii]     = polymask
        Centre_coors[0,ii] = centre_coor[1]
        Centre_coors[1,ii] = centre_coor[0]
        for jj in range(numlabels):
            if polygons[ii]['label'] == ValidLabels[jj]:
                labels[ii]     = jj + 1
                ValidIndex[ii] = True
                break
    if False in ValidIndex:
        # speeds up this if statement, otherwise everyime memory will be
        # realocated wich can make it an order of magnitude slower
        return mask[:,:,ValidIndex], labels[:,:,ValidIndex].astype(np.int32), Centre_coors[:,ValidIndex].astype(np.float32)
    else:
        return mask, labels.astype(np.int32), Centre_coors.astype(np.float32)

def _CreateMaskFromPolygon(polygonPoints, imageHeight, imageWidth):
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

def CreateBBoxes(mask):
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

def Rotate(coordinates,theta,imageshape):
    theta *= (np.pi/180)
    h = imageshape[0]
    w = imageshape[1]
    rotated = []
    for coor in coordinates:
        xrot = ((coor[0]-w/2)*np.cos(theta) - (coor[1]-h/2)*np.sin(theta)) + (w/2)
        yrot = ((coor[0]-w/2)*np.sin(theta) + (coor[1]-h/2)*np.cos(theta)) + (h/2)
        rotated.append([xrot,yrot])
    return(rotated)


jsonfile  = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\2020-4-15 Archon library 144FOVs 4grid__trial_2\Validation\Round1_Coords3_R0C3300_PMT_0Zmax.json'
imagefile = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\2020-4-15 Archon library 144FOVs 4grid__trial_2\Validation\Round1_Coords3_R0C3300_PMT_0Zmax.png'


angle = 0
Mask,label,centre = _createBinaryMask(jsonfile,angle)
coor = []
for ii in range(len(label)):
    coor.append([centre[0,ii],centre[1,ii]])
    


plt.close('Test')
fig,(ax1,ax2,ax3) = plt.subplots(1,3,num='Test')

Image = ConvertImage2RGB(plt.imread(imagefile))
Image = skimage.transform.rotate(Image, -angle)
# print(np.shape(CreateBBoxes(Mask)))
# print(CreateBBoxes(Mask))
visualize.display_instances(Image, CreateBBoxes(Mask), Mask, label,
                            ['BG'] + ValidLabels, ax=ax1, Centre_coor_radius = 2 ,centre_coors =coor)

filejson  = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Cache\Rotation\000_Rot090_Round1_Coords3_R0C3300_PMT_0Zmax.json'
imagefile = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Cache\Rotation\000_Rot090_Round1_Coords3_R0C3300_PMT_0Zmax.png'
Mask,label,centre = _createBinaryMask(filejson,0)
Image = ConvertImage2RGB(plt.imread(imagefile))
coor = []
for ii in range(len(label)):
    coor.append([centre[0,ii],centre[1,ii]])
visualize.display_instances(Image, CreateBBoxes(Mask), Mask, label,
                            ['BG'] + ValidLabels, ax=ax2, Centre_coor_radius = 2 ,centre_coors =coor)



filejson  = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Cache\Mirror\000_udlr_Round1_Coords3_R0C3300_PMT_0Zmax.json'
imagefile = r'C:\MaskRCNN\MaskRCNNGit\CentreCoordinates\FinalVersion\Data\Cache\Mirror\000_udlr_Round1_Coords3_R0C3300_PMT_0Zmax.png'
Mask,label,centre = _createBinaryMask(filejson,0)
Image = ConvertImage2RGB(plt.imread(imagefile))
coor = []
for ii in range(len(label)):
    coor.append([centre[0,ii],centre[1,ii]])
visualize.display_instances(Image, CreateBBoxes(Mask), Mask, label,
                            ['BG'] + ValidLabels, ax=ax3, Centre_coor_radius = 2 ,centre_coors =coor)




# fig.canvas.draw()
# fig.canvas.flush_events() 
# fig.show()
# nul = np.zeros(np.shape(Image),dtype=np.uint8)
# for ii in range(41):
#     for idx in range(3):
#         nul[:,:,idx] += (Image[:,:,idx]*Mask[:,:,ii]).astype(np.uint8)
    

# ax2.imshow(nul)
# ax2.axis('off')

raise
config = cellConfig()



class ImageTransform(object):
    
    def __init__(self,config):
        self.config = config
        self.num    = 0
        
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
        return skimage.transform.rotate(Image, -theta)
    
    def CreateRotatedImage(self,Imagepath,jsonPath):
        pass
                
    
        
























