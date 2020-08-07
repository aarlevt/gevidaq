# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:28:21 2020

@author: meppenga
"""

import numpy as np
import skimage.io
from MaskRCNN.Miscellaneous.ImageTransform import ImageTransform
from MaskRCNN.Miscellaneous.Misc import updateProgressBar

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    To rotate all images 90, 180, 270 deg and add them to the data set set the
    option UseRotation to True in the init method
    To mirror all images ud, lr, udlr and add them to the data set set the
    option UseMirror to True in the init method
    When one sets one of these option make sure that you also specify ValTrain,
    which is the folder where the rotated and or mirroed images will be stored
    within the Cahe folder.
    This folder will be cleared from all files when one uses the rotation and or
    mirror option
    """

    def __init__(self,config, class_map=None, UseRotation=False, UseMirror=False, ValTrain = ''):
        """Input:
            config: congigurations file
            UseRotation, boolean, when True will rotate each image 90, 180 and 270
                deg and add the results to the dataholder. You need to specify 
                ValTrain input when using this option. The ValTrain directory will
                be cleared
            UseMirror, boolean, same as UseRotation but it will mirror the images
                ud lr and udlr
            ValTrain: str, directory to store the roated and or mirrored images
                within the cache folder. This folder will be removed when it already
                exist and all items stored within are thus lost
                Example of input: 'myFolder\\Validation'
            """
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.UseRotation = UseRotation
        self.UseMirror   = UseMirror
        self.ValTrain    = ValTrain
        self.config      = config

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        self.num2transform  = len(self.image_info)
        if self.UseRotation or self.UseMirror:
            if self.ValTrain == '':
                raise Exception('Cannot use roatation and/or mirroring as no "ValTrain" input is given (see __init__)')
            self.ImageTransform = ImageTransform(self.config, self.ValTrain)           
            if self.UseRotation: 
                self.ImageTransform.ClearCacheRotation()
                self.CreateRotatedImagesAndAdd()
            if self.UseMirror:
                self.ImageTransform.ClearCacheMirror()
                self.CreatMirrorImagesAndAdd()
                
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            centre_coor: Centre coordinates of a cell in format [y,x, instance count]
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        # logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask        = np.empty([0, 0, 0])
        class_ids   = np.empty([0], np.int32)
        centre_coor = np.empty([0,0])
        return mask, class_ids, centre_coor


    def CreateRotatedImagesAndAdd(self):
        count = 0
        if self.UseRotation:
            for ii in range(self.num2transform):
                Image = self.load_image(ii)
                mask, class_ids, centre_coor = self.load_mask(ii)
                for angle in [90, 180, 270]:
                    JsonSavePath = self.ImageTransform.CreateAndSaveRotatedImage(Image.copy(), mask.copy(), class_ids.copy(), centre_coor.copy(), angle)
                    image_info = {
                                    "id": len(self.image_info)+1,
                                    "source": self.image_info[ii]['source'],
                                    "path": JsonSavePath,
                                }
                    self.image_info.append(image_info)
                    count += 1
                    updateProgressBar((count)/(self.num2transform*3), message='Creating rotated images. Progress')

                    
                
        
        
    def CreatMirrorImagesAndAdd(self):
        count = 0
        if self.UseMirror:
            for ii in range(self.num2transform):
                Image = self.load_image(ii)
                mask, class_ids, centre_coor = self.load_mask(ii)
                for MirrorType in ['ud','lr','udlr']:
                    JsonSavePath = self.ImageTransform.CreateAndSaveMirroredImage(Image.copy(), mask.copy(), class_ids.copy(), centre_coor.copy(), MirrorType)
                    image_info = {
                                    "id": len(self.image_info)+1,
                                    "source": self.image_info[ii]['source'],
                                    "path": JsonSavePath,
                                }
                    self.image_info.append(image_info)
                    count +=1
                    updateProgressBar((count)/(self.num2transform*3), message='Creating mirrored images. Progress')

    def load_Rot_Mir_Data(self,image_id):
        Path = self.image_info[image_id]['path']
        return self.ImageTransform.load_Rot_Mir_data(Path)
# To do Add progress bar to addMirror and addRotation, Add funtion to load rotated data to data generator (use self.num2transform as index counter), test all




