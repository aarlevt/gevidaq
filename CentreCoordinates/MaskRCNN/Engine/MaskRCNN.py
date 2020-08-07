# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:04:23 2020

@author: meppenga
"""

import os
import re
import datetime
from MaskRCNN.Engine.RegionProposal import RegionProposal
from MaskRCNN.Engine.MaskRCNNLayers import ProposalLayer, fpn_classifier_graph_centre_coor, DetectionLayer, build_fpn_mask_graph, DetectionTargetLayer
import tensorflow.keras.layers as KL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from MaskRCNN.Engine.LossFunctions import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph,  mrcnn_bbox_loss_graph,  mrcnn_mask_loss_graph, mrcnn_Centre_coor_loss_graph
import tensorflow.keras.optimizers as KO
import  MaskRCNN.Miscellaneous.utils as ut
from MaskRCNN.DataGenerators.DataGenerator import DataLoader
import logging
from MaskRCNN.Miscellaneous.Misc import setuplogger
setuplogger()




class MaskRCNN(object):
    
    def __init__(self,config, mode, model_dir):
        """Implementation of the MaskRCNN algorithm using tensorflow 2.2.0 as
        backend. This code is based on the MaskRCNN implemenation of matterport
        but upgraded to tensorflow 2.2.0 and extended to predict centre coordinates
        of objects"""
        self.config = config
        self.model_dir = model_dir
        self.mode   = mode
        logging.info('Creating model. Mode: '+mode)
        inputs, outputs = self.build_Model()
        logging.info('Creating custom trainer')
        self.model = CustomModel(inputs,outputs)
        self.model.addConfigfile(self.config)
        self.set_log_dir()
        self.TF_Version = tf.__version__
        

        
    
    def build_Model(self):
        config = self.config
        assert self.mode in ['training', 'inference']
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
        # Define Inputs
        anchors          = KL.Input(shape=[None, 4], name="input_anchors")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        
        if self.mode == 'training':
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            active_class_ids = KL.Lambda(
                lambda x: ut.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)
            
            
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            input_gt_centre_coor = KL.Input(
                shape=[None, 2], name="input_gt_centre_coor", dtype=tf.float32)

            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        
        
        # Create Region proposal network
        RPNBuilder = RegionProposal(self.config) 
        input_image, RPN_outputs, featureMaps = RPNBuilder.BuildRegionProposalModel()
        rpn_class_logits, rpn_class, rpn_bbox = RPN_outputs
        rpn_feature_maps, mrcnn_feature_maps  = featureMaps
        
        # Generate proposals
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.mode == "training"\
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=self.config.RPN_NMS_THRESHOLD,
                name="ROI",
                config=self.config)([rpn_class, rpn_bbox, anchors])
        
        if self.mode == 'training':
            
            rois, target_class_ids, target_bbox, target_mask, target_centre_coor =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    rpn_rois, input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_centre_coor])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_centre_coor =\
                fpn_classifier_graph_centre_coor(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)
#            CentreCoordinates = 
            # Calculate losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])
            Center_coor_loss = KL.Lambda(lambda x: mrcnn_Centre_coor_loss_graph(*x), name="mrcnn_centre_coor_loss")(
                [target_centre_coor, target_class_ids, mrcnn_centre_coor])
            
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            
            inputs  = [input_image, input_image_meta, anchors, input_gt_class_ids, 
                       input_gt_boxes, input_gt_masks, input_rpn_match, input_rpn_bbox, input_gt_centre_coor]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, Center_coor_loss]                    
                    
        else:
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_centre_coor =\
                    fpn_classifier_graph_centre_coor(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                         self.config.POOL_SIZE, self.config.NUM_CLASSES,
                                         train_bn=self.config.TRAIN_BN,
                                         fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
            
            
            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(self.config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta, mrcnn_centre_coor])
        
            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              self.config.MASK_POOL_SIZE,
                                              self.config.NUM_CLASSES,
                                              train_bn=self.config.TRAIN_BN)

    
        
            inputs  = [input_image, input_image_meta, anchors]
            outputs = [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, mrcnn_centre_coor]
        return inputs, outputs
 

        
        
        
    def compileModel(self): 
        logging.info('Compile model')
        optimizer = KO.SGD(lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM,
                    clipnorm=self.config.GRADIENT_CLIP_NORM)
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(lambda: tf.add_n(reg_losses))
        self.model.compile(optimizer=optimizer)
    
    
    
    def LoadWeigths(self,weight_file, by_name=False):
        logging.info('Load weigths: '+weight_file)
        self.model.load_weights(weight_file, by_name=by_name)
    
    
    
    def Train(self,Train_Data, Val_Data, epochs, UserCallBack=None):
        logging.info('Init training')
        if not os.path.exists(self.log_dir):
            logging.info('Create log dir: '+self.log_dir)
            os.makedirs(self.log_dir)
        logging.info('Create Callbacks')    
        epochloggerCallback = LossLogger(self.log_dir+'\\TrainingLoss.log', self.config, len(Train_Data.image_ids), len(Val_Data.image_ids))
        callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
            epochloggerCallback]
        if UserCallBack:
            callbacks += UserCallBack
        assert self.mode == 'training'
        logging.info('Create data generators')
        TrainGenerator = DataLoader(self.config, Train_Data)
        ValGenerator   = DataLoader(self.config, Val_Data)
        logging.info('Start Training')
        self.model.fit(TrainGenerator, verbose=1, validation_data=ValGenerator,
                       epochs=epochs, callbacks=callbacks,
                       steps_per_epoch  = min(self.config.STEPS_PER_EPOCH, len(TrainGenerator.image_ids)),
                       validation_steps = min(self.config.VALIDATION_STEPS, len(ValGenerator.image_ids)))
        logging.info('End training')
      
        
      
        
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = ut.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = ut.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = ut.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
      


    def detect(self,images,verbose=0):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        # logging.info('Init detection')
        config  = self.config
        assert self.mode == "inference", "Create model in inference mode."
        # assert len(images) == config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Convert images to RGB
        # Replace this line if you want to use different kind of images
        # Input of the nerual network are float32 type of images
        images = ut.ConvertImages2RGB(images)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = ut.mold_inputs(self.config, images)


        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        # logging.info('Run detection on %03d images' %(len(images)))
        if len(images) <= config.BATCH_SIZE:
            # This method should be faster if all input images fit within one batch
            detections, _, _, mrcnn_mask, _, _, _ , center_coor =\
                self.model([molded_images, image_metas, anchors], training=False)
        else:
            detections, _, _, mrcnn_mask, _, _, _ , center_coor =\
                self.model.predict([molded_images, image_metas, anchors], verbose=0) #[molded_images, image_metas, anchors]
        # Process detections
        results = []
        for i, image in enumerate(images):
            if len(images) <= config.BATCH_SIZE:
                final_rois, final_class_ids, final_scores, final_masks, Centre_coor =\
                    ut.unmold_detections(detections[i].numpy(), mrcnn_mask[i].numpy(),
                                           image.shape, molded_images[i].shape,
                                           windows[i],ReturnMiniMask=config.RETURN_MINI_MASK)
            else:
                final_rois, final_class_ids, final_scores, final_masks, Centre_coor =\
                    ut.unmold_detections(detections[i], mrcnn_mask[i],
                                           image.shape, molded_images[i].shape,
                                           windows[i],ReturnMiniMask=config.RETURN_MINI_MASK)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                'Centre_coor': Centre_coor,
                'maskshape': image.shape[0:2],
            })
        return results
    
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        
        
        
loss_tracker = keras.metrics.Mean(name="loss")    
class CustomModel(keras.Model):
    
    
    def addConfigfile(self,config):
        self.config = config
    
    def train_step(self, data):
        # Unpack data
        Inputs, Outputs = data

        with tf.GradientTape() as tape:
            # y_pred =  # Forward pass
            _, _, _, _, _, _, _, _, _, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, centre_loss = self(Inputs, training=True) 
            loss = [tf.reduce_mean(rpn_class_loss, keepdims=True), tf.reduce_mean(rpn_bbox_loss, keepdims=True), 
                    tf.reduce_mean(class_loss, keepdims=True), tf.reduce_mean(bbox_loss, keepdims=True), tf.reduce_mean(mask_loss, keepdims=True), 
                    tf.reduce_mean(centre_loss, keepdims=True)]


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        # mae_metric.update_state(Inputs, y_pred)
        return {"loss": loss_tracker.result()}
    

    def test_step(self, data):
        # Unpack the data
        Inputs, Outputs = data
        # Compute predictions
        _, _, _, _, _, _, _, _, _, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, centre_loss = self(Inputs, training=True)
        loss = [tf.reduce_mean(rpn_class_loss, keepdims=True), tf.reduce_mean(rpn_bbox_loss, keepdims=True), 
                    tf.reduce_mean(class_loss, keepdims=True), tf.reduce_mean(bbox_loss, keepdims=True), tf.reduce_mean(mask_loss, keepdims=True),
                     tf.reduce_mean(centre_loss, keepdims=True)]
        # Updates the metrics tracking the loss
        loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": loss_tracker.result()}




    



class LossLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, config, NumTrain, NumVal):
        self.log_dir = log_dir
        self._initLogFile(config, NumTrain, NumVal)
        self._saveConfig(config)
        
    def on_epoch_end(self, epoch, logs=None):
        message = "Epoch {} Loss: {:5.4f} - Val_loss: {:5.4f}\n".format(
                epoch,logs["loss"], logs["val_loss"])
        with open(self.log_dir,'a') as fileobj:
            fileobj.write(message)
    

    def _initLogFile(self, config, NumTrain, NumVal):
        if os.path.isfile(self.log_dir):
            with open(self.log_dir,'a') as fileobj:
                fileobj.write('\n\nContinue training\n\n')
        with open(self.log_dir,'a') as fileobj:
            fileobj.write('Training results\n')
            fileobj.write('\nNumber of training images used per epoch:   %03d' % min([config.STEPS_PER_EPOCH, NumTrain])) 
            fileobj.write('\nNumber of Validation images used per epoch: %03d' % min([config.VALIDATION_STEPS, NumVal])) 
            fileobj.write('\nTotal number of training images used:       %03d' % (NumTrain) )
            fileobj.write('\nTotal number of validation images used :    %03d' % (NumVal)) 
            fileobj.write('\nResults:\n\n')

    def _saveConfig(self,config):
        """Create a file called Config.txt in the SavePath directory
        This file contains all the setting of the configuration file"""
        SavePath, _ = os.path.split(self.log_dir)
        with open(os.path.join(SavePath,'Config.txt'),'a') as fileobj:
            for setting in dir(config):
                if not setting.startswith("__") and not callable(getattr(config, setting)):
                    fileobj.write("{:30} {}".format(setting, getattr(config, setting))+'\n') 






        
        














