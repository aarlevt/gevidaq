#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:38:26 2020

@author: Izak de Heer
"""

import importlib.resources
import sys

import numpy as np

from .backend import ALP4


class DMDActuator:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_DMD()

        # self.repeat specifies whether continuously projecting or not.
        self.repeat = True

    def initialize_DMD(self):
        # Load the Vialux .dll
        traversable = importlib.resources.files(sys.modules[__package__])
        with importlib.resources.as_file(traversable) as path:
            self.DMD = ALP4.ALP4(
                version="4.3", libDir=str(path)
            )  # Use version 4.3 for the alp4395.dll

        # Initialize the device
        self.DMD.Initialize(13388)  # TODO magic number
        print("DMD Initialized")

    def disconnect_DMD(self):
        # Clear onboard memory and disconnect
        self.DMD.Free()
        print("DMD disconnected")

    def send_data_to_DMD(self, img_seq):
        """
        Load image or image sequence to onboard memory of DMD.
        In case of binary illumination, bit depth of image should be 1.

        param mask: 2d binary numpy array or stack of 2d arrays
        type mask: Illumination mask
        """

        if len(img_seq.shape) == 2:
            self.seq_length = 1
            self.image = img_seq.ravel()

        # In case it's one image with newaxis(empty axis on 3rd dimension)
        elif img_seq.shape[2] == 1:
            self.seq_length = 1
            self.image = img_seq[:, :, 0].ravel()
        elif img_seq.shape[2] > 1:
            self.seq_length = img_seq.shape[2]

            self.image = np.concatenate(
                [img_seq[:, :, 0].ravel(), img_seq[:, :, 1].ravel()]
            )
            for i in range(2, self.seq_length):
                self.image = np.hstack([self.image, img_seq[:, :, i].ravel()])

        self.image = (
            self.image > 0
        ) * 1  # First part makes it True/False, multiplying by 1 converts it to binary

        # Binary amplitude image (0 or 1)
        bitDepth = 1
        self.image *= 2 ** 8 - 1
        self.image = self.image.astype(int)
        # Allocate the onboard memory for the image sequence
        # nbImg defines the number of masks

        self.DMD.SeqAlloc(nbImg=self.seq_length, bitDepth=bitDepth)

        # Send the image sequence as a 1D list/array/numpy array
        self.DMD.SeqPut(imgData=self.image)
        print("Data loaded to DMD")

    def start_projection(self):
        """
        In case of a image sequence, the loop parameter determines whether the
        sequence is projected once of repeatedly.
        """
        self.inquire_status()

        self.DMD.Run(loop=self.repeat)

        print("Projection started")

    def inquire_status(self):
        PICTURE_TIME = self.DMD.SeqInquire(inquireType=ALP4.ALP_PICTURE_TIME)
        ILLUMINATE_TIME = self.DMD.SeqInquire(inquireType=ALP4.ALP_ILLUMINATE_TIME)
        BITNUM = self.DMD.SeqInquire(inquireType=ALP4.ALP_BITNUM)
        BIN_MODE = self.DMD.SeqInquire(inquireType=ALP4.ALP_BIN_MODE)
        OFF_TIME = self.DMD.SeqInquire(inquireType=ALP4.ALP_OFF_TIME)
        PICNUM = self.DMD.SeqInquire(inquireType=ALP4.ALP_PICNUM)
        MIN_PICTURE_TIME = self.DMD.SeqInquire(inquireType=ALP4.ALP_MIN_PICTURE_TIME)

        print("-------------DMD status-------------")
        print("ALP_PICTURE_TIME: {} μs".format(PICTURE_TIME))
        print("ALP_ILLUMINATE_TIME: {} μs".format(ILLUMINATE_TIME))
        print(f"ALP_BITNUM: {BITNUM}.")
        if BIN_MODE == 2015:  # TODO magic number
            print("ALP_BIN_MODE: with dark phase.")
        elif BIN_MODE == 2016:  # TODO magic number
            print("ALP_BIN_MODE: Operation without dark phase.")
        print("ALP_OFF_TIME: {} μs (total inactive projection time)".format(OFF_TIME))
        print("Number of pictures in sequence: {}".format(PICNUM))
        print("minimum duration of the display of one picture in μs(MIN_PICTURE_TIME): {}".format(MIN_PICTURE_TIME))
        print("------------------------------------")

    def stop_projection(self):
        self.DMD.Halt()
        print("Projection stopped")

    def set_timing(self, frame_rate):
        """
        Set the duration of one frame in a image sequence. NB: time is in microseconds.
        """
        self.DMD.SetTiming(illuminationTime=frame_rate)
        # print('Timing set')

    def set_repeat(self, repeat):
        self.repeat = repeat

    def free_memory(self):
        self.DMD.Halt()
        self.DMD.FreeSeq()

        self.repeat = True
