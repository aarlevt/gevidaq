# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:35:22 2020

@author: xinmeng
"""

import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.constants import AcquisitionType, TaskMode
from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalMultiChannelWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader
from PyQt5.QtCore import pyqtSignal, QThread

import os
# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname+'/../')

import numpy as np
import NIDAQ.wavegenerator
from NIDAQ.wavegenerator import blockWave
from NIDAQ.constants import MeasurementConstants

class RasterScan():
    
    def __init__(self, Daq_sample_rate, edge_volt, pixel_number = 500, average_number = 1, continuous = False, return_image = True):
        """
        

        Parameters
        ----------
        Daq_sample_rate : int
            Sampling rate used to generate the waveforms.
        edge_volt : float
            The bounding raster scan voltage.
        pixel_number : int, optional
            Number of pixels in the final image. The default is 500.
        average_number : int, optional
            Number of frames to average on. The default is 1.
        continuous : TYPE, optional
            Whether to do continuous scanning or not. The default is False.
        return_image : TYPE, optional
            Whether return the processed image. The default is True.

        Returns
        -------
        None.

        """
        
        #---------------------Generate the waveforms---------------------------
        self.Daq_sample_rate = Daq_sample_rate
        self.averagenum = average_number
        self.edge_volt = edge_volt
        self.pixel_number = pixel_number
        self.flag_continuous = continuous
        self.flag_return_image = return_image
    
        # Generate galvo samples            
        self.samples_X, self.samples_Y= NIDAQ.wavegenerator.waveRecPic(sampleRate = self.Daq_sample_rate, imAngle = 0, voltXMin = -1*self.edge_volt, voltXMax = self.edge_volt, 
                         voltYMin = -1*self.edge_volt, voltYMax = self.edge_volt, xPixels = self.pixel_number, yPixels = self.pixel_number, sawtooth = True)
        # Calculate number of all samples to feed to daq.
        self.Totalscansamples = len(self.samples_X)*self.averagenum 
        # Number of samples of each individual line of x scanning, including fly backs.
        # Devided by pixel number as it's repeated for each y line.
        self.total_X_sample_number = int (len(self.samples_X)/self.pixel_number) 
        
        self.repeated_samples_X = np.tile(self.samples_X, self.averagenum)
        self.repeated_samples_Y = np.tile(self.samples_Y, self.averagenum)
        
        self.Galvo_samples = np.vstack((self.repeated_samples_X,self.repeated_samples_Y))
        
    def run(self):
        """
        Starts writing a waveform continuously while reading 
        the buffer periodically
        """

        with nidaqmx.Task() as slave_Task, nidaqmx.Task() as master_Task:

            slave_Task.ao_channels.add_ao_voltage_chan("/Dev1/ao0:1")
            master_Task.ai_channels.add_ai_voltage_chan("/Dev1/ai0")
            
            if self.flag_continuous == False:
                # Timing of analog output channels
                slave_Task.timing.cfg_samp_clk_timing(rate = self.Daq_sample_rate, source='ai/SampleClock',
                                                       sample_mode = AcquisitionType.FINITE, samps_per_chan=self.Totalscansamples)
                                                     
                # Timing of recording channels
                master_Task.timing.cfg_samp_clk_timing(rate = self.Daq_sample_rate,
                                                       sample_mode= AcquisitionType.FINITE, samps_per_chan=self.Totalscansamples)
            else:
                # Timing of analog output channels
                slave_Task.timing.cfg_samp_clk_timing(rate = self.Daq_sample_rate, source='ai/SampleClock',
                                                       sample_mode = AcquisitionType.CONTINUOUS)
                                                     
                # Timing of recording channels
                master_Task.timing.cfg_samp_clk_timing(rate = self.Daq_sample_rate,
                                                       sample_mode= AcquisitionType.CONTINUOUS, samps_per_chan=self.Totalscansamples)                
            
            reader = AnalogSingleChannelReader(master_Task.in_stream)
            writer = AnalogMultiChannelWriter(slave_Task.out_stream)
            
            reader.auto_start = False
            writer.auto_start = False
            
            writer.write_many_sample(self.Galvo_samples)
            
            """Reading data from the buffer in a loop. 
            The idea is to let the task read more than could be loaded in the buffer for each iteration.
            This way the task will have to wait slightly longer for incoming samples. And leaves the buffer
            entirely clean. This way we always know the correct numpy size and are always left with an empty
            buffer (and the buffer will not slowly fill up)."""
            output = np.zeros(self.Totalscansamples)
            slave_Task.start() #Will wait for the readtask to start so it can use its clock
            master_Task.start()
            
            while not self.isInterruptionRequested():
                reader.read_many_sample(data = output, 
                                        number_of_samples_per_channel = self.Totalscansamples)

                Dataholder_average = np.mean(output.reshape(self.averagenum, -1), axis=0)
                
                if self.flag_return_image == True:
                    # Calculate the mean of average frames.
                    self.data_PMT = np.reshape(Dataholder_average, (self.pixel_number, self.total_X_sample_number))
                    
                    # Cut off the flying back part.
                    if self.pixel_number == 500:
                        self.image_PMT= self.data_PMT[:, 50:550]*-1
                    elif self.pixel_number == 256:
                        self.image_PMT= self.data_PMT[:, 70:326]*-1
                    
                    return self.image_PMT
                
if __name__ == '__main__':
    galvo = RasterScan(Daq_sample_rate = 500000, edge_volt = 5)
    image = galvo.run()
        