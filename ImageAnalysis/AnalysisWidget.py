# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:27:04 2020

@author: xinmeng

-------------------------------------------------------------------------------------------------------------------------------------
                                Image analysis GUI
-------------------------------------------------------------------------------------------------------------------------------------
"""

from __future__ import division
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QRect, QObject
from PyQt5.QtGui import QColor, QPen, QPixmap, QIcon, QTextCursor, QFont

from PyQt5.QtWidgets import (QWidget, QButtonGroup, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton, QGroupBox, 
                             QLineEdit, QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QTabWidget, QCheckBox, QRadioButton, 
                             QFileDialog, QProgressBar, QTextEdit)
import os
from datetime import datetime
import pyqtgraph as pg
from pyqtgraph import PlotDataItem, TextItem
from IPython import get_ipython
import sys
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d
from skimage.io import imread
import threading
import time
# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname+'/../')
from ImageAnalysis.ImageProcessing import ProcessImage, CurveFit
import ImageAnalysis.Plotanalysis


class AnalysisWidgetUI(QWidget):
    
#    waveforms_generated = pyqtSignal(object, object, list, int)
#    SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    MessageBack = pyqtSignal(str)
    Cellselection_DMD_mask_contour =  pyqtSignal(list)
    #------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        os.chdir('./')# Set directory to current folder.
        self.setFont(QFont("Arial"))
        
        self.setMinimumSize(1250,850)
        self.setWindowTitle("AnalysisWidget")
        self.layout = QGridLayout(self)
        self.savedirectory = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Patch clamp"
        self.OC = 0.1 # Patch clamp constant
        #**************************************************************************************************************************************
        #--------------------------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------GUI for Data analysis tab--------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------          
        #**************************************************************************************************************************************
        readimageContainer = QGroupBox("Readin images")
        self.readimageLayout = QGridLayout()
        
        self.Construct_name = QLineEdit(self)
        self.Construct_name.setPlaceholderText("Enter construct name")
        self.Construct_name.setFixedWidth(150)
        self.readimageLayout.addWidget(self.Construct_name, 1, 0)
        
        self.switch_Vp_or_camtrace = QComboBox()
        self.switch_Vp_or_camtrace.addItems(['Correlate to Vp', 'Correlate to video'])
        self.readimageLayout.addWidget(self.switch_Vp_or_camtrace, 1, 1)
        
        # self.readimageLayout.addWidget(QLabel('Video of interest:'), 1, 0)
       
        self.textbox_directory_name = QLineEdit(self)        
        self.readimageLayout.addWidget(self.textbox_directory_name, 1, 4)
        
        self.button_browse = QPushButton('Set data folder', self)
        self.readimageLayout.addWidget(self.button_browse, 1, 5) 
        
        self.button_browse.clicked.connect(self.getfile)

        self.button_load = QPushButton('Load', self)
        self.button_load.setStyleSheet("QPushButton {color:white;background-color: blue; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
                                       "QPushButton:pressed {color:black;background-color: blue; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
                                       "QPushButton:hover:!pressed {color:gray;background-color: blue; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}")
        self.readimageLayout.addWidget(self.button_load, 1, 6)
        self.button_load.clicked.connect(self.start_analysis)
        
        self.Spincamsamplingrate = QSpinBox(self)
        self.Spincamsamplingrate.setMaximum(20000)
        self.Spincamsamplingrate.setValue(1000)
        self.Spincamsamplingrate.setSingleStep(500)
        self.readimageLayout.addWidget(self.Spincamsamplingrate, 1, 3)
        self.readimageLayout.addWidget(QLabel("Camera FPS:"), 1, 2)
        
        self.button_clearpolts = QPushButton('Clear', self)
        self.readimageLayout.addWidget(self.button_clearpolts, 1, 8)         
        
        self.button_clearpolts.clicked.connect(self.clearplots)
        

        self.button_Background_load = QPushButton('Analyse', self)
        self.button_Background_load.setStyleSheet("QPushButton {color:white;background-color: orange; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
                                       "QPushButton:pressed {color:black;background-color: blue; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
                                       "QPushButton:hover:!pressed {color:gray;background-color: orange; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}")
        self.readimageLayout.addWidget(self.button_Background_load, 1, 7) 
        
        self.button_Background_load.clicked.connect(self.finish_analysis)
        
        readimageContainer.setLayout(self.readimageLayout)
        readimageContainer.setMaximumHeight(120)
        
        #-----------------------------------------------------Image analysis display Tab-------------------------------------------------------
        Display_Container = QGroupBox("Image analysis display")
        Display_Layout = QGridLayout()
        # Setting tabs
        Display_Container_tabs = QTabWidget()

        #------------------------------------------------------Image Analysis-Average window-------------------------------------------------------
        image_display_container_layout = QGridLayout()
        
        imageanalysis_average_Container = QGroupBox("Image Analysis-Average window")
        self.imageanalysisLayout_average = QGridLayout()
                
        #self.pw_averageimage = averageimagewindow()
        self.pw_averageimage = pg.ImageView()
        self.pw_averageimage.ui.roiBtn.hide()
        self.pw_averageimage.ui.menuBtn.hide()   

        self.roi_average = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
        self.pw_averageimage.view.addItem(self.roi_average)
        #self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_average.addWidget(self.pw_averageimage, 0, 0, 5, 3)
        
        imageanalysis_average_Container.setLayout(self.imageanalysisLayout_average)
        imageanalysis_average_Container.setMinimumHeight(180)
        #------------------------------------------------------Image Analysis-weighV window-------------------------------------------------------
        imageanalysis_weight_Container = QGroupBox("Image Analysis-Weight window")
        self.imageanalysisLayout_weight = QGridLayout()
                
        #self.pw_averageimage = averageimagewindow()
        self.pw_weightimage = pg.ImageView()
        self.pw_weightimage.ui.roiBtn.hide()
        self.pw_weightimage.ui.menuBtn.hide()
        
        self.roi_weighted = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
        self.pw_weightimage.view.addItem(self.roi_weighted)
        #self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_weight.addWidget(self.pw_weightimage, 0, 0, 5, 3)
        
        imageanalysis_weight_Container.setLayout(self.imageanalysisLayout_weight)
        imageanalysis_weight_Container.setMinimumHeight(180)
        
        image_display_container_layout.addWidget(imageanalysis_average_Container, 0, 0)
        image_display_container_layout.addWidget(imageanalysis_weight_Container, 0, 1)
        
        #----------------------------------------------------------------------
        Display_Container_tabs_tab3 = ImageAnalysis.Plotanalysis.PlotAnalysisGUI()
#        Display_Container_tabs_tab3.setLayout(self.Curvedisplay_Layout)
        
        #----------------------------------------------------------------------
        # Display_Container_tabs_tab2 = QWidget()
        # Display_Container_tabs_tab2.setLayout(self.VIdisplay_Layout)        
        
        #----------------------------------------------------------------------
        Display_Container_tabs_Galvo_WidgetInstance = QWidget()
        Display_Container_tabs_Galvo_WidgetInstance.setLayout(image_display_container_layout)
        
        #----------------------------------------------------------------------
        # self.Display_Container_tabs_Cellselection = QWidget()
        # self.Display_Container_tabs_Cellselection_layout = QGridLayout()
        
        # self.show_cellselection_gui_button = QPushButton('show')
        # self.show_cellselection_gui_button.clicked.connect(self.show_cellselection_gui)
        # self.Display_Container_tabs_Cellselection_layout.addWidget(self.show_cellselection_gui_button, 0,0)
        # self.Display_Container_tabs_Cellselection.setLayout(self.Display_Container_tabs_Cellselection_layout)
        
        Display_Container_tabs_tab4 = QWidget()
        Display_Container_tabs_tab4_layout = QGridLayout()
        
        self.textbox_single_waveform_filename = QLineEdit(self)        
        Display_Container_tabs_tab4_layout.addWidget(self.textbox_single_waveform_filename, 0, 0)
        
        self.button_browse_tab4 = QPushButton('Browse', self)
        Display_Container_tabs_tab4_layout.addWidget(self.button_browse_tab4, 0, 1) 
        
        self.button_browse_tab4.clicked.connect(self.get_single_waveform)
        
        Display_Container_tabs_tab4.setLayout(Display_Container_tabs_tab4_layout)          
        
        # Add tabs
        Display_Container_tabs.addTab(Display_Container_tabs_Galvo_WidgetInstance,"Graph display")
        # Display_Container_tabs.addTab(Display_Container_tabs_tab2,"Patch display")        
        Display_Container_tabs.addTab(Display_Container_tabs_tab3,"Patch perfusion")
        # Display_Container_tabs.addTab(self.Display_Container_tabs_Cellselection,"Cell selection")
        Display_Container_tabs.addTab(Display_Container_tabs_tab4,"show trace")
        
        Display_Layout.addWidget(Display_Container_tabs, 0, 0)  
        Display_Container.setLayout(Display_Layout)        

        self.layout.addWidget(readimageContainer, 0, 0, 1, 2)
        self.layout.addWidget(Display_Container, 1, 0, 1, 2)
#        master_data_analysis.addWidget(imageanalysis_average_Container, 2, 0, 1,1)
#        master_data_analysis.addWidget(imageanalysis_weight_Container, 2, 1, 1,1)
        
        
        #**************************************************************************************************************************************
        #--------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------Functions for Data analysis Tab------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------  
        #**************************************************************************************************************************************        
    def getfile(self):
        self.main_directory = str(QtWidgets.QFileDialog.getExistingDirectory(directory='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data'))
        self.textbox_directory_name.setText(self.main_directory)
        
        for file in os.listdir(self.main_directory):
			# For Labview generated data.
            if file.endswith(".tif") or file.endswith(".TIF"):
                self.fileName = self.main_directory + '/'+file
                print(self.fileName)
        
    def start_analysis(self):
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        
        if not os.path.exists(os.path.join(self.main_directory, 'Analysis results')):
            # If the folder is not there, create the folder
            os.mkdir(os.path.join(self.main_directory, 'Analysis results')) 
        
        print('Loading data...')
        self.MessageToMainGUI('Loading data...'+'\n')        

        t1 = threading.Thread(target = self.load_data_thread)
        t1.start()
        
        get_ipython().run_line_magic('matplotlib', 'inline')
        
    def finish_analysis(self):
        t2 = threading.Thread(target = self.finish_analysis_thread)
        t2.start()        
        
    def load_data_thread(self):
        # Load tif video file.
        self.videostack = imread(self.fileName)
        print(self.videostack.shape)
        self.MessageToMainGUI('Video size: '+str(self.videostack.shape)+'\n')
        self.roi_average.maxBounds= QRectF(0,0,self.videostack.shape[2],self.videostack.shape[1])
        self.roi_weighted.maxBounds= QRectF(0,0,self.videostack.shape[2],self.videostack.shape[1])
        print('Loading complete, ready to fire')
        self.MessageToMainGUI('Loading complete, ready to fire'+'\n')
        
        # Load wave files.
        self.loadcurve()
        
        # display electrical signals
        self.display_electrical_signals()
        
        time.sleep(0.5)
        # Calculate the mean intensity of video, for background substraction.
        self.video_mean()
        
    def finish_analysis_thread(self):
        # Calculate the background
        self.calculate_background_from_ROI_average()
        
        # Substract background
        self.substract_background()
        
        # calculate_weight
        self.calculate_weight()
        
        # display_weighted_trace
        self.display_weighted_trace()
        
        # Fit on weighted trace and sumarize the statistics.
        self.fit_on_trace()
        
    def ReceiveVideo(self, videosentin):  

        self.videostack = videosentin
        print(self.videostack.shape)
        self.MessageToMainGUI('Video size: '+str(self.videostack.shape)+'\n')
        self.roi_average.maxBounds= QRectF(0,0,self.videostack.shape[2],self.videostack.shape[1])
        self.roi_weighted.maxBounds= QRectF(0,0,self.videostack.shape[2],self.videostack.shape[1])
        print('Loading complete, ready to fire')
        self.MessageToMainGUI('Loading complete, ready to fire'+'\n')
        
    def loadcurve(self):
        """
        Load the 1D array files, like voltage, current recordings or waveform information.

        Returns
        -------
        None.

        """
        for file in os.listdir(self.main_directory):
			# For Labview generated data.
            if file.endswith(".Ip"):
                self.Ipfilename = self.main_directory + '/'+file
                curvereadingobjective_i =  ProcessImage.readbinaryfile(self.Ipfilename)               
                self.Ip, self.samplingrate_curve = curvereadingobjective_i.readbinarycurve()                
                self.Ip = self.Ip[0:len(self.Ip)-2]
                
            elif file.endswith(".Vp"):
                self.Vpfilename = self.main_directory + '/'+file
                curvereadingobjective_V =  ProcessImage.readbinaryfile(self.Vpfilename)               
                self.Vp, self.samplingrate_curve = curvereadingobjective_V.readbinarycurve()                
                self.Vp = self.Vp[0:len(self.Vp)-2] # Here -2 because there are two extra recording points in the recording file.
            
			# For python generated data
            elif file.startswith('Vp'):
                self.Vpfilename_npy = self.main_directory + '/'+file
                curvereadingobjective_V =  np.load(self.Vpfilename_npy)
                print(curvereadingobjective_V[10])
                self.Vp = curvereadingobjective_V[5:len(curvereadingobjective_V)]
                self.samplingrate_curve = curvereadingobjective_V[0]
                self.Vp = self.Vp[0:-2]
                
            elif file.startswith('Ip'):
                self.Ipfilename_npy = self.main_directory + '/'+file
                curvereadingobjective_I =  np.load(self.Ipfilename_npy)
                print('I raw: {}'.format(curvereadingobjective_I[1000]))
                self.Ip = curvereadingobjective_I[5:len(curvereadingobjective_I)]
                self.Ip = self.Ip[0:-2]
                self.samplingrate_curve = curvereadingobjective_I[0]
                
            elif 'Wavefroms_sr_' in file and 'npy' in file:
                self.Waveform_filename_npy = self.main_directory + '/'+file
                # Read in configured waveforms
                configwave_wavenpfileName = self.Waveform_filename_npy
                self.waveform_display_temp_loaded_container = np.load(configwave_wavenpfileName, allow_pickle=True)
                self.samplingrate_display_curve = int(float(configwave_wavenpfileName[configwave_wavenpfileName.find('sr_')+3:-4]))
                print('Wavefroms_sampling rate: {}'.format(self.samplingrate_display_curve))
                
    def getfile_background(self):
        self.fileName_background, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', 'M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data',"Image files (*.jpg *.tif)")
        self.textbox_Background_filename.setText(self.fileName_background)
        
    def substract_background(self):
        """
        Substract the background from the original video.

        Returns
        -------
        None.

        """
        print('Loading...')
        
        self.background_rolling_average_number = int(self.samplingrate_cam/5)
        print("background_rolling_average_number is : {}".format(self.background_rolling_average_number))
        
        # if self.switch_bg_Video_or_image.currentText() == 'Video':
        #     self.videostack_background = imread(self.fileName_background)
        #     print(self.videostack_background.shape)
        #     self.videostack = self.videostack - self.videostack_background
        #     print('Substraction complete.')
            
        # elif self.switch_bg_Video_or_image.currentText() == 'ROI':
            
        unique, counts = np.unique(self.averageimage_ROI_mask,return_counts=True)
        count_dict = dict(zip(unique, counts))
        print('number of 1 and 0:'+str(count_dict))
        
        self.background_trace = []
        
        for i in range(self.videostack.shape[0]):
            ROI_bg = self.videostack[i]\
            [self.roi_avg_coord_raw_start:self.roi_avg_coord_raw_start+self.averageimage_ROI_mask.shape[0], \
             self.roi_avg_coord_col_start:self.roi_avg_coord_col_start+self.averageimage_ROI_mask.shape[1]] * self.averageimage_ROI_mask
            
            # Sum of all pixel values and devided by non-zero pixel number
            bg_mean = np.sum(ROI_bg)/count_dict[1] 
            
            self.background_trace.append(bg_mean)

        fig, ax0 = plt.subplots()
        fig.suptitle("Raw ROI background trace")
        plt.plot(self.cam_time_label, self.background_trace)
        ax0.set_xlabel('time(s)')
        ax0.set_ylabel('Pixel values')
        fig.savefig(os.path.join(self.main_directory, 'Analysis results//ROI raw background trace.png'))
        plt.show() 
        
        # Use rolling average to smooth the background trace
        self.background_trace = uniform_filter1d(self.background_trace, size=self.background_rolling_average_number)
        
        fig, ax1 = plt.subplots()
        fig.suptitle("Smoothed background trace")

        plt.plot(self.cam_time_label, self.background_trace)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('Pixel values')
        fig.savefig(os.path.join(self.main_directory, 'Analysis results//Smoothed background trace.png'))
        plt.show()    
        
        # For each frame in video, substract the background
        for i in range(self.videostack.shape[0]):
            
            raw_frame = self.videostack[i]
            
            background_mean = self.background_trace[i]
            
            background_mean_2d = background_mean * np.ones((self.videostack.shape[1], self.videostack.shape[2]))
            
            self.videostack[i] = np.abs(raw_frame - background_mean_2d)
            
        print('ROI background correction done.')
        
        # Show the background corrected trace.
        self.mean_camera_counts_backgroubd_substracted = []
        for i in range(self.videostack.shape[0]):
            self.mean_camera_counts_backgroubd_substracted.append(np.mean(self.videostack[i]))
        
        fig2, ax2 = plt.subplots()
        fig2.suptitle("Mean camera intensity after backgroubd substracted")
        plt.plot(self.cam_time_label, self.mean_camera_counts_backgroubd_substracted)
        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('Pixel values')
        fig2.savefig(os.path.join(self.main_directory, 'Analysis results//Mean camera intensity after backgroubd substracted.png'))
        plt.show()
        
        # Updates the mean intensity
        self.imganalysis_averageimage = np.mean(self.videostack, axis = 0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)
        
        fig3 = plt.figure()
        fig3.suptitle("Mean intensity")
        plt.imshow(self.imganalysis_averageimage)
        fig3.savefig(os.path.join(self.main_directory, 'Analysis results//Mean intensity.png'))
        plt.show()

    def display_electrical_signals(self):
        """
        Display the patch clamp electrode recored membrane potential and current signals.

        Returns
        -------
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            
            self.patchcurrentlabel = np.arange(len(self.Ip))/self.samplingrate_curve
            
            self.patchvoltagelabel = np.arange(len(self.Vp))/self.samplingrate_curve
            
            self.electrical_signals_figure, (ax1, ax2) = plt.subplots(2, 1)
            # plt.title('Electrode recording')
            # Current here is already
            # Probe gain: low-100M ohem
            # DAQ recording / 10**8 (voltage to current)* 10**12 (A to pA) == pA
            ax1.plot(self.patchcurrentlabel, self.Ip*10000, label='Current', color='b')
            ax1.set_title('Electrode recording')        
            ax1.set_xlabel('time(s)')
            ax1.set_ylabel('Current (pA)')
            # ax1.legend()
            
            # ax2 = self.electrical_signals_figure.add_subplot(212)
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier.
            ax2.plot(self.patchvoltagelabel, self.Vp*1000/10, label='Voltage', color='b')
            # ax2.set_title('Voltage')        
            ax2.set_xlabel('time(s)')
            ax2.set_ylabel('Volt (mV)')
            # ax2.legend()
            
            plt.show()
            self.electrical_signals_figure.savefig(os.path.join(self.main_directory, 'Analysis results//Electrode recording.png'))
        else:
            pass
        
        
    def video_mean(self):
        """
        Calculating the average 2d image from the video.

        Returns
        -------
        None.

        """
        self.imganalysis_averageimage = np.mean(self.videostack, axis = 0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)
        self.samplingrate_cam = self.Spincamsamplingrate.value()        
        self.cam_time_label = np.arange(self.videostack.shape[0])/self.samplingrate_cam
        
        fig = plt.figure()
        fig.suptitle("Mean intensity of raw video")
        plt.imshow(self.imganalysis_averageimage)
        fig.savefig(os.path.join(self.main_directory, 'Analysis results//Mean intensity of raw video.png'))
        plt.show()
        
        self.mean_camera_counts = []
        for i in range(self.videostack.shape[0]):
            self.mean_camera_counts.append(np.mean(self.videostack[i]))
                
        fig2, ax2 = plt.subplots()
        fig2.suptitle("Mean intensity trace of raw video")
        plt.plot(self.cam_time_label, self.mean_camera_counts)
        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('Pixel values')
        fig2.savefig(os.path.join(self.main_directory, 'Analysis results//Mean intensity trace of raw video.png'))
        plt.show()
            
    def calculate_background_from_ROI_average(self):
        """
        Calculate the mean background value from the ROI selector.

        Returns
        -------
        None.

        """
        self.averageimage_imageitem = self.pw_averageimage.getImageItem()
        self.averageimage_ROI = self.roi_average.getArrayRegion(self.imganalysis_averageimage, self.averageimage_imageitem)
        self.averageimage_ROI_mask = np.where(self.averageimage_ROI > 0, 1, 0)
        
        #self.roi_average_pos = self.roi_average.pos()
        self.roi_average_Bounds = self.roi_average.parentBounds()
        self.roi_avg_coord_col_start = round(self.roi_average_Bounds.topLeft().x())
        self.roi_avg_coord_col_end = round(self.roi_average_Bounds.bottomRight().x())
        self.roi_avg_coord_raw_start = round(self.roi_average_Bounds.topLeft().y())
        self.roi_avg_coord_raw_end = round(self.roi_average_Bounds.bottomRight().y())
        
    def calculateweight(self): 
        t2 = threading.Thread(target = self.calculate_weight)
        t2.start()
    
    def calculate_weight(self):
        """
        Calculate the pixels weights using correlation between the video and voltage reocrding.

        Returns
        -------
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            self.samplingrate_cam = self.Spincamsamplingrate.value()
            self.downsample_ratio = int(self.samplingrate_curve/self.samplingrate_cam) 
            
            print("Vp downsampling ratio: {}".format(self.downsample_ratio))
            print("Sampling rate camera: {}".format(self.samplingrate_cam))
            print("Sampling rate DAQ: {}".format(self.samplingrate_curve))
            
            try:
                self.Vp_downsample = self.Vp.reshape(-1, self.downsample_ratio).mean(axis=1)
                
                self.Vp_downsample = self.Vp_downsample[0:len(self.videostack)]
            except:
                print("Vp downsampling ratio is not an integer.")
                small_ratio = int(np.floor(self.samplingrate_curve/self.samplingrate_cam))
                resample_length = int(len(self.videostack) * small_ratio)
                self.Vp_downsample = signal.resample(self.Vp, resample_length)
                plt.figure()
                plt.plot(self.Vp_downsample)
                plt.show()

                self.Vp_downsample = self.Vp_downsample.reshape(-1, small_ratio).mean(axis=1)
                self.Vp_downsample = self.Vp_downsample[0:len(self.videostack)]
            
            print(self.videostack.shape, self.Vp_downsample.shape)
            self.corrimage, self.weightimage, self.sigmaimage = ProcessImage.extractV(self.videostack, self.Vp_downsample*1000/10) 
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier.
    
            self.pw_weightimage.setImage(self.weightimage)
            
        elif self.switch_Vp_or_camtrace.currentIndex() == 1:
            self.corrimage, self.weightimage, self.sigmaimage = ProcessImage.extractV(self.videostack, self.camsignalsum*1000/10)
    
            self.pw_weightimage.setImage(self.weightimage)

        fig = plt.figure()
        fig.suptitle("Weighted pixels")
        plt.imshow(self.weightimage)
        fig.savefig(os.path.join(self.main_directory, 'Analysis results//Weighted pixel image.png'))
        np.save(os.path.join(self.main_directory,'Analysis results//Weighted pixel image.npy'), self.weightimage)
        plt.show()                

    def display_weighted_trace(self):     
        """
        Display the mean sum weight and display frame by frame.

        Returns
        -------
        None.

        """
        self.videolength = len(self.videostack)    
        
        # datv = squeeze(mean(mean(mov.*repmat(Wv./sum(Wv(:))*movsize(1)*movsize(2), [1 1 length(sig)]))));
        k=np.tile(self.weightimage/np.sum(self.weightimage)*self.videostack.shape[1]*self.videostack.shape[2], (self.videolength,1,1))
        self.weighttrace_tobetime = self.videostack*k
        
        self.weight_trace_data = np.zeros(self.videolength)
        for i in range(self.videolength):
            self.weight_trace_data[i] = np.mean(self.weighttrace_tobetime[i])
            
        self.patch_camtrace_label_weighted = np.arange(self.videolength)/self.samplingrate_cam
        
        np.save(os.path.join(self.main_directory,'Analysis results//Weighted_trace.npy'), self.weight_trace_data)
        
        fig, ax1 = plt.subplots()
        fig.suptitle("Weighted pixel trace")
        plt.plot(self.patch_camtrace_label_weighted, self.weight_trace_data)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('Weighted trace(counts)')
        fig.savefig(os.path.join(self.main_directory, 'Analysis results//Weighted pixel trace.png'))
        plt.show()


    def fit_on_trace(self):
        """
        Using curve fit function to calculate all the statistics.

        Returns
        -------
        None.

        """
        fit = CurveFit\
        (self.weight_trace_data, self.Vp*1000/10, camera_fps = self.samplingrate_cam, DAQ_Hz = self.samplingrate_display_curve, rhodopsin = self.Construct_name.text())    
        fit.Photobleach()
        fit.IsolatePeriods()
        fit.TransformCurves()
        fit.CurveAveraging()
        fit.ExponentialFitting()
        fit.Statistics()

    def save_analyzed_image(self, catag):
        if catag == 'weight_image':
            Localimg = Image.fromarray(self.weightimage) #generate an image object
            Localimg.save(os.path.join(self.savedirectory, 'Weight_' +datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.tif')) #save as tif
     
        
    def clearplots(self):
        self.pw_patch_voltage.clear()
        self.pw_patch_current.clear()
        self.pw_patch_camtrace.clear()
                
        self.vLine_cam = pg.InfiniteLine(pos=0.4, angle=90, movable=True)
        self.pw_patch_camtrace.addItem(self.vLine_cam, ignoreBounds=True)
        self.vLine = pg.InfiniteLine(pos=0.4, angle=90, movable=True)
        self.pw_patch_current.addItem(self.vLine, ignoreBounds=True)
        
        self.vLine.sigPositionChangeFinished.connect(self.showpointdata)
        self.vLine_cam.sigPositionChangeFinished.connect(self.showpointdata_camtrace)

    def MessageToMainGUI(self, text):
        self.MessageBack.emit(text)
        
        
    def send_DMD_mask_contour(self, contour_from_cellselection):
        self.Cellselection_DMD_mask_contour.emit(contour_from_cellselection)
        
    def get_single_waveform(self):
        self.single_waveform_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', 'M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data',"Image files (*.npy)")
        self.textbox_single_waveform_filename.setText(self.single_waveform_fileName)
        
        self.single_waveform = np.load(self.single_waveform_fileName, allow_pickle=True)[5:]
        
        try:
            plt.figure()
            plt.plot(self.single_waveform)
            plt.show()
        except:
            pass
#    def closeEvent(self, event):
#        QtWidgets.QApplication.quit()
#        event.accept()

        
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder='row-major')
        mainwin = AnalysisWidgetUI()
        mainwin.show()
        app.exec_()
    run_app() 