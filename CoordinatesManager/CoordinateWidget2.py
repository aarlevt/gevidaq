# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:17:02 2020

@author: ideheer
"""

import sys
import os

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname+'/../')
    # os.chdir(os.getcwd()+'/')

# Backend

from CoordinatesManager.backend.readRegistrationImages import touchingCoordinateFinder
from CoordinatesManager.backend.polynomialTransformation import polynomialRegression
from CoordinatesManager import DMDActuator, ManualRegistration, DMDWidget, GalvoWidget, StageRegistrationWidget

from NIDAQ.DAQoperator import DAQmission
# from NIDAQ.generalDaqerThread import execute_analog_readin_optional_digital_thread
from NIDAQ.wavegenerator import waveRecPic

from ImageAnalysis.ImageProcessing import ProcessImage
from HamamatsuCam.HamamatsuActuator import CamActuator
from GalvoWidget.pmt_thread import pmtimagingTest_contour

# UI
from CoordinatesManager.ui_widgets.adaptedQButtonGroupClass import adaptedQButtonGroup
from CoordinatesManager.ui_widgets.DrawingWidget import DrawingWidget
from CoordinatesManager.ui_widgets.SelectPointImageView import SelectPointImageView

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QWidget, QPushButton, QRadioButton, QVBoxLayout, 
                             QCheckBox, QGridLayout, QHBoxLayout, QVBoxLayout, 
                             QGroupBox, QTabWidget, QGraphicsView, QGraphicsScene, 
                             QListWidget, QSizePolicy, QLabel, QComboBox, QLayout,
                             QStackedWidget, QSpinBox)

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from StylishQT import MySwitch, cleanButton, roundQGroupBox, SquareImageView

import pyqtgraph as pg
from pyqtgraph import QtGui

# General libraries
import threading
import sys
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

class CoordinatesWidgetUI(QWidget):
    
    sig_cast_mask_coordinates_to_dmd = pyqtSignal(list)
    sig_cast_mask_coordinates_to_galvo = pyqtSignal(list)
    sig_start_registration = pyqtSignal()
    sig_finished_registration = pyqtSignal()
    sig_cast_camera_image = pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.main_application = parent
        self.init_gui()
        self.sig_to_calling_widget = []
        
    def closeEvent(self, event):
        try:
            self.DMD
        except:
            pass
        else:
            self.DMD.disconnect_DMD()
            
        QtWidgets.QApplication.quit()
        event.accept()
    #%%
    def init_gui(self):
        self.setWindowTitle("Coordinate control")
        
        self.layout = QGridLayout()
        self.setMinimumSize(1250,1000)
        self.setLayout(self.layout)
        
        self.image_mask_stack = QTabWidget()
        
        self.selection_view = DrawingWidget(self)
        self.selection_view.enable_drawing(True)
        self.selection_view.getView().setLimits(xMin = 0, xMax = 2048, yMin = 0, yMax = 2048, minXRange = 2048, minYRange = 2048, maxXRange = 2048, maxYRange = 2048)
        self.selection_view.ui.roiBtn.hide()
        self.selection_view.ui.menuBtn.hide() 
        self.selection_view.ui.normGroup.hide()
        self.selection_view.ui.roiPlot.hide()
        # self.selection_view.setImage(plt.imread('CoordinatesManager/Registration_Images/StageRegistration/Distance200_Offset0/A1.png'))
        
        self.mask_view = SquareImageView()
        self.mask_view.getView().setLimits(xMin = 0, xMax = 2048, yMin = 0, yMax = 2048, minXRange = 2048, minYRange = 2048, maxXRange = 2048, maxYRange = 2048)
        self.mask_view.ui.roiBtn.hide()
        self.mask_view.ui.menuBtn.hide() 
        self.mask_view.ui.normGroup.hide()
        self.mask_view.ui.roiPlot.hide()
        self.mask_view.ui.histogram.hide()
        
        self.image_mask_stack.addTab(self.selection_view, 'Select')
        self.image_mask_stack.addTab(self.mask_view, 'Mask')
        
        self.layout.addWidget(self.image_mask_stack, 0, 0, 5, 1)
        
        # ---------------------- Mask generation Container  --------------
        
        self.maskGeneratorContainer = roundQGroupBox()
        self.maskGeneratorContainer.setFixedSize(320, 220)
        self.maskGeneratorContainer.setTitle("Mask generator")
        self.maskGeneratorContainerLayout = QGridLayout()
        
        self.maskGeneratorLayout = QGridLayout()
        self.maskGeneratorContainer.setLayout(self.maskGeneratorContainerLayout)
        
        self.loadMaskFromFileButton = QPushButton('Open mask')
        self.loadMaskFromFileButton.clicked.connect(self.load_mask_from_file)
        
        self.addRoiButton = QPushButton("Add ROI")
        self.createMaskButton = QPushButton("Add mask")
        self.deleteMaskButton = QPushButton("Delete mask")
        self.removeSelectionButton = cleanButton()
        self.removeSelectionButton.setFixedHeight(25)
        self.removeSelectionButton.setToolTip("Remove rois and output signals")
        self.addRoiButton.clicked.connect(self.add_polygon_roi)
        
        self.createMaskButton.clicked.connect(self.create_mask)
        self.deleteMaskButton.clicked.connect(self.delete_mask)
        self.removeSelectionButton.clicked.connect(self.remove_selection)
        
        self.maskGeneratorContainerLayout.addWidget(self.addRoiButton, 1, 0)
        self.maskGeneratorContainerLayout.addWidget(self.createMaskButton, 2, 0)
        self.maskGeneratorContainerLayout.addWidget(self.deleteMaskButton, 2, 1)
        self.maskGeneratorContainerLayout.addWidget(self.removeSelectionButton, 1, 1)
        self.selectionOptionsContainer = roundQGroupBox()
        self.selectionOptionsContainer.setTitle('Options')
        self.selectionOptionsLayout = QGridLayout()
        self.fillContourButton = QCheckBox()
        self.invertMaskButton = QCheckBox()
        self.thicknessSpinBox = QSpinBox()
        self.thicknessSpinBox.setRange(1, 25)
        self.selectionOptionsLayout.addWidget(QLabel('Fill contour:'), 0, 0)
        self.selectionOptionsLayout.addWidget(self.fillContourButton, 0, 1)
        self.selectionOptionsLayout.addWidget(QLabel('Invert mask:'), 1, 0)
        self.selectionOptionsLayout.addWidget(self.invertMaskButton, 1, 1)
        self.selectionOptionsLayout.addWidget(QLabel('Thickness:'), 2, 0)
        self.selectionOptionsLayout.addWidget(self.thicknessSpinBox, 2, 1)
        
        lasers = ['640', '532', '488']
        self.transform_for_laser_menu = QListWidget()
        self.transform_for_laser_menu.addItems(lasers)
        self.transform_for_laser_menu.setFixedHeight(48)
        self.transform_for_laser_menu.setFixedWidth(65)
        self.transform_for_laser_menu.setCurrentRow(0)
        
        self.selectionOptionsLayout.addWidget(QLabel('To be used with laser:'), 0, 2)
        self.selectionOptionsLayout.addWidget(self.transform_for_laser_menu, 1, 2)
        
        self.selectionOptionsContainer.setLayout(self.selectionOptionsLayout)
        
        self.snapFovButton = QPushButton('Image FOV')
        self.snapFovButton.clicked.connect(self.snap_fov)
        
        self.maskGeneratorContainerLayout.addWidget(self.snapFovButton, 0, 0, 1, 1)
        self.maskGeneratorContainerLayout.addWidget(self.loadMaskFromFileButton, 0, 1, 1, 1)
        self.maskGeneratorContainerLayout.addWidget(self.selectionOptionsContainer, 3, 0, 2, 2)
        
        self.layout.addWidget(self.maskGeneratorContainer, 0, 1)
        
        self.DMDWidget = DMDWidget.DMDWidget()
        self.layout.addWidget(self.DMDWidget, 1, 1)
        
        """--------------------------------------------------------------------
        # Singal sent out from DMDWidget to ask for mask generated here.
        # And then the generated roi list is sent back to function:receive_mask_coordinates in DMDWidget.
        #  --------------------------------------------------------------------
        """
        self.DMDWidget.sig_request_mask_coordinates.connect(lambda: self.cast_mask_coordinates('dmd'))
        self.sig_cast_mask_coordinates_to_dmd.connect(self.DMDWidget.receive_mask_coordinates)
        
        self.DMDWidget.sig_start_registration.connect(lambda: self.sig_start_registration.emit())
        self.DMDWidget.sig_finished_registration.connect(lambda: self.sig_finished_registration.emit())
        
        self.GalvoWidget = GalvoWidget.GalvoWidget()
        self.layout.addWidget(self.GalvoWidget, 2, 1)
        
        self.GalvoWidget.sig_request_mask_coordinates.connect(lambda: self.cast_mask_coordinates('galvo'))
        self.sig_cast_mask_coordinates_to_galvo.connect(self.GalvoWidget.receive_mask_coordinates)
        self.GalvoWidget.sig_start_registration.connect(lambda: self.sig_start_registration.emit())
        self.GalvoWidget.sig_finished_registration.connect(lambda: self.sig_finished_registration.emit())
        
        self.ManualRegistrationWidget = ManualRegistration.ManualRegistrationWidget(self)
        self.ManualRegistrationWidget.sig_request_camera_image.connect(self.cast_camera_image)
        self.sig_cast_camera_image.connect(self.ManualRegistrationWidget.receive_camera_image)
        
        self.layout.addWidget(self.ManualRegistrationWidget, 3, 1)
        
        self.StageRegistrationWidget = StageRegistrationWidget.StageWidget(self)
        self.layout.addWidget(self.StageRegistrationWidget, 4, 1)
    #%%
        
    def cast_transformation_to_DMD(self, transformation, laser):
        self.DMDWidget.transform[laser] = transformation
        self.DMDWidget.save_transformation()
    
    def cast_transformation_to_galvos(self, transformation):
        self.GalvoWidget.transform = transformation
        self.GalvoWidget.save_transformation()
        
    def cast_camera_image(self):
        image = self.selection_view.image
        if type(image) == np.ndarray:
            self.sig_cast_camera_image.emit(image)
        
    def snap_fov(self):
        self.DMDWidget.interupt_projection()
        
        self.DMDWidget.project_full_white()
        
        self.cam = CamActuator()
        self.cam.initializeCamera()
        image = self.cam.SnapImage(0.04)
        self.cam.Exit()
        self.selection_view.setImage(image)
        
    def cast_mask_coordinates(self, receiver):
        """
        Upon receiving mask rois request signal from DMD or galvo widget, generate list signal including
        list of rois from the current ROIitems in "Select" Drawwidget and send it back to the calling widget.
        
        Parameters
        ----------
        receiver : string.
            Specifies which widget is requesting.
                   
        """        
        if receiver == 'dmd':
            self.sig_cast_mask_coordinates_to_dmd.emit(self.sig_to_calling_widget)
        else:
            self.sig_cast_mask_coordinates_to_galvo.emit(self.sig_to_calling_widget)
        
    def get_list_of_rois(self):
        """
        Return the list of rois from the current ROIitems in "Select" Drawwidget.
        """
        view = self.selection_view
        list_of_rois = []
        
        for roi in view.roilist:
            roi_handle_positions = roi.getLocalHandlePositions()
            roi_origin = roi.pos()
            
            for idx, pos in enumerate(roi_handle_positions):
                roi_handle_positions[idx] = pos[1]
                
            num_vertices = len(roi_handle_positions)
            vertices = np.zeros([num_vertices,2])

            for idx, vertex in enumerate(roi_handle_positions):
                vertices[idx,:] = np.array([vertex.x() + roi_origin.x(), vertex.y() + roi_origin.y()])
            
            list_of_rois.append(vertices)
        
        return list_of_rois
        
    def create_mask(self):
        """
        Create untransformed binary mask, sent out the signal to DMD widget for
        further transformation.

        Returns
        -------
        None.

        """
        flag_fill_contour = self.fillContourButton.isChecked()
        flag_invert_mode = self.invertMaskButton.isChecked()
        contour_thickness = self.thicknessSpinBox.value()
        target_laser = self.transform_for_laser_menu.selectedItems()[0].text()
        
        # Get the list of rois from the current ROIitems in "Select" Drawwidget.
        list_of_rois = self.get_list_of_rois()
        
        # Signal to mask requesting widget.
        self.current_mask_sig = [list_of_rois, flag_fill_contour, contour_thickness, flag_invert_mode, target_laser]
        
        #---- This is the roi list sent to DMD to generate final stack of masks.----
        self.sig_to_calling_widget.append(self.current_mask_sig)
        
        # Show the untransformed mask
        self.current_mask = ProcessImage.CreateBinaryMaskFromRoiCoordinates(list_of_rois, \
                                                       fill_contour = flag_fill_contour, \
                                                       contour_thickness = contour_thickness, \
                                                       invert_mask = flag_invert_mode)
        
        self.mask_view.setImage(self.current_mask)
        
    def delete_mask(self):
        """
        Remove the last mask from the list.
        """
        del self.current_mask_sig[-1]
        
    def remove_selection(self):
        self.sig_to_calling_widget = []
        self.selection_view.clear_rois()
    
    def set_camera_image(self, sig):
        self.selection_view.setImage(sig)

    def add_polygon_roi(self):
        view = self.selection_view
        
        x = (view.getView().viewRect().x()) * 0.3
        y = (view.getView().viewRect().y()) * 0.3
        a = (view.getView().viewRect().width() + x) * 0.3
        b = (view.getView().viewRect().height() + y) * 0.3
        c = (view.getView().viewRect().width() + x) * 0.7
        d = (view.getView().viewRect().height() + y) * 0.7
        polygon_roi = pg.PolyLineROI([[a,b], [c,b], [c,d], [a,d]], pen = view.pen,closed=True, movable=True, removable=True)
        
        view.getView().addItem(polygon_roi)
        view.append_to_roilist(polygon_roi)

    def load_mask_from_file(self):
        """
        Open a file manager to browse through files, load image file
        """
        self.loadFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select file', './CoordinateManager/Images/',"(*.png, *.tiff, *.jpg)")
        try:
            image = plt.imread(self.loadFileName)
            
            self.current_mask = image
            self.mask_view.setImage(self.current_mask)
        except:
            print('fail to load file.')
    
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = CoordinatesWidgetUI()
        mainwin.show()
        app.exec_()
    run_app()