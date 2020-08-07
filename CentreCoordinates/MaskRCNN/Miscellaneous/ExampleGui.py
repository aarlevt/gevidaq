# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:31:41 2020

@author: meppenga
"""

 


import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from MaskRCNN.Miscellaneous.CellGuiFigure import CellGuiFigureHandle
from MaskRCNN.Configurations.ConfigFileInferenceOld import cellConfig
try:
    import Tkinter as tk
    from Tkinter.filedialog import askopenfilename
except:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

class CellGui(object):
    
    def __init__(self,config):
        self.config = config
        plt.close('Select Cell GUI')
        self.fig, self.ax = plt.subplots(1, num='Select Cell GUI')
        self.FigureHandle = CellGuiFigureHandle(config, self.fig, self.ax)

        
        # Create Buttons axis
        self.Button_All_Label  = ['Show Cells','Show Mask','Show BBox','Show Label','Show centre']
        self.Button_High_Label = ['Show Mask', 'Show BBox','Show Label','Show centre']
        self.fig.subplots_adjust(right = 0.8)
        self.Ax_Button_All          = self.fig.add_axes([0.81, 0.71, 0.1, 0.2])
        self.Ax_Button_High         = self.fig.add_axes([0.81, 0.41, 0.1, 0.2])
        self.Ax_Button_MultiSelect  = self.fig.add_axes([0.81, 0.21, 0.1, 0.075])
        self.Ax_Button_File         = self.fig.add_axes([0.81, 0.11, 0.1, 0.075])
        
        # Create buttons
        self.Button_All         = CheckButtons(self.Ax_Button_All,   self.FigureHandle.Button_All_Label)
        self.Button_High        = CheckButtons(self.Ax_Button_High,  self.FigureHandle.Button_High_Label)
        self.Button_MultiSelect = CheckButtons(self.Ax_Button_MultiSelect, ['Enable multi select'])
        self.Button_File        = Button(self.Ax_Button_File,'Change file')

        
        # Set button text
        self.Ax_Button_All.text(0.4, 1,'All cells')
        self.Ax_Button_High.text(0.3, 1,'Selected cell')
        
        # Init button state
        self.State_list_all    = [True, False, False, False, False]
        self.State_list_Select = [True, True,  True,  False, True]
        for ii, state in enumerate(self.State_list_all):
            if state:
                self.Button_All.set_active(ii)
        for ii, state in enumerate(self.State_list_Select ):
            if state:
                self.Button_High.set_active(ii)
        if self.FigureHandle.EnableMultiCellSelection:
            self.Button_MultiSelect.set_active(0)
        self.FigureHandle.init_setButtonStates('All',    self.State_list_all)
        self.FigureHandle.init_setButtonStates('Select', self.State_list_Select )
        
            
        # Bind CallBacks
        self.Button_All.on_clicked(self.on_click_All)
        self.Button_High.on_clicked(self.on_click_High)
        self.Button_MultiSelect.on_clicked(self.on_click_MultiSelect)
        self.Button_File.on_clicked(self.ChangeFile)


    def on_click_MultiSelect(self, label):
        self.FigureHandle.EnableMultiCellSelection = self.Button_MultiSelect.get_status()[0]
        

    def on_click_All(self, label):
        """Callback which updates the figure when a checkbox setting is changed"""
        for idx, state in enumerate(self.Button_All.get_status()):
             self.State_list_all[idx] = state
        self.FigureHandle.on_click_All(self.State_list_all)
        self.FigureHandle.DisplayImage()
    
    def on_click_High(self, label):
        """Callback which updates the figure when a checkbox setting is changed"""
        for idx, state in enumerate(self.Button_High.get_status()):
            self.State_list_Select[idx] = state
        self.FigureHandle.on_click_High(self.State_list_Select)
        self.FigureHandle.DisplayImage()

 
    def ChangeFile(self,event):
        """Callback which opens a file selection GUI. A detection will run on
        the selected file and the results will be displayed in the figure"""
        obj = tk.Tk()
        filename = askopenfilename(filetypes = (("png files","*.png"),("jpeg files","*.jpg"),('tiff files','*.tif'),("all files","*.*"))) 
        obj.destroy()
        if filename:
            self.FigureHandle.RunDetection(filename)


    



config = cellConfig()
config.IMAGE_RESIZE_MODE = "square"
config.IMAGE_MIN_DIM = 512
config.IMAGE_MAX_DIM = 1024
config.IMAGE_MIN_SCALE = 2.0
config.LogDir = ''
config.WeigthPath = r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\FinalResults\ModelWeights.h5'    
    
    
CellGui(config)   
    
    
    
    
    
    
    
    
    
    
    
    