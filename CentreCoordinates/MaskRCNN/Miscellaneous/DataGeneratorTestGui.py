# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:40:52 2020

@author: meppenga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import MaskRCNN.Miscellaneous.visualize as visualize
from  MaskRCNN.Miscellaneous.Timer import Timer

class DataGeneratorTestGui(object):
    
    def __init__(self, config, generator):
        self.num = 0
        self.loadingtime = 0
        self.Timer = Timer()
        self.config = config
        plt.close('Test2')
        self.fig, self.ax = plt.subplots(1,1,num='Test2')
        self.fig.subplots_adjust(right = 0.8)
        self.axbutton = self.fig.add_axes([0.81, 0.71, 0.1, 0.2])
        self.axbuttonText = self.fig.add_axes([0.81, 0.01, 0.1, 0.05])
        
        self.ButtonText     = TextBox(self.axbuttonText,'Index')
        self.ButtonText.on_submit(self.tetx_fcn)
        self.Button_All     = Button(self.axbutton, 'Next')
        self.Button_All.on_clicked(self.CreatePlot_fcn)
        self.fig_msg = self.fig.text(0,0.01,'')
        self.fig.show()
        self.generator = generator
        self.CreatePlot_fcn(None)
        
        
        

    def CreatePlot_fcn(self,event):
        self.Updatemsg('Loading data')
        self.Timer.tic()
        r,_ = self.generator.__getitem__(self.num)
        self.loadingtime = self.Timer.toc_ns()*1e-9
        self.Updatemsg('Parse data')
        imagePath = self.generator.dataset.image_info[self.generator.CurrentImageIds[0]]['path']           
        zero_ix = np.where(r[4][0,:,:] == 0)[0][0]
        image = (r[0][0,:,:,0:zero_ix] + self.config.MEAN_PIXEL).astype(np.uint8)
        mask = r[5][0,:,:,0:zero_ix]
        box = r[4][0,:,:][0:zero_ix]
        class_ids = r[3][0,0:zero_ix]
        coor = r[-1][0,:,:][0:zero_ix]

        self.Updatemsg('Creating figure')
        self.fig_msg.set_text(str(self.num))
        self.num += 1 
        self.num = self.num % (len(self.generator.dataset.image_ids))       
        self.ax.clear()
        visualize.display_instances(image, box, mask, class_ids,
                                ['BG'] + self.config.ValidLabels, ax=self.ax,
                                centre_coors=coor, Centre_coor_radius = 2 )
        self.ax.set_title(imagePath)
        self.Updatemsg('Creating figure done')
  
    def tetx_fcn(self,text):
        self.num = min(int(text),len(self.generator.dataset.image_ids)-1)
        self.CreatePlot_fcn(None)


    def Updatemsg(self,msg):
        self.fig_msg.set_text('Index: '+str(self.num)+'\n'+msg+'\nData loading time (s): '+str(round(self.loadingtime,4)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

















