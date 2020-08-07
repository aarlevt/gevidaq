# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:41:05 2020

@author: meppenga
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, Polygon



import tkinter as tk
from matplotlib.path import Path as CreatePolygon
import numpy as np

from MaskRCNN.Miscellaneous.MovePolygonPoint import MovePolygonPoint



class DrawPolygons(object):
    
    def __init__(self, fig, ax, enable=True):
        """GUI to draw polygons. When enabled one can dawn a polygon by clicking
        on the figure. to finish a polygon click on the first point.
        to remove a coordinate while drawing hit the n key
        to added a callback after a polygon is created use the On_PolygonCreation
        method.
        All polygons are stored in the Mask instance. This is a list with on each 
        entry an array with the mask of size of the bounding box enclosing the polygon
        All bounding box coordinates are stored in the Bbox instance. These are saved
        as a list of array os size 1xy as y1,x1,y2,x2
        The names given to a polygon are stored in the names instance. This is a
        list of strings
        All the polygon coordinates are stored in the polypoints instance. This
        is a list containing a list of tuples with the polygon coordiantes stored
        as (x,y)
        """
        self.enable = enable
        self.fig = fig
        self.ax = ax
        
        self.points          = []
        self.pacthesCircle   = []
        self.pacthesLines    = []  
        self.pachesPolygons  = []
        self.polypointsPatch = []
        self.polygons        = []
        self.polypoints      = []
        self.Names           = []
        self.Mask            = []
        self.Bbox            = []
    
        self.width = 30
        self.radius = 6
        self.PolyCreate_Fcn = self.NoneFunction
        self.fig.canvas.mpl_connect('key_press_event', self.Key_press_event)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouseMove)
        self.mousePress          = False       
        self.MovePolygonPoint = MovePolygonPoint(self.fig, self.ax, True)
        self.MovePolygonPoint.On_moved_Polygon(self._On_moved_Polygon)
        
        self.MovePolygonPrev_index = None
        self.xy = (0,0)
        # self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onrelease(self, event):
        """Checks when the mouse button is released if the mouse location 
        is at the same location as when one pressed the mouse button
        if True it adds a point to the figure. this allows one to zoom in or
        out or drag something along the figure without creating a point"""
        self.mousePress          = False
        if self.enable:
            if event.xdata == None or event.ydata == None:
                return
            if self.Compare_points(self.xy, (event.xdata,event.ydata)):
                self.on_Mouse_Click(event)
    
    def onpress(self,event):
        """Stores the mouse location when one pressed the mouse button"""
        self.mousePress    = True
        self.xy = (event.xdata,event.ydata)
    
    def on_mouseMove(self,event):
        pass
                
    def toggle_enable_MovePolygonPoint(self, value=None):
        self.MovePolygonPoint.toggle_enable(value)

    def Move_Points_Polygon(self, index):
        if self.MovePolygonPrev_index != None:
            self.pachesPolygons[self.MovePolygonPrev_index].set_visible(True)
            for points in self.polypointsPatch[self.MovePolygonPrev_index]:
                points.set_visible(True)
        if index > (len(self.polypointsPatch) -1):
            print('Cannot change selected polygon for polygon pointmover as index selected index is out of bounds\n'+
                  'DrawPolygons does not contain any polygons')
            return
        self.pachesPolygons[index].set_visible(False)  
        for points in self.polypointsPatch[index]:
            points.set_visible(False)
        self.MovePolygonPoint.ChangePoints(self.polypoints[index])
        self.MovePolygonPrev_index = index
        
    def _On_moved_Polygon(self):
        if self.MovePolygonPrev_index != None:
            self.pachesPolygons[self.MovePolygonPrev_index]  = self.MovePolygonPoint.polygonPatch
            self.polypointsPatch[self.MovePolygonPrev_index] = self.MovePolygonPoint.polypointsPointPatch
            self.polypoints[self.MovePolygonPrev_index]      = self.MovePolygonPoint.points
            bbox, mask = self.CreateMaskFromPolygon(self.polypoints[self.MovePolygonPrev_index])
            self.Mask[self.MovePolygonPrev_index] = mask
            self.Bbox[self.MovePolygonPrev_index] = bbox

        
        
        

    def Key_press_event(self,event):
        """Removes the last added point when one presses the n key
        backspace is not used as this key is a default key of a matplotlib figure
        callback"""
        if self.enable and event.key == 'n':
            if len(self.points) > 1:
                self.pacthesCircle[-1].remove()
                self.pacthesLines[-1].remove()
                self.points.pop(-1)
                self.pacthesCircle.pop(-1)
                self.pacthesLines.pop(-1)
            elif len(self.points) == 1:
                self.pacthesCircle[-1].remove()
                self.points        = []
                self.pacthesCircle = []
                self.pacthesLines  = []
            self.Update_figure()
            
            
    def Update_figure(self):
        """Updates the canvas figure"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def toggle_enable(self,value=None):
        """Enables or diables drawing and reset a
        the points constants"""
        try:
            self.InputBox_Cancel_Fcn()
        except:
            pass
        self.Reset(All = False)
        if value == None:
            self.enable = not self.enable
        else:
            self.enable = value
            
    def Reset(self,All = True):
        """Resets all variables 
        us the option With All to reset all variables
        or only the polygon point of not fully created polygons"""
        self.RemovePatchesPoints()
        self.points         = []
        self.pacthesCircle  = []
        self.pacthesLines   = []        
        if All:
            self.RemovePatchesPolygons()
            self.pachesPolygons  = []
            self.polygons        = []
            self.polypoints      = []
            self.polypointsPatch = []
            self.Names           = []
            self.Mask            = []
            self.Bbox            = []
            self.MovePolygonPrev_index = None
            self.MovePolygonPoint.Reset()


    def RemovePatchesPoints(self):
        if len(self.points) > 1:
            self.pacthesCircle[-1].remove()
            for a,b in zip(self.pacthesCircle,self.pacthesLines):
                a.remove()
                b.remove()
        elif len(self.points) == 1:
            self.pacthesCircle[-1].remove()
        self.Update_figure()
    
    def RemovePatchesPolygons(self):
        if len(self.pachesPolygons) > 0:
            for patch in self.pachesPolygons:
                try:
                    patch.remove()
                except:
                    pass
            for points in self.polypointsPatch:
                try:
                    patch.remove()
                except:
                    pass
        
    def NoneFunction(self):
        pass
    
    def On_PolygonCreation(self,function):
        """Binds a callback. This callback is excecuted when a polgygon is created
        The input function should be a function which takes zero inputs and returns
        nothing """
        self.PolyCreate_Fcn = function
        pass
        
    def CheckPoints(self,x,y):
        """Verifies is a newly created point corespond to the first polygon point
        returns a boolean True"""
        # resize the with for when one zooms in
        axsize = self.ax.get_xlim()
        axsize = axsize[1] - axsize[0]
        width = self.width*axsize/2500
        if x < (self.points[0][0]+width) and x > (self.points[0][0]-width) and y < (self.points[0][1]+width) and y > (self.points[0][1]-width):      
            return True
        else:
            return False
    

    
    
    def on_Mouse_Click(self,event):
        """Handles a mouse click. Addes point to figure, draws polygons
        and creates polygon if one cliks on the starting point"""
        try:
            # check if submit polygon window is open
             self.tk_root.state()
             return
        except:
             pass
        axsize = self.ax.get_xlim()
        axsize = axsize[1] - axsize[0]
        radius = self.radius*axsize/2500
        
        x = event.xdata
        y = event.ydata
        if x == None or y == None:
            x = 0.5
            y = 0.5
        if x >1.01 and y > 1.01:     
            self.points.append((x,y))
            if len(self.points) > 1:
                if self.CheckPoints(x,y):                
                    self.RemovePatchesPoints()
                    poly = Polygon(self.points,facecolor="none", edgecolor='red')
                    self.ax.add_patch(poly)
                    self.Update_figure()
                    self.InputBox()
                    if self.InputSubmit:
                        self.pachesPolygons.append(poly)
                        self.polypointsPatch.append(self.pacthesCircle)
                        bbox, mask = self.CreateMaskFromPolygon(self.points)
                        self.Mask.append(mask)
                        self.Bbox.append(bbox)
                        self.Names.append(self.Obj_name)
                        self.polypoints.append(self.points)
                        self.pacthesCircle = []
                        self.pacthesLines  = []
                        self.points = []
                        for points in self.polypointsPatch[-1]:
                            self.ax.add_patch(points)
                        self.PolyCreate_Fcn()
                    else:
                        self.points.pop(-1)
                        poly.remove()
                        self.ax.add_patch(self.pacthesCircle[-1])
                        for a,b in zip(self.pacthesCircle,self.pacthesLines):
                            self.ax.add_patch(a)
                            self.ax.add_patch(b)
                else:
                    num = len(self.points) - 1
                    patch1 = ConnectionPatch(self.points[num],self.points[num-1],"data","data",color='green')
                    self.pacthesLines.append(patch1)
                    self.ax.add_patch(patch1)
                    patch2 = Circle((x,y),radius=radius,color='green')
                    self.ax.add_patch(patch2)
                    self.pacthesCircle.append(patch2)           
            else:        
                patch = Circle((x,y),radius=radius,color='green')
                self.pacthesCircle.append(patch)
                self.ax.add_patch(patch)      
            self.Update_figure()

    
    def InputBox(self):
        """Assigns name for the polygon using a question box"""
        self.InputSubmit = False
        self.tk_root = tk.Tk()
        tk.Label(self.tk_root, 
                 text="Object name").grid(row=0)
        
        self.SubmitBox = tk.Entry(self.tk_root)
        self.SubmitBox.grid(row=0, column=1)
        self.SubmitBox.bind('<Return>', self.InputBox_On_Return)
        tk.Button(self.tk_root, text='Cancel', command=self.InputBox_Cancel_Fcn).grid(row=3, 
                   column=0, sticky=tk.W, pady=4)
        tk.Button(self.tk_root, text='Submit', command=self.InputBox_Submit_Fcn).grid(row=3, 
                   column=1, sticky=tk.W, pady=4)
        tk.mainloop()

        
    def InputBox_Cancel_Fcn(self):
        self.InputCancel = True
        self.tk_root.destroy()
        
    def InputBox_On_Return(self,event):
        self.InputBox_Submit_Fcn()
        
    def InputBox_Submit_Fcn(self):     
        self.Obj_name = self.SubmitBox.get()
        if self.Obj_name:
            self.InputSubmit = True
            self.tk_root.destroy()
        else:
            print('Object name cannot be an empty string')
            
    def Compare_points(self,Points_a,Points_b,threshold=0.001):
        """This function compares if two points are close to each other
        To do this it compares the two selected points by verifing if the
        first point is within a delta_x and delta_y of the second point
        These deltas are determined using the percentage of the number of
        pixels in the x and y frame of the image
        Inputs:
            Points_a: tuple (x,y)
            Points_b: tuple (x,y)
            threshold: float, """
        axsize_x = self.ax.get_xlim()
        Delta_x  = abs((axsize_x[1] - axsize_x[0])*threshold)
        axsize_y = self.ax.get_ylim()
        Delta_y  = abs((axsize_y[0] - axsize_y[1])*threshold)
        value_x  = Points_a[0] < (Points_b[0] + Delta_x) and Points_a[0] > (Points_b[0] - Delta_x)
        value_y  = Points_a[1] < (Points_b[1] + Delta_y) and Points_a[1] > (Points_b[1] - Delta_y)
        return value_x and value_y

    
    def CreateMaskFromPolygon(self, polygonPoints):
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
        xmax = int(np.ceil(xmax+1))
        xmin = int(max(np.floor(xmin-1),0))
        ymax = int(np.ceil(ymax+1))
        ymin = int(max(np.floor(ymin-1),0))
        
        # Create the grid only for the part that contains the polygons
        # Otherwise we are calculating many times a zero which makes the algorithm slow
        x, y   = np.meshgrid(np.arange(xmin,xmax), np.arange(ymin,ymax))
        x, y   = x.flatten(), y.flatten()
        p = CreatePolygon(polygonPoints)
        grid = p.contains_points(np.vstack((x,y)).T)
        return np.array([ymin, xmin, ymax, xmax],dtype=np.int32), grid.reshape(ymax-ymin, xmax-xmin).astype(np.float32)
        
if __name__ == '__main__':
    plt.close('Draw')
    fig, ax = plt.subplots(1,num='Draw')
    def printmsg():
        print('It works')
    
    ax.imshow(plt.imread(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\JSF_Example.jpg'))
    ax.axis('off')
    drawer = DrawPolygons(fig,ax)
    drawer.On_PolygonCreation(printmsg)
    # fig.canvas.mpl_connect('button_press_event', drawer.on_Mouse_Click)


































