# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:17:56 2020

@author: meppenga
"""

from matplotlib.patches import Circle, Polygon

class MovePolygonPoint(object):
    
    def __init__(self, fig, ax, enable=True):
        self.fig = fig
        self.ax = ax
        self.enable = enable
        self.ax.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',  self.on_mouseMove)
        # self.fig.canvas.mpl_connect('resize_event',  self.on_resize_event)
        
        
        self.points = [None]
        self.radius = 12
        
        # Init state variables
        self.mousePress          = False
        self.PointSelected       = False
        self.FirstMoveAfterPress = True
        self.pointsMoved         = False
        
        self.index_point          = -1
        self.polygonPatch         = []
        self.polypointsPointPatch = []
        self.width = 30
        
        self._On_moved_Polygon = self.NoneFunc
        
    
    def Reset(self):
        self.removePoints()
        self.RemovePolygon()
        self.polygonPatch         = []
        self.polypointsPointPatch = []
        self.points               = [None]
        self.index_point          = -1
        self.Update_figure()
        
    
    def ChangePoints(self, points):
        self.Reset()
        self.points = points
        axsize = self.ax.get_xlim()
        axsize = axsize[1] - axsize[0]
        radius = self.radius*axsize/2500
        for point in points:
            patch = Circle(point,radius=radius,color='red')
            self.polypointsPointPatch.append(patch)
            self.ax.add_patch(patch)
        self.polygonPatch = Polygon(self.points, facecolor="none", edgecolor='red')
        self.ax.add_patch(self.polygonPatch)
        self.Update_figure()
    
    def Addpacthes(self):
        self.RemovePolygon()
        self.removePoints()
        for patch in self.polypointsPointPatch:
            self.ax.add_patch(patch)
        self.ax.add_patch(self.polygonPatch)
        self.Update_figure()
        
    
    def RemovePolygon(self):
        try:
            self.polygonPatch.remove()
        except:
            pass
    
    def removePoints(self):
        for point in self.polypointsPointPatch:
            try:
                point.remove()
            except:
                break
    
    def toggle_enable(self,Value=None):
        if Value == None:
            self.enable = not self.enable
        else:
            self.enable = Value
        self.index_point = -1
        self.SetVisibility(self.enable)
            
    
    def SetVisibility(self,Value):
        for point in self.polypointsPointPatch:
            try:
                point.set_visible(Value)
            except:
                pass
        try:
            self.polygonPatch.set_visible(Value)
        except:
            pass
        self.Update_figure()
    
    
    def NoneFunc(self):
        pass
    
    def on_resize_event(self,event):
        if self.enable and len(self.points) > 1:
            axsize = self.ax.get_xlim()
            axsize = axsize[1] - axsize[0]
            radius = self.radius*axsize/2500
            if self.polypointsPointPatch[0].radius != radius:
                self.RemovePolygon()
                self.removePoints()
                for point in self.points:
                    patch = Circle(point,radius=radius,color='red')
                    self.polypointsPointPatch.append(patch)
                    self.ax.add_patch(patch)
                self.polygonPatch = Polygon(self.points, facecolor="none", edgecolor='red')
                self.ax.add_patch(self.polygonPatch)
                self.Update_figure()
    
    def On_moved_Polygon(self,function):
        self._On_moved_Polygon = function
    
    def on_press(self,event):
        self.mousePress          = True
        self.FirstMoveAfterPress = True
           
    def on_release(self,event):
        self.FirstMoveAfterPress = False
        self.mousePress          = False
        if self.pointsMoved and self.enable:
            self._On_moved_Polygon()
            self.pointsMoved= False
    
    def on_mouseMove(self,event):
        self.on_resize_event(event)
        if self.mousePress and self.enable:
            self.MovePoint(event)
        self.FirstMoveAfterPress = False
            
    def Update_figure(self):
        """Updates the canvas figure"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def MovePoint(self,event):
        # one should first select a polygon before the possibility of moving a
        # point, this will speed up it significantly
        if event.xdata == None or event.ydata == None:
            return
        if self.FirstMoveAfterPress:
            axsize = self.ax.get_xlim()
            axsize = axsize[1] - axsize[0]
            width  = self.width*axsize/2500
            x      = event.xdata
            y      = event.ydata
            self.index_point = -1
            if len(self.points) > 1:
                for idx, point in enumerate(self.points):
                    if  x < (point[0]+width) and x > (point[0]-width) and y < (point[1]+width) and y > (point[1]-width):
                        self.index_point = idx
                        break
        else:
            if self.index_point > -1 and len(self.points) > 1:
                self.pointsMoved = True
                x = event.xdata
                y = event.ydata
                # Update selected point position
                self.points[self.index_point] = (x,y)
                
                try:
                    self.polygonPatch.remove()
                    self.polypointsPointPatch[self.index_point].remove()
                except:
                    pass
                # Update patches
                axsize = self.ax.get_xlim()
                axsize = axsize[1] - axsize[0]
                radius = self.radius*axsize/2500
                self.polypointsPointPatch[self.index_point] = Circle((x,y),radius=radius,color='red')
                self.polygonPatch = Polygon(self.points, facecolor="none", edgecolor='red')
                # Update figure
                self.ax.add_patch(self.polygonPatch)  
                self.ax.add_patch(self.polypointsPointPatch[self.index_point])  
                self.Update_figure()
     
                
     
        
     
        
     
        
     
        