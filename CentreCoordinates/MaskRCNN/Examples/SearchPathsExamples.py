# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:18:19 2020

@author: meppenga
"""
import sys
import os
def addSearchPaths():
    path, _= os.path.split( os.path.abspath(__file__))
    path, _= os.path.split(path)
    EnginePath = os.path.join(path, 'Engine')
    if not EnginePath in sys.path:
        sys.path.append(EnginePath)
    from SearchPaths import addSearchPaths as AllSearchPaths
    AllSearchPaths()  
        