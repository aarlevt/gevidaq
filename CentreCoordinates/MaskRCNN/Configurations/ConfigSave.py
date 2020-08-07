# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:46:43 2020

@author: meppenga
"""
import os
from datetime import datetime
def CreateSavePath():
    path, _= os.path.split( os.path.abspath(__file__))
    path, _= os.path.split(path)
    path = os.path.join(path,'Data\\Inference')
    Name     = 'Cell_'
    Date     = datetime.now().strftime("%Y%m%dT%H%M%S")
    return os.path.join(path,Name + Date)

class ConfigSave():
    SaveDir = CreateSavePath()
    CreateSaveDir    = True
    FigSize          = (20,10)
    Add_GT           = True
    Add_orgIm        = True
    Captions         = True
    dispMask         = True
    dispCentreCoor   = True
    CenterCoorRadius = 5
    

    
    