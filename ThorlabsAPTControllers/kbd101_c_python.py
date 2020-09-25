# -*- coding: utf-8 -*-
"""
Simple example showing use of KBD101 with DDSM100
"""


import os
import time


from ctypes import *

# os.chdir(r"C:\Labsoftware\Thorlabs")
lib = cdll.LoadLibrary(r"C:\Labsoftware\Thorlabs\Thorlabs.MotionControl.KCube.BrushlessMotor.dll")



  
#Build device list
lib.TLI_BuildDeviceList()

#set up serial number variable
serialNumber = c_char_p("28251139".encode('utf-8'))
moveTimeout=60.0


#set up device
lib.BMC_Open(serialNumber)
lib.BMC_StartPolling(serialNumber, c_int(200))
lib.BMC_EnableChannel(serialNumber)

time.sleep(3)
    
lib.BMC_ClearMessageQueue(serialNumber)

#home device
print('Homing Device')
homeStartTime=time.time()
lib.BMC_Home(serialNumber)

homed = False
messageType = c_ushort()
messageID = c_ushort()
messageData = c_ulong()
while(homed == False):
    lib.BMC_GetNextMessage(serialNumber, byref(messageType), byref(messageID), byref(messageData))
    if ((messageID.value == 0 and messageType.value == 2) or (time.time()-homeStartTime) > moveTimeout): 
        homed = True
lib.BMC_ClearMessageQueue(serialNumber)


deviceUnit= c_int()

#here, we move to position 100.0 in real units (mm)
realUnit= c_double(100.0)

#Load settings for attached stage
lib.BMC_LoadSettings(serialNumber)

#convert real units to device units
lib.BMC_GetDeviceUnitFromRealValue(serialNumber,realUnit,byref(deviceUnit),0)

#send move command
print('Moving Device')

moveStartTime=time.time()
lib.BMC_MoveToPosition(serialNumber, deviceUnit)

moved=False

while(moved == False):
    lib.BMC_GetNextMessage(serialNumber, byref(messageType), byref(messageID), byref(messageData))
    
    if ((messageID.value == 1 and messageType.value == 2) or (time.time()-moveStartTime) > moveTimeout): 
        moved = True

#clean up and exit
lib.BMC_ClearMessageQueue(serialNumber)
#print lib.BMC_GetPosition()

lib.BMC_StopPolling(serialNumber)

lib.BMC_Close(serialNumber)