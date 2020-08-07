# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:41:41 2019

@author: xinmeng
"""
from pipython import GCSDevice, pitools
from random import uniform

CONTROLLERNAME = 'C-863.11'
#STAGES = None
STAGES = ('RS40 DS')
REFMODE = ('FNL')

with GCSDevice() as pidevice:
        # InterfaceSetupDlg() is an interactive dialog. There are other methods to
        # connect to an interface without user interaction.
        print(pidevice.EnumerateUSB())
        serialstring = pidevice.EnumerateUSB()

        #pidevice.InterfaceSetupDlg(key='sample')
        # pidevice.ConnectRS232(comport=1, baudrate=115200)
        pidevice.ConnectUSB(serialnum=serialstring[0])
        # pidevice.ConnectTCPIP(ipaddress='192.168.178.42')

        # Each PI controller supports the qIDN() command which returns an
        # identification string with a trailing line feed character which
        # we "strip" away.

        print('connected: {}'.format(pidevice.qIDN().strip()))

        # Show the version info which is helpful for PI support when there
        # are any issues.

        if pidevice.HasqVER():
            print('version info: {}'.format(pidevice.qVER().strip()))

        print('done - you may now continue with the simplemove.py example...')
        print('initialize connected stages...')
        allaxes = pidevice.qSAI_ALL()
        
        #pidevice.StopAll()
        pidevice.SVO(pidevice.axes, [True] * len(pidevice.axes))
        pitools.waitontarget(pidevice, axes=pidevice.axes)
        
        #pitools.startup(pidevice, stages=STAGES, refmode=REFMODE)

        # Now we query the allowed motion range of all connected stages.
        # GCS commands often return an (ordered) dictionary with axes/channels
        # as "keys" and the according values as "values".

        rangemin = list(pidevice.qTMN().values())
        rangemax = list(pidevice.qTMX().values())
        ranges = zip(rangemin, rangemax)
        
        targets = [-10.8]
        pidevice.MOV(pidevice.axes, targets)
        pitools.waitontarget(pidevice)
        positions = pidevice.qPOS(pidevice.axes)
        for axis in pidevice.axes:
            print('position of axis {} = {:.2f}'.format(axis, positions[axis]))
            
        print('done')
        pidevice.CloseConnection()

        # To make this sample a bit more vital we will move to five different
        # random targets in a loop.
"""
        for _ in range(5):
            targets = [uniform(rmin, rmax) for (rmin, rmax) in ranges]
            print('move stages...')

            # The GCS commands qTMN() and qTMX() used above are query commands.
            # They don't need an argument and will then return all availabe
            # information, e.g. the limits for _all_ axes. With setter commands
            # however you have to specify the axes/channels. GCSDevice provides
            # a property "axes" which returns the names of all connected axes.
            # So lets move our stages...

            pidevice.MOV(pidevice.axes, targets)

            # To check the on target state of an axis there is the GCS command
            # qONT(). But it is more convenient to just call "waitontarget".

            pitools.waitontarget(pidevice)

            # GCS commands usually can be called with single arguments, with
            # lists as arguments or with a dictionary.
            # If a query command is called with an argument the keys in the
            # returned dictionary resemble the arguments. If it is called
            # without an argument the keys are always strings.

            positions = pidevice.qPOS(pidevice.axes)
            for axis in pidevice.axes:
                print('position of axis {} = {:.2f}'.format(axis, positions[axis]))

            # positions = pidevice.qPOS()
            # for axis in positions:
            #    print('position of axis {} = {.2f}'.format(axis, positions[axis]))

        print('done')
        pidevice.CloseConnection()



gcs = GCSDevice('C-884')
gcs.InterfaceSetupDlg()
print (gcs.qIDN())
gcs.CloseConnection()
"""