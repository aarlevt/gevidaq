# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:16:23 2019

@author: xinmeng
"""
from random import uniform

from pipython import GCSDevice, pitools

CONTROLLERNAME = 'C-863.11'
STAGES = ('RS40 DC',)  # connect stages to axes
REFMODE = ('FNL',)  # reference the connected stages

# CONTROLLERNAME = 'C-884.DB'  # 'C-884' will also work
# STAGES = ('M-111.1DG', 'M-111.1DG', 'NOSTAGE', 'NOSTAGE')
# REFMODE = ('FNL', 'FNL')

# CONTROLLERNAME = 'E-709'
# STAGES = None  # this controller does not need a 'stages' setting
# REFMODE = None


def main():
    """Connect to a PIPython device."""

    # We recommend to use GCSDevice as context manager with "with".

    with GCSDevice() as pidevice:
        # InterfaceSetupDlg() is an interactive dialog. There are other methods to
        # connect to an interface without user interaction.

        # pidevice.InterfaceSetupDlg(key='sample')
        # pidevice.ConnectRS232(comport=1, baudrate=115200)
        pidevice.ConnectUSB(serialnum='USB SN 0185500641')
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


if __name__ == '__main__':
    # To see what is going on in the background you can remove the following
    # two hashtags. Then debug messages are shown. This can be helpful if
    # there are any issues.

    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    main()
