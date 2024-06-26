# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:02:08 2018

@author: lhuismans
"""

import logging
import time

import serial


class LudlStage:
    """
    This class is for controlling the stage using the MAC5000 from Ludl. So far
    this only works using a serial to usb connection and having the converting
    piece provided by Teun and Greta in between.
    """

    def __init__(self, address):
        """
        Initializing the stage, Ludl recommends that the user refrain from use
        of the joystick or RS-232 serial communication until completion of
        interface initialization routine which is approximately 5 seconds after
        power up.
        The COM port has to be specified as a string: "COM#".
        """
        self.address = address
        self.baudrate = 9600
        self.endOfLine = "\r"

    def Try_until_Success(func):
        """This is the decorator to try to execute the function until succeed."""

        def wrapper(*args, **kwargs):
            success = None
            failnumber = 0

            while success is None:
                if failnumber < 8:
                    try:
                        returnValue = func(*args, **kwargs)
                        success = True

                    except Exception as exc:
                        logging.critical("caught exception", exc_info=exc)
                        failnumber += 1
                        logging.info(
                            "Stage move failed, failnumber: {}".format(
                                failnumber
                            )
                        )
                        time.sleep(0.2)
                else:
                    logging.info("Fail for 8 times, give up - -")
                    success = False

            return returnValue

        return wrapper

    @Try_until_Success
    def getPos(self):
        """
        Gets the current position of the stage. A positive reply is of the from:
            :A X Y
        A negative reply is of the form:
            :N -code
        See the documentation to find which code corresponds to which error.
        Returns the x and y position
        """
        command = "Where X Y" + self.endOfLine

        with serial.Serial(
            self.address,
            self.baudrate,
            stopbits=serial.STOPBITS_TWO,
            timeout=1,
        ) as stage:
            stage.write(command.encode())
            position = stage.readline()  # Reads an '\n' terminated line
            position = position.decode().split(
                " "
            )  # Go from bytes to string and split

            if len(position) > 2:
                xPosition = position[1]
                yPosition = position[2]
                stage.close()

                return xPosition, yPosition
            else:
                return False, False

    @Try_until_Success
    def moveAbs(self, x, y):
        """
        Moves the motor to an absolute position defined by X and Y.
        A positive reply is sent back when the command is received correctly.
        The reply does not mean the movement is finished.

        Note: After sending the command, the execution seems to be running in backend then.
        """
        command = "Move X = %d Y = %d" % (x, y) + self.endOfLine

        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())
            # stage.flush()
            stage.close()

            return True

    @Try_until_Success
    def moveVec(self, x, y):
        """
        Moves the motor to an absolute position but as opposed to moveAbs it
        adjusts the speeds to plot a straight line.
        """
        command = "Vmove X = %d Y = %d" % (x, y) + self.endOfLine
        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())

    @Try_until_Success
    def moveRel(self, xRel=None, yRel=None):
        """
        Moves the motor to a relative position.
        Set position to None if this axis is not used.
        """
        if xRel is None:
            command = "Movrel Y = %d" % yRel + self.endOfLine
        elif yRel is None:
            command = "Movrel X = %d" % xRel + self.endOfLine
        else:
            command = "Movrel X = %d Y = %d" % (xRel, yRel) + self.endOfLine

        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())
            stage.flush()
            stage.close()

            return True

    @Try_until_Success
    def home(self):
        """
        Moves to motors toward the end limit switch. This is reached by running
        the motor to a large negative position.
        If there are no errors a positive reply is sent back.
        """
        command = "Home" + self.endOfLine
        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())

    @Try_until_Success
    def setZero(self):
        """
        Sets the current position as zero position.
        """
        command = "Here X = 0 Y = 0" + self.endOfLine
        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())

    @Try_until_Success
    def joystick(self, on=True):
        """
        Enable or disable the joystick.
        """
        switch = "+"
        if on is False:
            switch = "-"

        command = "Joystick X%s Y%s" % (switch, switch) + self.endOfLine
        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())

    @Try_until_Success
    def motorsStopped(self):
        """
        Checks if the motors are still running.
        Reply:
            "N" if all motors are stopped. Returns True
            "B" if one or more motors are is still running. Returns False
        """
        command = "Status" + self.endOfLine
        with serial.Serial(
            self.address,
            self.baudrate,
            stopbits=serial.STOPBITS_TWO,
            timeout=1,
        ) as stage:
            stage.write(command.encode())
            status = stage.readline()
            status = status.decode()  # Go from bytes to string
            if status == "N":
                return True
            return False

    @Try_until_Success
    def reset(self):
        """
        Resets the whole controller and restart from a power up condition. No
        reply is provided.
        """
        command = "REMRES" + self.endOfLine
        with serial.Serial(
            self.address, self.baudrate, stopbits=serial.STOPBITS_TWO
        ) as stage:
            stage.write(command.encode())
            time.sleep(5)  # Sleep for 5 seconds to initialize.


if __name__ == "__main__":
    ludlStage = LudlStage("COM12")

    step = 1500
    for i in range(10):  # Repeat twice
        logging.info("start moving..")
        ludlStage.moveAbs(i * step, i * step)

        time.sleep(1)
