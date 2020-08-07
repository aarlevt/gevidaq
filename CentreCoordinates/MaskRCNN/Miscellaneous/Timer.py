# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:28:17 2020

@author: meppenga
"""
import time
class Timer():
    def __init__(self):
        """Timer class.
        Author: Martijn Eppenga
        With this class you can create a timer object.
        Methods:
            tic:       start timer or restart timer
            toc:       get elapsed time in seconds as string
            toc_s:     get elapsed time in seconds as int
            toc_ms:    get elapsed time in miliseconds as int
            toc_mcs:   get elapsed time in microseconds as int
            toc__ns:   get elapsed time in nanoseconds as int
            totalTime: get total elapsed time since class instanciation as string
            CreateTimeString: (Time) Create time string from input time"""
        self.StartTime    = time.time()
        self.StartTime_ns = time.time_ns()
        self.TimerStart   = time.time()
        
    def tic(self):
        """Start or restart timer"""
        self.TimerStart   = time.time()
        self.StartTime_ns = time.time_ns()
        
    def toc(self):
        """Return elapsed time in seconds as a string"""
        return self.CreateTimeString(time.time() - self.TimerStart)
    
    def toc_s(self):
        """Return elapsed time in seconds as integer"""
        return int(time.time() - self.TimerStart)
    
    def toc_ms(self):
        """Return elapsed time in miliseconds as integer"""
        return int((time.time_ns() - self.StartTime_ns) * 1e-6)
    
    def toc_mcs(self):
        """Return elapsed time in microseconds as integer"""
        return int((time.time_ns() - self.StartTime_ns) * 1e-3)
    
    def toc_ns(self):
        """Return elapsed time in nanoseconds as integer"""
        return int(time.time_ns() - self.StartTime_ns)
     
    def totalTime(self):
        """Return total time elapsed since instanciation of timer class
        in seconds"""
        return self.CreateTimeString(time.time() - self.StartTime)
    
    def CreateTimeString(self,Time):
        """Return formated time string
        Input:
            Time, time in seconds to be formated
        Return:
            str, foramted time string. Format: %d day: %d hr: %d min: %d s"""
        day         = Time // (3600*24)
        Time        = Time % (3600*24)
        hour        = Time // 3600
        Time        = Time % 3600
        minutes     = Time // 60
        seconds     = (Time % 60) // 1
        return '%d day: %d hr: %d min: %d s'%(day, hour, minutes, seconds)