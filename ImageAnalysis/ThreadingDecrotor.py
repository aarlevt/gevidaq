# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:14:01 2020

@author: xinmeng
"""

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication
import sys
import threading
import time

def run_in_thread(fn):
    """
    https://stackoverflow.com/questions/23944657/typeerror-method-takes-1-positional-argument-but-2-were-given
    """
    # @staticmethod
    def run(*args, **kwargs):
        
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        print('run_in_thread start')
        return thread # <-- return the thread
    return run

class test(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = QVBoxLayout() 
        self.aabuttom = QPushButton('Top')
        self.layout.addWidget(self.aabuttom)
        self.aabuttom.clicked.connect(lambda:self.run_in_thread(self.call))
        
        self.num = 1
        print(self.num)
        self.setLayout(self.layout) 
                
    
    def run_in_thread(self, fn, *args, **kwargs):
        """
        Send target function to thread.
        Usage: lambda: self.run_in_thread(self.fn)
        
        Parameters
        ----------
        fn : function
            Target function to put in thread.

        Returns
        -------
        thread : TYPE
            Threading handle.

        """
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        print('run_in_thread start')
        
        return thread
        
    # @run_in_thread
    def call(self):
        for i in range(10):
            self.num+=1
            print(self.num)
            time.sleep(2)

        
if __name__ == "__main__":
    def run_app():
        app = QApplication(sys.argv)
        mainwin = test()
        mainwin.show()
        app.exec_()
    run_app()  