# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:54:51 2020

@author: xinmeng
"""

# =============================================================================
# -------------------------Auto focus workflow---------------------------------
# --1. Find local minima range. (one direction until value getting smaller)
# --2. Bisection to find the optimal.
# =============================================================================

import os
# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname+'/../')

from PI_ObjectiveMotor.focuser import PIMotor
from ImageAnalysis.ImageProcessing import ProcessImage
from GalvoWidget.GalvoScan_backend import RasterScan
import time
import matplotlib.pyplot as plt

class FocusFinder():
    
    def __init__(self, source_of_image = "PMT", init_step_size = 0.008, total_step_number = 5, motor_handle = None, twophoton_handle = None, *args, **kwargs):
        """
        

        Parameters
        ----------
        source_of_image : string, optional
            The input source of image. The default is PMT.
        init_step_size : int, optional
            The step size when first doing coarse searching. The default is 0.010.
        step_number : int, optional
            Number of steps in total to find optimal focus. The default is 5.
        motor_handle : TYPE, optional
            Handle to control PI motor. The default is None.
        twophoton_handle : TYPE, optional
            Handle to control Insight X3. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        
        # The step size when first doing coarse searching.
        self.init_step_size = init_step_size
        
        # Number of steps in total to find optimal focus.
        self.total_step_number = total_step_number
        
        if motor_handle == None:
            # Connect the objective if the handle is not provided.
            self.pi_device_instance = PIMotor()
        else:
            self.pi_device_instance = motor_handle
        
        # Current position of the focus.
        self.current_pos = self.pi_device_instance.GetCurrentPos()
        
        # Threshold for focus-degree
        self.focus_degree_thres = 0.00008

        # Number of steps already tried.
        self.steps_taken = 0
        # The focus degree of previous position.
        self.previous_degree_of_focus = 0
        # Number of going backwards.
        self.turning_point = 0
        # The input source of image.
        self.source_of_image = source_of_image
        if source_of_image == "PMT":
            self.galvo = RasterScan(Daq_sample_rate = 500000, edge_volt = 3)
        
    def bisection(self):
        # The upper edge in which we run bisection.
        upper_position = self.current_pos + self.init_step_size
        # The lower edge in which we run bisection.
        lower_position = self.current_pos - self.init_step_size

        for step_index in range(1, self.total_step_number + 1):   
            # In each step of bisection finding.
            
            # In the first round, get degree of focus at three positions.
            if step_index == 1:
                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position)/2
                degree_of_focus_mid = self.evaluate_focus(mid_position)
                print("mid focus degree: {}".format(round(degree_of_focus_mid, 5)))
                
                # Break the loop if focus degree is below threshold which means
                # that there's no cell in image.
                if degree_of_focus_mid <= self.focus_degree_thres:
                    mid_position = [False, self.current_pos]
                    break

                # Move to top and evaluate.
                degree_of_focus_up = self.evaluate_focus(obj_position = upper_position)
                print("top focus degree: {}".format(round(degree_of_focus_up, 5)))
                # Move to bottom and evaluate.
                degree_of_focus_low = self.evaluate_focus(obj_position = lower_position)
                print("bot focus degree: {}".format(round(degree_of_focus_low, 5)))
                # Sorting dicitonary of degrees in ascending.
                biesection_range_dic = {"top":[upper_position, degree_of_focus_up], 
                                        "bot":[lower_position, degree_of_focus_low]}
                
            # In the next rounds, only need to go to center and update boundaries.
            elif step_index > 1:
                # The upper edge in which we run bisection.
                upper_position = biesection_range_dic["top"][0]
                # The lower edge in which we run bisection.
                lower_position = biesection_range_dic["bot"][0]
                
                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position)/2
                degree_of_focus_mid = self.evaluate_focus(mid_position)
                
                print("Current focus degree: {}".format(round(degree_of_focus_mid, 5)))
                
            # If sits in upper half, make the middle values new bottom.
            if biesection_range_dic["top"][1] > biesection_range_dic["bot"][1]:
                biesection_range_dic["bot"] = [mid_position, degree_of_focus_mid]
            else:
                biesection_range_dic["top"] = [mid_position, degree_of_focus_mid]
            
            print("The upper pos: {}; The lower: {}".format(biesection_range_dic["top"][0], biesection_range_dic["bot"][0]))
            
        return mid_position
                
                
    
    def evaluate_focus(self, obj_position = None):
        """
        Evaluate the focus degree of certain objective position.

        Parameters
        ----------
        obj_position : float, optional
            The target objective position. The default is None.

        Returns
        -------
        degree_of_focus : float
            Degree of focus.

        """
        
        if obj_position != None:
            self.pi_device_instance.move(obj_position)
            
        # Get the image.
        if self.source_of_image == "PMT":
            image = self.galvo.run()
            plt.figure()
            plt.imshow(image)
            plt.show()
        degree_of_focus = ProcessImage.local_entropy(image)
        time.sleep(0.2)
        
        return degree_of_focus
    
            
    # def explore(self, image):
        
    #     # Calculate the focus degree.
    #     self.current_degree_of_focus = ProcessImage.variance_of_laplacian(image)
        
    #     # If a turning point is met (except first attempt at wrong direction),
    #     # starts to bisection.
    #     if self.turning_point >= 1 and self.steps_taken != 1:
    #         self.steps_taken_after_turning = 
            
    #     self.init_step_size *= (1/2) ** (self.steps_taken_after_turning)
        
    #     # if focus degree increases, move one step forwards.
    #     if self.current_degree_of_focus > self.previous_degree_of_focus:
    #         PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + self.init_step_size)
            
    #     else: # else move downwards.
        
    #         self.init_step_size *= -1 
            
    #         # If the first attempt goes towards the wrong direction, 
    #         # turn around and move one step the other way.
    #         if self.steps_taken == 1 and self.turning_point == 1:
    #             # Move two step downwards.
    #             PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + 2 * self.init_step_size)
    #             # Clean up trace, make sure the correct condition for first attempt.
    #             self.steps_taken = 0
    #             self.turning_point = 0
            
    #         else: # In normal bisection situation.
    #             PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + self.init_step_size)
    #             # Add one turning point.
    #             self.turning_point += 1
            
    #     self.current_pos = self.pi_device_instance.GetCurrentPos()
    #     # Cast the current focus degree to previous for next round.
    #     self.previous_degree_of_focus = self.current_degree_of_focus


    #     # Update total number of steps.
    #     self.steps_taken += 1
if __name__ == "__main__":
    ins = FocusFinder()
    ins.bisection() # will return false if there's no cell in view.
    ins.pi_device_instance.CloseMotorConnection()