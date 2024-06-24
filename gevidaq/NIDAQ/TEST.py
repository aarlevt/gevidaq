# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:30:39 2024

@author: LocalAdmin
"""


import matplotlib.pyplot as plt
import numpy as np
import nidaqmx

# Constants for the DAQ device and channel configuration
device_name = 'Dev1'  # Replace with your device name if different
channels = ['ai20', 'ai1']  # Analog input channels to read from

# Configure DAQ tasks for analog input
with nidaqmx.Task() as task1, nidaqmx.Task() as task2:
    task1.ai_channels.add_ai_voltage_chan(f'{device_name}/{channels[0]}')
    task2.ai_channels.add_ai_voltage_chan(f'{device_name}/{channels[1]}')
    
    # Define number of samples to read and sampling rate 
    num_samples = 1000  # Number of samples to read
    sampling_rate = 1000.0  # Sampling rate in Hz 

    # Configure timing (optional)
    task1.timing.cfg_samp_clk_timing(sampling_rate)
    task2.timing.cfg_samp_clk_timing(sampling_rate)

    # Read analog input data
    data1 = task1.read(number_of_samples_per_channel=num_samples)
    data2 = task2.read(number_of_samples_per_channel=num_samples)
    timestamps = np.arange(num_samples) / sampling_rate  # Generate timestamps for plotting

# Plotting the received voltage data in subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(timestamps, data1, marker='o', linestyle='-', color='b')
ax1.set_title(f'Analog Input Voltage {channels[0]} vs Time')
ax1.set_ylabel('Voltage (V)')
ax1.grid(True)

ax2.plot(timestamps, data2, marker='o', linestyle='-', color='r')
ax2.set_title(f'Analog Input Voltage {channels[1]} vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (V)')
ax2.grid(True)

plt.tight_layout()
plt.show()