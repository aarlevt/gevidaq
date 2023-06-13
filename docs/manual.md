Brinks Lab
Octoscope Software Manual
02/02/2021






OVERVIEW
This software is developed to facilitate lab activities in Brinks lab. It combines octoscope hardware control,  flexible routine data acquisition, machine learning cell segmentation, and data analysis. 







ICONS
Across all graphical user interfaces (GUI), the same set of button icons is used to keep consistency. The icons and their general purposes are listed below.
Connect/Disconnect
The toggle button establishes or closes the connection with a  particular device. 
If fail to connect, a warning window will pop up:
 Upon connection, the button will change to a disconnect button:
 Click again will disconnect the device.
Add
		Add previously generated waveform/scanning routine to the assembly.
Delete
Delete selected waveform/scanning routine from the assembly.

Configure
		Assemble and organize all the configured information, last step before execution.
Browse
		Browse and feed in folder/file location through pop-up file explorer.
Save
		Save the generated data/pipeline.
Execute
		Run the configured waveform/pipeline.
Clear
		Clear the current configuration.
Snap
		Snap an image.
Motor translation
		For motors (translation stage, objective motor), move to the specified position.
Arrow
		Relative movement of a motor, or going through a stack.
Shutter control
		The toggle button which opens or closes the laser shutter.
LED control
		The checkable button which turns on or off the white LED light.




  



WIDGETS
The main GUI is the assembly of individually developed widgets, granting flexibility. Here each widget is introduced.

Focus control widget

The objective sits on a holder mounted on a Physik Instrumente (PI) M-110.1DG DC gear motor. This widget can communicate with the motor so that the objective focus can be controlled. 


The connect/disconnect toggle button establishes or turns off the connection with the motor. Click to connect first before sending commands.

To move to an absolute position, first input the target position in millimeters:

Then hit the move to defined position button:

To move relatively, type in the step size of each movement:

Then hit the arrow buttons to move up or down:

For coarse movement, one can drag the slider, or click the arrow button at both ends of this slider:

The current position of the motor will show here:


Sample stage widget

This widget controls the MAC5000 sample stage from Ludl. The sample is clamped on the stage and moved in horizontal directions.


To move to an absolute position, input the stage x and y coordinate:
And then click the move button to move to the defined position :

To move relatively, type in the step size in stage coordinates:
Then click the arrow buttons to move in  horizontal directions:
After movement, the current position will show up in 

AOTF control widget

The acousto-optical tunable filter in the setup gives temporal control over different laser lines. While the binary blanking for all lines can be controlled through TTL triggers, the amplitude of each laser line is tuned through analog signals. Physical shutters are also added to each laser line.


The binary blanking gives binary on/off control over all laser lines, and it can be toggled through the button on top:
The analog amplitude control of each laser line (from top to bottom: red, green, and blue) lies in individual slider scaling from 0 to 5, one can drag the slider to make a quick change. The precise value will show in the text box beside the slider with two decimals. A specific value can also be typed into the text box and press the “Enter” key on the keyboard to set the value.




The physical shutters driven by servo motors can be controlled through the checkable buttons on the right side:


Two-photon laser widget

The two-photon laser widget controls the Insight X3 pulsed laser in the setup. It communicates with the laser through a serial port.




The widget is called from the main panel:


It will then try to query the laser status and show it in the container:

The laser has two operation modes: running mode and alignment mode. MODE RUN sets the laser to standard operating mode. MODE ALIGN sets the laser to low power so that it is easier and safer to align the optical beam through a microscope. The switch is a toggle button:

The beam shutter can be opened and closed through:
 

The wavelength can be set between 680 and 1300 nm:

To turn on/off the pump laser:

NOTE: The shutter is not automatically opened when the ON command is issued. Turning off the pump diode laser here will maintain the oven temperatures for a quick warm-up time. To turn off the laser system entirely, refer to the SHUTDOWN command.
To set the number of seconds for the software watchdog timer (A value of 0 disables the software watchdog timer): 

When closing the widget, a window will pop up:

There are two ways here to leave the laser:

Hibernate — This is the default day-to-day operating mode. It shuts off the laser diode, closes the shutters, and saves the wavelength and motor positions.  Using Hibernate DOES NOT shut down the laser's internal computer operations so that the laser can be restarted without the delay of the internal computers rebooting. Because Hibernate does not shut down the internal computers, the power supply MUST BE LEFT ON when using the Hibernate mode. DO NOT TURN OFF THE POWER!

Standby — It closes the shutters and disables the watchdog timer so that the laser does not stop when the GUI is closed. It also saves the last set wavelength and motor positions, and it leaves all other components powered up and operating. Choose this when there are following laser operations to save the warmup time.

Seal test widget

The patch clamp seal test widget facilitates patch clamping experiments. It controls the NI-DAQ to generate electrical signals for the patch clamp amplifier, and receive and display recording signals in the meantime. 











For later calculation, the gain settings of the physical knobs on the patch amplifier need to be input into the “Gains” container:

The sealtest voltage step can be adjusted:

To start and stop seal test:

The recorded current will be displayed in real-time:

Based on the feedback, the resistance and capacitance that the probe electrode feels are calculated and shown below: 

To keep track of patch quality, the current displayed plot can be saved together with the input pipette resistance.
To measure the membrane voltage in current clamp mode, go to the Vm measurement container:  
In case one finds it hard to break in the cell after a giga seal, the zap function is there to help. The amplitude and width of a voltage pulse can be tuned: 

Waveform widget

The waveform widget enables easy configuration of National Instruments data acquisition (NI-DAQ) device commands. Different devices in Octoscope can be synchronized by receiving these analog and digital signals from NI-DAQ. The recording channels can also be configured and run at the same sampling rate.


The general settings need to be set first:
The reference waveform determines the length of all other waveforms in one single assembly, meaning that waveforms shorter will be padded with 0 and longer will be cut to the same length:
The recording container contains checkboxes that will add corresponding parallel NI-DAQ recording channels: 
The sampling rate textbox sets the sampling rate when running the waveforms. The master clock combo box selects the master clock source and can be the internal clock of a NI-DAQ or Hamamatsu camera. Previously generated waveform files can also be loaded through the “Load waveform” button to repeat experiments:


The analog signals container allows the configuration of analog control signals: 
Select the channel from the combo box you want to configure first:
Then the waveform can be shaped by filling the text boxes below, different modes on each tab:
Block waves:
Frequency: the absolute frequency of the square steps.
Duration: the duration of the waveform in one cycle in milliseconds.
Duty cycle: the duty cycle of each period.
Offset: the padding zero in milliseconds of each waveform before start.
Starting amplitude: the initial value in voltage of the waveform.
Baseline: the baseline value in voltage of the waveform.
Steps: the number of increasing steps in the duration of one cycle.
Change per step: the elevation in voltage of each step.
Number of cycles: the repeat number of each cycle.
Gap between cycles: the gap in milliseconds between cycles.
Ramp waves:
Frequency: the absolute frequency of the ramps.
Duration: the duration of the waveform in one cycle in milisecond.
Symmetry: the symmetry degree of the ramp, being symmetric when it is 0.5.
Height: the relative peak value compared to baseline.
Baseline: the baseline value in voltage of the waveform.
Galvo:
There are two ways here to configure the scanning routine of the galvanometer scanners: raster scanning and contour scanning. In raster scanning mode, the scanning area is rectangular, while in contour scanning mode, the scanners navigate the focus along a predefined routine.
In raster scanning mode:
voltXMin/voltXMax: the minimal/maximal  scanning voltage in X direction.
voltYMin/voltYMax: the minimal/maximal scanning voltage in Y direction.
X/Y pixel number: the sampling points or pixel number in the final reconstructed image.
Offset: the padding zero in milliseconds of each waveform before start.
Gap between scans: the gap in milliseconds between scans.
average over: number of acquired images for averaging.
Repeat: number of repeats of the whole waveform.
In contour scanning mode:
Duration: the period in milliseconds to repeat the pre-configured contour scanning.

The digital signals container allows the configuration of digital control signals, which work similarly to the analog one: 
All the waveforms generated can be inspected visually in the plotting window at the bottom.

After all waveforms are generated, there are last steps before executing them:
If the ‘Save waveforms’ checkbox is checked, a file will be automatically generated in the same folder for later use.
In the emission filter container, one can choose the emission filter used for the acquisition if necessary. The emission filter will switch to the configured one before running the waveforms.
Click the configure button will organize all the waveforms and get ready for execution.
Then the execute button will light up, allowing execution.
The clean canvas button allows the reset of waveform configuration.
After execution, the progress bar will show the execution progress:

Two-photon imaging widget

Two-photon imaging widget is to inspect and acquire two-photon fluorescence images of the sample. The voltage signals which define the galvo mirrors’ scanning route can be configured and the reconstructed image will be displayed in this widget. Raster scanning can be done to image a rectangular area, after which an arbitrary path can be drawn along the contour to do contour scanning.



For the continuous raster scanning, all the parameters can be configured in the “Continuous scanning” tab on the top right of the panel:

The sampling rate defines the sampling rate of the galvo signals as well as the PMT acquisition: 
The voltage range determines the galvo scanning voltage range, i.e., the field of view size:
The pixel number will affect the pixel size of the final image:
To decrease noise in the final image, one can average over multiple images by increasing the average number:
The run and stop buttons can start and stop the scanning:
The displayed image can be saved by clicking the save button:

To do an automatic z-axis scanning, go to the tab “Stack scanning”:
 
Same as continuous scanning, imaging parameters, i.e., sampling rate, voltage range, image pixel number, and average number, are configured in the spinboxes:

The z-axis step size can be set in millimeters:

The total scanning depth is also set in millimeters:

The start and stop buttons are at the bottom :

For the contour scanning, place the handles along the contour to define the trajectory. Handles can be added by left click on lines, or removed by right click on the handle.

There are two ways to distribute the points along the lines: 
Non-uniform: Equal number of points are placed between handles. The denser the handles, the more samples are placed in the region.
Uniform: Regardless of how handles are distributed, generate points with uniform distance along the contour.
This can be selected in the comb box, on the left of which there is a text line showing the number of handles. 
The total number of contour points in route and the sampling rate of generated signal can be input below.
Hit the configure button     would generate the scanning waveforms, and then one can start continuous contour scanning , or use it in the Waveform widget:


Camera widget

The camera widget controls the Hamamatsu camera through the provided DCAM-API (Digital CAMera Application Programming Interface, https://dcam-api.com/). At the backend it uses Python Ctypes library to allow calling functions in DCAM DLLs, and is adapted from this Github repository: https://github.com/ZhuangLab/storm-control.git. The frontend is designed to make it more user-friendly, and allows interaction with other widgets.


The “General settings” container contains all the hardware setting information. 

The connection status is shown besides the connection toggle button. The camera needs to be switched on before running the widget otherwise it will not recognize the camera.
In the image format container, the binning and pixel bit depth can be set:
The binning setting allows integration of charges from adjacent pixels, improving signal-to-noise ratio (SNR) and possible frame rate, with the price of lower resolution.
The pixel bit depth in units of bit determines the bit depth of the digitized image.


There are two readout speed options for the camera: The standard scan readout speed can achieve a frame rate of 100 fps for full resolution with low noise (1.0 electrons (median), 1.6 electrons (r.m.s.)), and the slow scan readout speed can achieve even lower noise (0.8 electrons (median), 1.4 electrons (r.m.s.)) with a frame rate of 30 fps for full resolution. One can switch in between these two modes:
There are a few pixels in CMOS image sensor that have slightly higher readout noise performance compared to surrounding pixels.And the extended exposures may cause a few white spots which is caused by failure in part of the silicon wafer in CMOS image sensor. The camera has real-time variant pixel correction features to improve image quality, it can be enabled/disabled here:
The camera exposure time setting can be configured in textbox by the units of seconds:

The second tab in general settings is about setting the region of interest (ROI):
Depending on the size of the field of view, there are two region readout modes: sub-array readout is a procedure where only a region of interest is scanned; while full size images the whole frame. It can be toggled quickly through the switch:
To narrow down to smaller ROI, a ROI selector can be first shown in camera live view and visually checked:
Another way is to put in the exact offsets and size:
To reach the fastest frame rate possible, the ROI needs to be centered to maximise the framerate of the Hamamatsu CMOS. The ROI selector can be automatically centered by clicking the button:
Before applying the ROI, one can inspect it first by clicking “Check ROI” button:
The ROI region will show below in acquisition container, and can switch back to normal again by unchecking the “Check ROI” button:

The ROI can be applied by clicking the apply ROI button. 
The trigger mode of the camera can be configured in the “Timing” container:
 
By default the camera is driven by its internal clock.
It can also run by external triggers. click “External” and then choose from the combobox the wanted external triggering configuration: edge, level and syncreadout. For the details of these modes please refer to the Hamamatsu manual.
The timing can also be set to Master pulse mode, in which the camera generates pulses that are independent of the exposure or readout timing of the image sensor. The master pulse can be set as a reference signal of the programmable timing output, so it is possible to set up a synchronous system with peripheral devices without an external pulse generator.
The acquisition container contains all the operation options needed for data acquisition. 

The top part shows important parameters like internal frame rate, current exposure time and readout speed:

The saving directory can be selected and file name can be typed in the textbox.
For real-time live, one can start it by switch the toggle button, and then the live image will show in the right image view:

To snap an image: 
To save current live/snap image: 
The contrast of displayed live image can be automatically adjusted by checking the checkable button:

To record a video, one can go to the “Stream” tab:

There are two ways to configure the duration of the video recording: by setting the total number of frames or the time. It can be switched through the combobox:

The spinboxes on the right calculate the RAM space to allocate in number of frames, and the number will show up in buffer size spinbox:

For now the images can only be streamed to RAM.
After the configuration, click “Apply” to enable record start button:
 
The stream progress will be displayed below:

