# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 10:45:37 2022

@author: TvdrBurgt
"""

import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QDoubleSpinBox, QGroupBox, QLabel


from PatchClamp.pressurethread import PressureThread


class PressureControllerUI(QWidget):
    def __init__(self):
        super().__init__()
        """
        =======================================================================
        ----------------------------- Start of GUI ----------------------------
        =======================================================================
        """
        """
        # ---------------------- General widget settings ---------------------
        """
        self.setWindowTitle("Pressure Control")

        """
        -------------------------- Settings container -------------------------
        """
        pressureContainer = QGroupBox(title="Pressure control")
        pressureLayout = QGridLayout()

        # Labels set and read pressure
        pressure_set_label = QLabel("Request pressure:")
        pressure_read_label = QLabel("Readout pressure:")
        pressure_units_label1 = QLabel("mBar")
        pressure_units_label2 = QLabel("mBar")

        # Spinbox to set pressure
        self.set_pressure_spinbox = QDoubleSpinBox(self)
        self.set_pressure_spinbox.setMinimum(-300)
        self.set_pressure_spinbox.setMaximum(200)
        self.set_pressure_spinbox.setDecimals(0)
        self.set_pressure_spinbox.setValue(0)
        self.set_pressure_spinbox.setSingleStep(10)

        # Label pressure readout value
        self.pressure_value_Label = QLabel("ATM")
        self.pressure_value_Label.setFont(QFont("Times", weight=QFont.Bold))

        # Label pressure status
        pressure_statustext_label = QLabel("Status:")
        self.pressure_status_label = QLabel("Idle")
        self.pressure_status_label.setFont(QFont("Times", weight=QFont.Bold))

        pressureLayout.addWidget(pressure_set_label, 0, 0, 1, 2)
        pressureLayout.addWidget(self.set_pressure_spinbox, 0, 2, 1, 1)
        pressureLayout.addWidget(pressure_units_label1, 0, 3, 1, 1)
        pressureLayout.addWidget(pressure_read_label, 1, 0, 1, 2)
        pressureLayout.addWidget(self.pressure_value_Label, 1, 2, 1, 1)
        pressureLayout.addWidget(pressure_units_label2, 1, 3, 1, 1)
        pressureLayout.addWidget(pressure_statustext_label, 2, 0, 1, 2)
        pressureLayout.addWidget(self.pressure_status_label, 2, 2, 1, 2)
        pressureContainer.setLayout(pressureLayout)

        """
        -------------------------- Buttons container --------------------------
        """
        buttonsContainer = QGroupBox()
        buttonsLayout = QGridLayout()

        # Buttons for relatively valued pressures
        low_positive_button = QPushButton(text="Low +", clicked=self.request_low_positive)
        low_negative_button = QPushButton(text="Low -", clicked=self.request_low_negative)
        high_positive_button = QPushButton(text="High +", clicked=self.request_high_positive)
        high_negative_button = QPushButton(text="High -", clicked=self.request_high_negative)
        stop_regulating_button = QPushButton(text="Off", clicked=self.request_stop_regulating)
        atmoshpere_button = QPushButton(text="Atm", clicked=self.request_atmosphere)

        # Buttons for operating modes
        confirm_button = QPushButton(text="Confirm", clicked=self.request_pressure)
        spike_button = QPushButton(text="Generate pulse", clicked=self.request_spike)
        self.toggle_lcd_button = QPushButton(text="LCD dark", clicked=self.toggle_lcd)
        self.toggle_lcd_button.setCheckable(True)

        buttonsLayout.addWidget(low_positive_button, 0, 0, 1, 1)
        buttonsLayout.addWidget(low_negative_button, 0, 1, 1, 1)
        buttonsLayout.addWidget(high_positive_button, 1, 0, 1, 1)
        buttonsLayout.addWidget(high_negative_button, 1, 1, 1, 1)
        buttonsLayout.addWidget(stop_regulating_button, 2, 0, 1, 1)
        buttonsLayout.addWidget(atmoshpere_button, 2, 1, 1, 1)
        buttonsLayout.addWidget(confirm_button, 0, 2, 1, 2)
        buttonsLayout.addWidget(spike_button, 1, 2, 1, 2)
        buttonsLayout.addWidget(self.toggle_lcd_button, 2, 2, 1, 2)
        buttonsContainer.setLayout(buttonsLayout)

        """
        ---------------------- Add widgets and set Layout ---------------------
        """
        layout = QGridLayout()
        layout.addWidget(pressureContainer, 0, 0, 1, 1)
        layout.addWidget(buttonsContainer, 1, 0, 1, 1)
        self.setLayout(layout)

        """
        =======================================================================
        -------------- Start up backend and connect signals/slots--------------
        =======================================================================
        """
        self.pressurethread = PressureThread(pressurecontroller_handle=None)
        self.pressurethread.measurement.connect(self.update_pressure_value)
        self.pressurethread.start()

        """
        =======================================================================
        ----------------------------- End of GUI ------------------------------
        =======================================================================
        """


    def request_low_positive(self):
        pressure = 30
        self.pressurethread.pressurecontroller.setPres(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_low_negative(self):
        pressure = -30
        self.pressurethread.pressurecontroller.setPres(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_high_positive(self):
        pressure = 100
        self.pressurethread.pressurecontroller.setPres(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_high_negative(self):
        pressure = -100
        self.pressurethread.pressurecontroller.setPres(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_atmosphere(self):
        pressure = 0
        self.pressurethread.pressurecontroller.setPres(pressure)
        self.pressure_status_label.setText("Stand-by")

    def request_stop_regulating(self):
        pressure = self.set_pressure_spinbox.value()
        self.pressurethread.pressurecontroller.setPresHold(pressure)
        self.pressure_status_label.setText("Not regulating")

    def request_pressure(self):
        pressure = self.set_pressure_spinbox.value()
        self.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_spike(self):
        pressure = self.set_pressure_spinbox.value()
        self.pressurethread.set_pulse_stop_waveform(pressure)
        self.pressure_status_label.setText("Applying pulse")

    def toggle_lcd(self):
        if self.toggle_lcd_button.isChecked():
            self.pressurethread.pressurecontroller.LCDoff()
        else:
            self.pressurethread.pressurecontroller.LCDon()


    def update_pressure_value(self, data):
        value = str(round(data[0]))
        self.pressure_value_Label.setText(value)


    def closeEvent(self, event):
        """ Close event
        This method is called when the GUI is shut down. It releases the python
        kernel, stops the pressurethread, and closes the COM-port.
        """
        # Stopping the pressurethread (also closes the connection)
        self.pressurethread.stop()

        event.accept()

        # Frees the console by quitting the application entirely
        QtWidgets.QApplication.quit()




if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = PressureControllerUI()
        mainwin.show()
        app.exec_()

    run_app()
