# -*- coding: utf-8 -*-

"""
Configuration file for instantiating menu frames which need to be instantiated prior to the display of the frames.
E.g. the data held within these objects links to the processing work, and needs to be present whether or not the
object's frame has been build in the menu
"""

from pycam.gui.figures_analysis import GeomSettings, ProcessSettings, PlumeBackground, DOASFOVSearchFrame
from pycam.gui.figures_doas import CalibrationWindow
from pycam.gui.cfg import gui_setts

# Geometry settings
geom_settings = GeomSettings()

# DOAS calibration window
calibration_wind = CalibrationWindow(fig_setts=gui_setts)

# Processing settings
process_settings = ProcessSettings()

# Plume background settings
plume_bg = PlumeBackground()

# DOAS FOV search frame
doas_fov = DOASFOVSearchFrame(fig_setts=gui_setts)