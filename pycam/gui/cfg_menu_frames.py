# -*- coding: utf-8 -*-

"""
Configuration file for instantiating menu frames which need to be instantiated prior to the display of the frames.
E.g. the data held within these objects links to the processing work, and needs to be present whether or not the
object's frame has been build in the menu
"""

from pycam.gui.figures_analysis import GeomSettings, ProcessSettings, PlumeBackground, DOASFOVSearchFrame, \
    CellCalibFrame, CrossCorrelationSettings, OptiFlowSettings, LightDilutionSettings
from pycam.cfg import pyplis_worker
from pycam.gui.figures_doas import CalibrationWindow
from pycam.gui.cfg import gui_setts, current_dir_img, current_dir_spec, sock, ftp_client
from pycam.gui.acquisition import BasicAcqHandler
from pycam.gui.network import InstrumentConfiguration

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

# Cell calibration frame
cell_calib = CellCalibFrame(fig_setts=gui_setts, process_setts=process_settings)

# Cross-correlation frame
cross_correlation = CrossCorrelationSettings(fig_setts=gui_setts)

# Optical flow frame
opti_flow = OptiFlowSettings(fig_setts=gui_setts)

# Light dilution frame
light_dilution = LightDilutionSettings(fig_setts=gui_setts)

# Communications handler
basic_acq_handler = BasicAcqHandler(pyplis_worker, img_dir=current_dir_img, spec_dir=current_dir_spec)

# Instrument configuration
instrument_cfg = InstrumentConfiguration(ftp_client)