# -*- coding: utf-8 -*-

# PycamUV
"""Setup for pyplis usage, controlling filter initiation etc
Scripts are an edited version of the pyplis example scripts, adapted for use with the PiCam"""
from __future__ import (absolute_import, division)

import pyplis

# ## SCRIPT FUNCTION DEFINITIONS
def create_picam_new_filters():
    # Start with creating an empty Camera object
    cam = pyplis.setupclasses.Camera()

    # Specify the camera filter setup

    # Create on and off band filters. Obligatory parameters are "type" and
    # "acronym", "type" specifies the filter type ("on" or
    # "off"), "acronym" specifies how to identify this filter in the file
    # name. If "id" is unspecified it will be equal to "type". The parameter
    # "meas_type_acro" is only important if a measurement type (e.g. M -> meas,
    # C -> calib ...) is explicitely specified in the file name.
    # This is not the case for ECII camera but for the HD camera,
    # see specifications in file cam_info.txt for more info.

    on_band = pyplis.utils.Filter(id="on", type="on", acronym="fltrA",
                                  meas_type_acro="F01", center_wavelength=310)
    off_band = pyplis.utils.Filter(id='off', type="off", acronym="fltrB",
                                   center_wavelength=330)

    # put the two filter into a list and assign to the camera
    filters = [on_band, off_band]

    cam.default_filters = filters
    cam.prepare_filter_setup()

    # Similar to the filter setup, access info for dark and offset images needs
    # to be specified. The ECII typically records 4 different dark images, two
    # recorded at shortest exposure time -> offset signal predominant, one at
    # low and one at high read gain. The other two recorded at longest possible
    # exposure time -> dark current predominant, also at low and high read gain

    offset_low_gain = pyplis.utils.DarkOffsetInfo(id="offset0", type="offset",
                                                  acronym="D0L", read_gain=0)
    offset_high_gain = pyplis.utils.DarkOffsetInfo(id="offset1", type="offset",
                                                   acronym="D0H", read_gain=1)
    dark_low_gain = pyplis.utils.DarkOffsetInfo(id="dark0", type="dark",
                                                acronym="D1L", read_gain=0)
    dark_high_gain = pyplis.utils.DarkOffsetInfo(id="dark1", type="dark",
                                                 acronym="D1H", read_gain=1)

    # put the 4 dark info objects into a list and assign to the camera
    dark_info = [offset_low_gain, offset_high_gain,
                 dark_low_gain, dark_high_gain]

    cam.dark_info = dark_info

    # Now specify further information about the camera

    # camera ID (needs to be unique, i.e. not included in data base, call
    # pyplis.inout.get_all_valid_cam_ids() to check existing IDs)
    cam.cam_id = "picam-1"

    # image file type
    cam.file_type = "fts"

    # File name delimiter for information extraction
    cam.delim = "_"

    # position of acquisition time (and date) string in file name after
    # splitting with delimiter
    cam.time_info_pos = 1

    # datetime string conversion of acq. time string in file name
    cam.time_info_str = "%Y-%m-%dT%H%M%S"

    # position of image filter type acronym in filename
    cam.filter_id_pos = 2

    # position of meas type info
    cam.meas_type_pos = 5

    # Define which dark correction type to use
    # 1: determine a dark image based on image exposure time using a dark img
    # (with long exposure -> dark current predominant) and a dark image with
    # shortest possible exposure (-> detector offset predominant). For more
    # info see function model_dark_image in processing.py module
    # 2: subtraction of a dark image recorded at same exposure time than the
    # actual image
    cam.darkcorr_opt = 2

    # If the file name also includes the exposure time, this can be specified
    # here:
    cam.texp_pos = 4  # the ECII does not...

    # the unit of the exposure time (choose from "s" or "ms")
    cam.texp_unit = "ms"

    # define the main filter of the camera (this is only important for cameras
    # which include, e.g. several on band filters.). The ID need to be one of
    # the filter IDs specified above
    cam.main_filter_id = "on"

    # camera focal length can be specified here (but does not need to be, in
    # case of the ECII cam, there is no "default" focal length, so this is left
    # empty)
    cam.focal_length = ""

    # Detector geometry
    # Using PiCam binned values (binning 2592 x 1944 to 648 x 486)
    cam.pix_height = 5.6e-6  # pixel height in m
    cam.pix_width = 5.6e-6  # pixel width in m
    cam.pixnum_x = 684
    cam.pixnum_y = 486

    cam._init_access_substring_info()

    cam.io_opts = dict(USE_ALL_FILES=False,
                       SEPARATE_FILTERS=True,
                       INCLUDE_SUB_DIRS=True,
                       LINK_OFF_TO_ON=True)

    # Set the custom image import method
    cam.image_import_method = pyplis.custom_image_import.load_ecII_fits
    # That's it...
    return cam
