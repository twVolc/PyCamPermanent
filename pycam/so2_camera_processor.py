# -*- coding: utf-8 -*-

# PycamUV
"""Setup for pyplis usage, controlling filter initiation etc
Scripts are an edited version of the pyplis example scripts, adapted for use with the PiCam"""
from __future__ import (absolute_import, division)

from pycam.setupclasses import CameraSpecs, SpecSpecs
from pycam.utils import make_circular_mask_line

import pyplis
from pyplis import LineOnImage
from pyplis.custom_image_import import load_picam_png
from pyplis.helpers import make_circular_mask
from pyplis.optimisation import PolySurfaceFit
from pyplis.plumespeed import OptflowFarneback, LocalPlumeProperties
from pyplis.dilutioncorr import DilutionCorr, correct_img
import pydoas

from tkinter import filedialog, messagebox
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import transform as tf
import warnings
from math import log10, floor
import datetime
import time
import queue
import threading

class PyplisWorker:
    """
    Main pyplis worker class
    :param  img_dir:    str     Directory where images are stored
    :param  cam_specs:  CameraSpecs     Object containing all details of the camera/images
    :param  spec_specs:  SpecSpecs      Object containing all details of the spectrometer/spectra
    """
    def __init__(self, img_dir=None, cam_specs=CameraSpecs(), spec_specs=SpecSpecs()):
        self.q = queue.Queue()      # Queue object for images. Images are passed in a pair for fltrA and fltrB
        self.q_doas = queue.Queue()     # Queue where processed doas values are placed (dictionary containing al relevant data)

        self.cam_specs = cam_specs  #
        self.spec_specs = spec_specs

        self.wait_time = 0.2

        # Setup memory allocation for images (need to keep a large number for DOAS fov image search).
        self.img_tau_buff_size = 200         # Buffer size for images (this number of images are held in memory)
        self.img_tau_buff = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, self.img_tau_buff_size],
                                 dtype=np.float32)
        self.img_tau_buff_time = [None] * self.img_tau_buff_size   # Names of images held within img_buff (on-band name held)
        self.idx_current = 0    # Used to track what the current index is for saving to image buffer

        self.doas_buff_size = 500
        self.column_densities = np.zeros(self.doas_buff_size, dtype=np.float32)    # Column densities array
        self.std_errs = np.zeros(self.doas_buff_size, dtype=np.float32)    # Array of standard error on column denisties
        self.cd_times = [None] * self.doas_buff_size                             # Times of column densities data points
        self.idx_current_doas = 0       # Current index for doas spectra

        test_doas_start = self.get_img_time('2018-03-26T144404')
        self.test_doas_times = [test_doas_start + datetime.timedelta(seconds=x) for x in range(0, 600, 4)]
        self.test_doas_cds = np.random.rand(len(self.test_doas_times)) * 1000
        self.test_doas_stds = np.random.rand(len(self.test_doas_times)) * 50

        # Pyplis object setup
        self.load_img_func = pyplis.custom_image_import.load_picam_png
        self.cam = create_picam_new_filters({})         # Generate pyplis-picam object
        self.meas = pyplis.setupclasses.MeasSetup()     # Pyplis MeasSetup object (instantiated empty)
        self.img_reg = ImageRegistration()              # Image registration object
        self.cell_calib = pyplis.cellcalib.CellCalibEngine(self.cam)
        self.plume_bg = pyplis.plumebackground.PlumeBackgroundModel()
        self.plume_bg.surface_fit_pyrlevel = 0
        self.plume_bg.mode = 4      # Plume background mode - default (4) is linear in x and y

        self.BG_CORR_MODES = [0,    # 2D poly surface fit (without sky radiance image)
                              1,    # Scaling of sky radiance image
                              2,
                              3,
                              4,    # Scaling + linear gradient correction in x & y direction
                              5,
                              6,    # Scaling + quadr. gradient correction in x & y direction
                              99]
        self.auto_param_bg = True   # Whether the line parameters for BG modelling are generated automatically
        self.ref_check_lower = 0
        self.ref_check_upper = 0    # Background check to ensure no gas is present in ref region
        self.ref_check_mode = True
        self.polyfit_2d_mask_thresh = 100
        self.PCS_lines = []
        self.maxrad_doas = self.spec_specs.fov * 1.1        # Max radius used for doas FOV search (degrees)
        self.opt_flow = OptflowFarneback()
        self.tau_thresh = 0.01          # Threshold used for generating pixel mask
        self.light_dil_lines = []
        self.ambient_roi = [0, 0, 0, 0]    # Ambient intensity ROI coordinates for light dilution
        self.I0_MIN = 0     # Minimum intensity for dilution fit
        self.ext_off = None     # Extinction coefficient for off-band
        self.ext_on = None      # Extinction coefficient for on-band
        self.got_light_dil = False  # Flags whether we have light dilution for this sequence
        self.lightcorr_A = None     # Light dilution corrected image
        self.lightcorr_B = None     # Light dilution corrected image

        # Figure objects (objects are defined elsewhere in PyCam. They are not matplotlib Figure objects, although
        # they will contain matplotlib figure objects as attributes
        self.fig_A = None               # Figure displaying off-band raw image
        self.fig_B = None               # Figure displaying off-band raw image
        self.fig_tau = None             # Figure displaying absorbance image
        self.fig_bg_A = None            # Figure displaying modelled background of on-band
        self.fig_bg_B = None            # Figure displaying modelled background of off-band
        self.fig_bg_ref = None          # Figure displaying ?
        self.fig_spec = None            # Figure displaying spectra
        self.fig_doas = None            # Figure displaying DOAS fit
        self.fig_doas_fov = None        # Figure for displaying DOAS FOV on correlation image
        self.fig_opt = None             # Figure for displaying optical flow
        self.fig_dilution = None        # Figure for displaying light dilution
        self.fig_cell_cal = None        # Figure for displaying cell calibration - CellCalibFrame obj
        self.calib_pears = None         # Pyplis object holding functions to plot results
        self.doas_fov_x = None          # X FOV of DOAS (from pyplis results)
        self.doas_fov_y = None          # Y FOV of DOAS
        self.doas_fov_extent = None     # DOAS FOV radius

        self.img_dir = img_dir
        self.dark_dict = {'on': {},
                          'off': {}}     # Dictionary containing all retrieved dark images with their ss as the key
        self.dark_dir = None
        self.img_list = None
        self.num_img_pairs = 0      # Total number of plume pairs
        self.num_img_tot = 0        # Total number of plume images
        self.first_image = True     # Flag for when in the first image of a sequence (wind speed can't be calculated)
        self._location = None       # String for location e.g. lascar
        self.source = None          # Pyplis object of location

        self.img_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.img_B = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.img_A_prev = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.img_B_prev = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.img_aa = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])     # Optical depth image
        self.img_cal = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])     # SO2 calibrated image
        self.img_tau = pyplis.image.Img()
        self.img_tau.img = self.img_A           # Set image to zeors as it may be used to generate empty figure
        self.img_tau_prev = pyplis.image.Img()
        self.img_tau_prev.img = self.img_A

        # Calibration attributes
        self.got_cal_doas = False
        self.got_cal_cell = False
        self._cell_cal_dir = None
        self.cal_type = 1       # Calibration method: 0 = Cell, 1= DOAS, 2 = Cell and DOAS (cell used to adjust FOV sensitivity)
        self.cell_dict_A = {}
        self.cell_dict_B = {}
        self.cell_tau_dict = {}     # Dictionary holds optical depth images for each cell
        self.cell_masks = {}        # Masks for each cell to adjust for sensitivity changes over FOV
        self.sensitivity_mask = np.ones([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])  # Mask of lowest cell ppmm - one to use for correcting all tau images.
        self.sens_mask_ppmm = None  # Cell ppmm value for that used for generating sensitivity mask
        self.cal_crop = False       # Bool defining if calibration region is cropped
        self.cal_crop_region = None # Bool Array defining crop region for calibration
        self.cal_crop_line_mask = None  # Bool array of the line for crop region (used to find mean of area to set everything outside to that value
        self.crop_sens_mask = False     # Whether cal crop region is used to crop the sensitivity mask
        self.cell_cal_vals = np.zeros(2)
        self.cell_fit = None        # The cal scalar will be [0] of this array
        self.cell_pol = None
        self.use_sensitivity_mask = True  # If true, the sensitivty mask will be used to correct tau images
        self.use_cell_bg = False    # If true, the bg image for bg modelling is automatically set from the cell calibration directory. Otherwise the defined path to bg imag is used

        # Load background image if we are provided with one
        self.bg_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vign_A = np.ones([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vigncorr_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.bg_A_path = None

        # Load background image if we are provided with one
        self.bg_B = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vign_B = np.ones([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vigncorr_B = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.bg_B_path = None

        self.img_A_q = queue.Queue()      # Queue for placing images once loaded, so they can be accessed by the GUI
        self.img_B_q = queue.Queue()      # Queue for placing images once loaded, so they can be accessed by the GUI

        self.plot_iter = True   # Bool defining if plotting iteratively is active. If it is, all images are passed to qs

        self.process_thread = None  # Placeholder for threading object

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value.lower()      # Make all lower case for ease of handling

        # Get currently existing source ids
        ids = pyplis.inout.get_source_ids()

        # Make all ids lower case
        ids_lower = [f.lower() for f in ids]

        # If the ID is not in the list, we get it from online
        if self.location not in ids_lower:
            source = pyplis.inout.get_source_info_online(self.location)

            # If download was successful the returned dictionary will have contain a name for the source
            if self.location in source.keys():
                pyplis.inout.save_default_source(source[self.location])
            else:
                raise UnrecognisedSourceError

            # Set location id by extracting the name from the oredered dictionary
            location_id = source[self.location]['name']

        else:
            # If the id already exists we extract the id using the lower case id list
            location_id = ids[ids_lower.index(self.location)]

        # Load the source
        self.source = pyplis.setupclasses.Source(location_id)

    @property
    def cell_cal_dir(self):
        return self._cell_cal_dir

    @cell_cal_dir.setter
    def cell_cal_dir(self, value):
        """When the cell calibration directory is changed we automatically load it in and process the data"""
        self._cell_cal_dir = value
        self.perform_cell_calibration(plot=False, set_bg_img=False)

    def update_cam_geom(self, geom_info):
        """Updates camera geometry info by creating a new object and updating MeasSetup object

        Parameters
        ----------
        geom_info: dict
            Dictionary containing geometry info for pyplis"""
        self.cam = create_picam_new_filters(geom_info)

    def measurement_setup(self, img_dir=None, start=None, stop=None, location=None, wind_info=None):
        """Creates pyplis MeasSetup object"""
        # Check the camera is correctly setup
        if not isinstance(self.cam, pyplis.setupclasses.Camera):
            print('Pyplis camera object not correctly setup, cannot create MeasSetup object')
            return

        if img_dir is not None:
            self.img_dir = img_dir

        if location is not None and location != self.location:
            self.location = location

        # Setup measurement object
        self.meas = pyplis.setupclasses.MeasSetup(self.img_dir, start, stop, camera=self.cam,
                                                  source=self.source, wind_info=wind_info)

    def show_geom(self):
        """Wrapper for pyplis plotting of measurement geometry"""
        self.geom_fig = self.meas.meas_geometry.draw_map_2d()
        self.geom_fig.fig.show()

    def get_img_time(self, filename):
        """
        Gets time from filename and converts it to datetime object
        :param filename:
        :return img_time:
        """
        # Make sure filename only contains file and not larger pathname
        filename = filename.split('\\')[-1].split('/')[-1]

        # Extract time string from filename
        time_str = filename.split('_')[self.cam_specs.file_date_loc]

        # Turn time string into datetime object
        img_time = datetime.datetime.strptime(time_str, self.cam_specs.file_datestr)

        return img_time

    def get_img_list(self):
        """
        Gets image list and splits it into image pairs (on/off), it flags if there are missing pairs
        :returns img_list: list
        """
        # Create full list of images
        full_list = [f for f in os.listdir(self.img_dir)
                     if self.cam_specs.file_img_type['meas'] in f and self.cam_specs.file_ext in f]

        self.num_img_tot = len(full_list)

        # Get A and B image lists (sort img_list_A as it will be the one we loop through, so needs to be in time order)
        img_list_A = [f for f in full_list if self.cam_specs.file_filterids['on'] in f]
        img_list_A.sort()
        img_list_B = [f for f in full_list if self.cam_specs.file_filterids['off'] in f]

        no_contemp = 0
        img_list = []

        # Loop through images in img_list_A and find contemporaneous img_B if it exists
        for img_A in img_list_A:
            timestr_A = img_A.split('_')[self.cam_specs.file_date_loc]
            img_B = [f for f in img_list_B if timestr_A in f]

            # If no contemporaneous image was found we flag it
            if len(img_B) == 0:
                no_contemp += 1
                continue
            elif len(img_B) > 1:
                warnings.warn('Multiple contemporaenous images found for {} in image directory {}\n'
                              'Selecting the first image as the pair'.format(img_A, self.img_dir))

            # Append the image pair to img_list, if we have a pair
            img_list.append([img_A, img_B[0]])

        if no_contemp > 0:
            warnings.warn('Image sequence has {} incomplete pairs\n'
                          'These images will not be used for processing.'.format(no_contemp))

        self.num_img_pairs = len(img_list)

        return img_list

    def reset_self(self):
        """
        Resets aspects of self to ensure we start processing in the correct manner
        :return:
        """
        self.img_tau_buff = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, self.img_tau_buff_size],
                                     dtype=np.float32)
        self.img_tau_buff_time = [None] * self.img_tau_buff_size  # Names of images held within img_buff (on-band name held)
        self.idx_current = 0  # Used to track what the current index is for saving to image buffer
        self.idx_current_doas = 0   # Used for tracking current index of doas points
        self.got_cal_doas = False
        self.got_cal_cell = False
        self.got_light_dil = False

    def load_sequence(self, img_dir=None, plot=True, plot_bg=True):
        """
        Loads image sequence which is defined by the user
        :param init_dir:
        :return:
        """
        if img_dir is None:
            img_dir = filedialog.askdirectory(title='Select image sequence directory', initialdir=self.img_dir)

        if len(img_dir) > 0 and os.path.exists(img_dir):
            self.img_dir = img_dir
        else:
            raise ValueError('Image directory not recognised: {}'.format(img_dir))

        # Update first_image flag
        self.first_image = True

        # Reset buffers as we have a new sequence
        self.reset_self()

        # Update image list
        self.img_list = self.get_img_list()

        # Display first images of sequence
        if len(self.img_list) > 0:
            self.process_pair(self.img_dir + '\\' + self.img_list[0][0],
                              self.img_dir + '\\' + self.img_list[0][1],
                              plot=plot, plot_bg=plot_bg)

            if len(self.img_list) > 1:
                # Load second image too so that we have optical flow output generated
                self.first_image = False
                self.process_pair(self.img_dir + '\\' + self.img_list[1][0],
                                  self.img_dir + '\\' + self.img_list[1][1],
                                  plot=plot, plot_bg=plot_bg)


    def load_BG_img(self, bg_path, band='A'):
        """Loads in background file

        :param bg_path: Path to background sky image
        :param band: Defines whether image is for on or off band (A or B)"""
        if not os.path.exists(bg_path):
            raise ValueError('File path specified for background image does not exist: {}'.format(bg_path))
        if band not in ['A', 'B']:
            raise ValueError('Unrecognised band for background image: {}. Must be either A or B.'.format(band))

        # Create image object
        img = pyplis.image.Img(bg_path, self.load_img_func)

        # Dark subtraction - first extract ss then hunt for dark image
        ss = str(int(img.texp * 10 ** 6))
        dark_img = self.find_dark_img(self.dark_dir, ss, band=band)

        if dark_img is not None:
            img.subtract_dark_image(dark_img)
        else:
            warnings.warn('No dark image provided for background image.\n '
                          'Background image has not been corrected for dark current.')

        # Set variables
        setattr(self, 'bg_{}'.format(band), img)
        self.generate_vign_mask(img.img, band)
        setattr(self, 'bg_{}_path'.format(band), bg_path)

    def load_img(self, img_path, band=None, plot=True):
        """
        Loads in new image and dark corrects if a dark file is provided

        :param img_path: str    Path to image
        :param band: str    If band is not provided, it will be found from the pathname
        :param plot: bool   If true, the image is added to the img_q so that it will be displayed in the gui
        """
        # Extract band if it isn't already provided
        if band is None:
            band = [f for f in img_path.split('_') if 'fltr' in f][0].replace('fltr', '')

        # Set previous image from current img
        setattr(self, 'img_{}_prev'.format(band), getattr(self, 'img_{}'.format(band)))

        # Get new image
        img = pyplis.image.Img(img_path, self.load_img_func)
        img.filename = img_path.split('\\')[-1].split('/')[-1]
        img.pathname = img_path

        # Dark subtraction - first extract ss then hunt for dark image
        ss = str(int(img.texp * 10 ** 6))
        dark_img = self.find_dark_img(self.dark_dir, ss, band=band)

        if dark_img is not None:
            img.subtract_dark_image(dark_img)
        else:
            warnings.warn('No dark image found, image has been loaded without dark subtraction')

        # Set object attribute to the loaded pyplis image
        # (must be done prior to image registration as the function uses object attribute self.img_B)
        setattr(self, 'img_{}'.format(band), img)

        # Warp image using current setup if it is B
        if band == 'B':
            self.register_image()

        # Add to plot queue if requested
        if plot:
            getattr(self, 'fig_{}'.format(band)).update_plot(img_path)

    def find_dark_img(self, img_dir, ss, band='on'):
        """
        Searches for suitable dark image in designated directory. First it filters the images for the correct filter,
        then searches for an image with the same shutter speed defined
        :param: ss  int,str     Shutter speed value to hunt for. Can be either int or str
        :return: dark_img
        """
        # If band is given in terms of A/B we convert to on/off
        if band == 'A':
            band = 'on'
        elif band == 'B':
            band = 'off'

        # Round ss to 2 significant figures
        ss_rounded = round(int(ss), -int(floor(log10(abs(int(ss))))) + 1)

        # Fast dictionary look up for preloaded dark images (using rounded ss value)
        if str(ss_rounded) in self.dark_dict[band].keys():
            dark_img = self.dark_dict[band][str(ss_rounded)]
            return dark_img

        # List all dark images in directory
        dark_list = [f for f in os.listdir(img_dir)
                     if self.cam_specs.file_img_type['dark'] in f and self.cam_specs.file_ext in f
                     and self.cam_specs.file_filterids[band] in f]

        # Extract ss from each image and round to 2 significant figures
        ss_str = self.cam_specs.file_ss.replace('{}', '')
        ss_images = [int(f.split('_')[self.cam_specs.file_ss_loc].replace(ss_str, '')) for f in dark_list]
        ss_rounded_list = [round(f, -int(floor(log10(abs(f)))) + 1) for f in ss_images]

        ss_idx = [i for i, x in enumerate(ss_rounded_list) if x == ss_rounded]
        ss_images = [dark_list[i] for i in ss_idx]

        if len(ss_images) < 1:
            # messagebox.showerror('No dark images found', 'No dark images at a shutter speed of {} '
            #                                              'were found in the directory {}'.format(ss, img_dir))
            return None

        # If we have images, we loop through them to create a coadded image
        dark_full = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, len(ss_images)])
        for i, ss_image in enumerate(ss_images):
            # Load image. Coadd.
            dark_full[:, :, i], meta = pyplis.custom_image_import.load_picam_png(img_dir + ss_image)

        # Coadd images to creat single image
        dark_img = np.mean(dark_full, axis=2)

        # Update lookup dictionary for fast retrieval of dark image later
        self.dark_dict[band][str(ss_rounded)] = dark_img

        return dark_img

    def register_image(self, **kwargs):
        """
        Registers B image to A by passing them to the ImageRegistration object
        kwargs: settings to be passed to image registartion object
        :return:
        """
        self.img_B.img_warped = self.img_reg.register_image(self.img_A.img, self.img_B.img, **kwargs)

    def update_img_buff(self, img_tau, filename):
        """
        Updates the image buffer and file time buffer
        :param img_tau:     np.array        n x m image matrix of tau image
        :param filname:     str             on- or off-band filename for image used to generate img_tau
        """
        # Extract time from filename into datetime object
        img_time = self.get_img_time(filename)

        # If we haven't exceeded buffer size then we simply add new data to buffer
        if self.idx_current < self.img_tau_buff_size:
            self.img_tau_buff[:, :, self.idx_current] = img_tau
            self.img_tau_buff_time[self.idx_current] = img_time

        # If we are beyond the buffer size we need to shift all images down one and add the new image to the end
        # The oldest image is therefore lost from the buffer
        else:
            self.img_tau_buff[:, :, :-1] = self.img_tau_buff[:, :, 1:]
            self.img_tau_buff[:, :, -1] = img_tau
            self.img_tau_buff_time[:-1] = self.img_tau_buff_time[1:]
            self.img_tau_buff_time[-1] = img_time

        # Increment current index
        self.idx_current += 1

    def generate_vign_mask(self, img, band):
        """
        Generates vign mask and updates self.vign_X from the imag and for the specified band X
        :param img:     np.array    Clear-sky image to be converted to vign_mask
        :param band:    str         Band
        :return:
        """
        if band.lower() == 'on':
            band = 'A'
        elif band.lower() == 'off':
            band = 'B'

        setattr(self, 'vign_{}'.format(band), img / np.amax(img))

    def perform_cell_calibration(self, plot=True, set_bg_img=True):
        """
        Loads in cell calibration images and performs the calibration so that it is ready if needed
        :param plot:        bool    States whether the results are plotted
        :param set_bg_img:  bool    States whether bg image should be set to coadded clears loaded herein
        :return:
        """
        # Create updated cell calib engine (includes current cam geometry - may not be necessary)
        self.cell_calib = pyplis.cellcalib.CellCalibEngine(self.cam)

        img_list_full = [x for x in os.listdir(self.cell_cal_dir) if self.cam_specs.file_ext in x]

        # Clear sky
        clear_list_A = [x for x in img_list_full
                        if self.cam_specs.file_filterids['on'] in x  and self.cam_specs.file_img_type['clear'] in x]
        clear_list_B = [x for x in img_list_full
                        if self.cam_specs.file_filterids['off'] in x and self.cam_specs.file_img_type['clear'] in x]
        num_clear_A = len(clear_list_A)
        num_clear_B = len(clear_list_B)
        img_array_clear_A = np.zeros((self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, num_clear_A), dtype=np.float32)
        img_array_clear_B = np.zeros((self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, num_clear_B), dtype=np.float32)

        if num_clear_A == 0 or num_clear_B == 0:
            print('No clear images present. Ensure the calibration directory contains clear images for both filters')
            return

        # Loop through clear images and load them into buffer
        for i in range(num_clear_A):
            img_array_clear_A[:, :, i], meta = load_picam_png(os.path.join(self.cell_cal_dir, clear_list_A[i]))

            # Find associated dark image by extracting shutter speed, then subtract this image
            ss = meta['texp'] / self.cam_specs.file_ss_units
            img_array_clear_A[:, :, i] -= self.find_dark_img(self.dark_dir, ss=ss, band='A')

            # Scale image to 1 second exposure (so that we can deal with images of different shutter speeds
            img_array_clear_A[:, :, i] *= (1 / meta['texp'])

        # Coadd images to create single clear image A
        img_clear_A = np.mean(img_array_clear_A, axis=2)    #  Co-added final clear image

        # As above but with off-band images
        for i in range(num_clear_B):
            img_array_clear_B[:, :, i], meta = load_picam_png(os.path.join(self.cell_cal_dir, clear_list_B[i]))

            # Find associated dark image by extracting shutter speed, then subtract this image
            ss = meta['texp'] / self.cam_specs.file_ss_units
            img_array_clear_B[:, :, i] -= self.find_dark_img(self.dark_dir, ss=ss, band='B')

            # Scale image to 1 second exposure (so that we can deal with images of different shutter speeds
            img_array_clear_B[:, :, i] *= (1 / meta['texp'])

        # Coadd images to create single clear image B
        img_clear_B = np.mean(img_array_clear_B, axis=2)  # Co-added final clear image

        self.mask_A = self.generate_vign_mask(img_clear_A, 'A')
        self.mask_B = self.generate_vign_mask(img_clear_B, 'B')

        # If requested, we update bg images to those from calibration, rather than explicitly defined bg images
        if set_bg_img:
            self.bg_A = pyplis.image.Img(img_clear_A)
            self.bg_B = pyplis.image.Img(img_clear_B)
            self.bg_A_path = 'Coadded. ...{}'.format(self.cell_cal_dir[-8])
            self.bg_B_path = 'Coadded. ...{}'.format(self.cell_cal_dir[-8])

        # -------------------------------
        # READ IN CALIBRATION CELL IMAGES
        # -------------------------------
        # Calibration file listssky
        cal_list_A = [x for x in img_list_full
                      if self.cam_specs.file_filterids['on'] in x and self.cam_specs.file_img_type['cal'] in x]
        cal_list_B = [x for x in img_list_full
                      if self.cam_specs.file_filterids['off'] in x and self.cam_specs.file_img_type['cal'] in x]
        num_cal_A = len(cal_list_A)
        num_cal_B = len(cal_list_B)

        if num_cal_A == 0 or num_cal_B == 0:
            print('Calibration directory does not contain expected image. Aborting calibration load!')
            return

        cell_vals_A = [x.split('.')[0].split('_')[self.cam_specs.file_type_loc].replace(self.cam_specs.file_img_type['cal'], '')
                       for x in cal_list_A]
        cell_vals_B = [x.split('.')[0].split('_')[self.cam_specs.file_type_loc].replace(self.cam_specs.file_img_type['cal'], '')
                       for x in cal_list_B]
        cell_vals = list(set(cell_vals_A))

        # Ensure that we only calibrate with cells which have both on and off band images. So first we determine values
        # which are in one list but not the other, removing them from cell_vals
        cell_vals_B = list(set(cell_vals_B))
        missing_vals = np.setdiff1d(cell_vals, cell_vals_B, assume_unique=True)
        for val in missing_vals:
            try:
                cell_vals.remove(val)
            except ValueError:
                pass
            print('Cell {}ppmm is not present in both on- and off-band images, so is not being processed.')

        # Reset cell image dictionary as we are loading a new folder (don't want old cells/values in there)
        self.cell_dict_A = {}
        self.cell_dict_B = {}

        for ppmm in cell_vals:
            # Set id for this cell (based on its filename)
            cal_id = ppmm + self.cam_specs.file_img_type['cal']

            for band in ['A', 'B']:
                # Make list for specific calibration cell
                cell_list = [x for x in locals()['cal_list_{}'.format(band)] if cal_id in x]
                num_img = len(cell_list)

                # Create empty array
                cell_array = np.zeros((self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, num_img),
                                       dtype=np.float32)

                for i in range(num_img):
                    cell_array[:, :, i], meta = load_picam_png(os.path.join(self.cell_cal_dir, cell_list[i]))

                    # Find associated dark image by extracting shutter speed, then subtract this image
                    ss = meta['texp'] / self.cam_specs.file_ss_units
                    cell_array[:, :, i] -= self.find_dark_img(self.dark_dir, ss=ss, band=band)

                    # Scale image to 1 second exposure (so that we can deal with images of different shutter speeds
                    cell_array[:, :, i] *= (1 / meta['texp'])

                # Coadd images to create single clear image B
                cell_img_coadd = np.mean(cell_array, axis=2)  # Co-added final clear image

                # Put image in dictionary
                getattr(self, 'cell_dict_{}'.format(band))[ppmm] = cell_img_coadd.copy()

        # GENERATE OPTICAL DEPTHS FOR EACH CELL
        num_cells = len(self.cell_dict_A)
        self.cell_cal_vals = np.zeros([num_cells, 2])
        self.cell_tau_dict = {}
        self.cell_masks = {}

        # Loop thorugh each cell calculating optical depth
        for i, ppmm in enumerate(self.cell_dict_A):
            # Set ppmm value
            self.cell_cal_vals[i, 0] = int(ppmm)

            # Generate absorbance image
            with np.errstate(divide='ignore', invalid='ignore'):
                self.cell_tau_dict[ppmm] = -np.log10(np.divide(np.divide(self.cell_dict_A[ppmm], img_clear_A),
                                                     np.divide(self.cell_dict_B[ppmm], img_clear_B)))
            self.cell_tau_dict[ppmm][np.isneginf(self.cell_tau_dict[ppmm])] = 0
            self.cell_tau_dict[ppmm][np.isinf(self.cell_tau_dict[ppmm])] = 0
            self.cell_tau_dict[ppmm][np.isnan(self.cell_tau_dict[ppmm])] = 0

            # Generate mask for this cell - if calibrating with just cell we use centre of image, otherwise we use
            # DOAS FOV for normalisation region

            if self.cal_type in [1, 2] and self.got_cal_doas:
                self.cell_masks[ppmm] = self.generate_sensitivity_mask(self.cell_tau_dict[ppmm],
                                                                       pos_x=self.doas_fov_x, pos_y=self.doas_fov_y,
                                                                       radius=self.doas_fov_extent, pyr_lvl=2)
            else:
                self.cell_masks[ppmm] = self.generate_sensitivity_mask(self.cell_tau_dict[ppmm], radius=3, pyr_lvl=2)

            # Correct cell images for sensitivity using mask
            self.cell_tau_dict[ppmm] = self.cell_tau_dict[ppmm] / self.cell_masks[ppmm].img

            # Finally calculate average cell optical depth
            if not self.cal_crop:
                self.cell_cal_vals[i, 1] = np.mean(self.cell_tau_dict[ppmm])
            else:
                self.cell_cal_vals[i, 1] = np.mean(self.cell_tau_dict[ppmm][self.cal_crop_region])

        # Use 2nd smallest cell for sensitvity mask (don't want to stray into non-linearity, but don't want 0 cell)
        ppmms = self.cell_cal_vals[:, 0].copy()
        ppmms.sort()
        self.sens_mask_ppmm = str(int(ppmms[1]))
        self.sensitivity_mask = self.cell_masks[self.sens_mask_ppmm].img

        # Perform linear fit (tau is x variable, so that self.cell_pol can be used directly to extract ppmm from tau)
        self.cell_fit = np.polyfit(self.cell_cal_vals[:, 1], self.cell_cal_vals[:, 0], 1)
        self.cell_pol = np.poly1d(self.cell_fit)

        # Flag that we now have a cell calibration
        self.got_cal_cell = True

        # Plot calibration
        if plot:
            self.fig_cell_cal.update_plot()

    def generate_sensitivity_mask(self, img_tau, pos_x=None, pos_y=None, radius=1, pyr_lvl=2):
        # TODO check pyr_lvl is working correctly, and pos_x/y and radius are scaled correctly following pyr
        """
        Generates mask which optical depth images are divided by to correct for sensitivity changes due to filter
        tranmission shifts with viewing angle change.

        Taken from pyplis.cellcalib.CellCalibEngine.get_sensitivity_corr_mask(), breaks it out to allow passing tau
        image and the centre point - easier use here as it doesn't require full setup of CellCalibEngine, which I don't
        want to use - it's a little clunky for my use.

        :param img_tau: np.array
            Optical depth image of cell whihc is used to generate the sensitivity mask
        pos_x : int
            x-pixel position of normalisation mask, if None the image center
            position is used (which is also the default pixel used to retrieve
            the vector of calibration optical densities from the cell OD
            images)
        pos_y : int
            y-pixel position of normalisation mask, if None the image center
            position is used (which is also the default pixel used to retrieve
            the vector of calibration optical densities from the cell OD
            images)
        radius : int
            radius specifying the disk size around ``pos_x_abs`` and
            ``pos_y_abs`` used to normalise the mask (i.e. uses average OD of
            cell image in this OD)
        :param pyr_lvl: int
            Pyramid level for downscaling of polynomial surface fit
        :return mask:   pyplis.image.Img
            Mask which OD images should be divided by to correct for changes in SO2 sensitivity
        """
        # If pos_x_abs or pos_y_abs is None, we use the centre position of the image for the mask normalisation
        if pos_x is None or pos_y is None:
            pos_x, pos_y = int(self.cam_specs.pix_num_x / 2.0), int(self.cam_specs.pix_num_y / 2.0)

        # Generate the mask for the central area for OD normalisation
        fov_mask = make_circular_mask(self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, pos_x, pos_y, radius)

        # Fit 2D model to tau image
        try:
            # This returns an array with 2 too many rows, so take from second to second last
            cell_img = PolySurfaceFit(img_tau, pyrlevel=pyr_lvl).model[1:-1, :]
        except:
            warnings.warn("2D polyfit failed while determination of sensitivity "
                 "correction mask, using original cell tau image for mask "
                 "determination")
            cell_img = img_tau

        # Generate mask
        mean = (cell_img * fov_mask).sum() / fov_mask.sum()
        mask = pyplis.image.Img(cell_img / mean)

        # Take the average value at the edge of the crop region and set this to all values outside of it if we are
        # requested to do a cal crop
        if self.crop_sens_mask:
            invert_fov_mask = np.invert(self.cal_crop_region)
            mean_edge = np.mean(mask.img[self.cal_crop_line_mask])
            mask.img[invert_fov_mask] = mean_edge

        return mask

    def compute_plume_dists(self):
        """
        Computes plume distances for each pixel and associated error
        :return:
        """
        # TODO may not want to use this as it could be done in model light dilution
        self.plume_dists = self.meas.meas_geometry.plume_dist()

        self.plume_dist_err = self.meas.meas_geometry.plume_dist_err()

    def model_light_dilution(self, lines=None, draw=True, **kwargs):
        """
        Models light dilution and applies correction to images
        :param lines:   list    List of pyplis LineOnImage objects for light dilution intensity extraction
        :param draw:    bool    Defines whether results are to be ploted
        :param kwargs:  Any further settings to be passed to DilutionCorr object
        :return:
        """
        # If we are passed lines, we use these, otherwise we use already defined lines
        if lines is not None:
            self.light_dil_lines = lines

        # Ensure we only use the correct objects for processing
        self.light_dil_lines = [f for f in self.light_dil_lines if isinstance(f, LineOnImage)]

        # Compute distances to plume
        pix_dists, _, self.plume_dists = self.meas.meas_geometry.compute_all_integration_step_lengths()

        # Create the pyplis light dilution object
        dil = DilutionCorr(self.light_dil_lines, self.meas.meas_geometry, **kwargs)

        # Determine distances to the two lines defined above (every 6th pixel)
        for line_id in dil.line_ids:
            dil.det_topo_dists_line(line_id)

        # Estimate ambient intensity using defined ROI
        amb_int_on = self.vigncorr_A.crop(self.ambient_roi, True).mean()
        amb_int_off = self.vigncorr_B_warped.crop(self.ambient_roi, True).mean()

        # perform dilution anlysis and retrieve extinction coefficients (on-band)
        self.ext_on, _, _, ax0 = dil.apply_dilution_fit(img=self.vigncorr_A,
                                                   rad_ambient=amb_int_on,
                                                   i0_min=self.I0_MIN,
                                                   plot=draw)
        # Off-band
        self.ext_off, _, _, ax1 = dil.apply_dilution_fit(img=self.vigncorr_B_warped,
                                                   rad_ambient=amb_int_off,
                                                   i0_min=self.I0_MIN,
                                                   plot=draw)

        self.got_light_dil = True

        # Update light dilution plots if requested
        if draw:
            # Plot the results in a 3D map
            basemap = dil.plot_distances_3d(alt_offset_m=10, axis_off=False, draw_fov=True)
            basemap.fig.show()

            # Pass figures to LightDilutionSettings object to be plotted
            ax0.set_ylabel("Terrain radiances (on band)")
            ax1.set_ylabel("Terrain radiances (off band)")
            fig_dict = {'1': ax0.figure,
                        '2': ax1.figure,
                        '3': None,
                        '4': None}
            self.fig_dilution.update_figs(fig_dict)

    def corr_light_dilution(self, img, tau_uncorr, band='A'):
        """
        Corrects images for light dilution - uses simlar structure to pyplis.ImageList.correct_dilution()
        :return:
        """
        # Check that we have a light dilution model
        if not self.got_light_dil:
            warnings.warn("No light dilution model is present, cannot correct for light dilution")
            return

        # Get all appropriate images anf parameters based on
        if band.upper() == 'A' or 'ON':
            ext_coeff = self.ext_on
        elif band.upper() == 'B' or 'OFF':
            ext_coeff = self.ext_off
        else:
            print('Unrecognised definition of <band>, cannot perform light dilution correction')
            return

        # Compute plume background in image (don't fully understand this line)
        plume_bg_vigncorr = img * np.exp(tau_uncorr.img)

        # Calculate plume pixel mask
        plume_pix_mask = self.calc_plume_pix_mask(tau_uncorr, tau_thresh=self.tau_thresh)

        # Perform light dilution correction using pyplis function
        corr_img = correct_img(img, ext_coeff, plume_bg_vigncorr, self.plume_dists, plume_pix_mask)

        return corr_img

    def calc_plume_pix_mask(self, od_img, tau_thresh=0.05, erosion_kernel_size=0, dilation_kernel_size=0):
        """
        Calculates mask for plume pixels based on tau image and threshold.
        From pyplis.ImageList.calc_plumepix_thresh()
        """
        if not od_img.is_tau:
            raise ValueError("Input image must be optical density image "
                             "(or similar, e.g. calibrated CD image)")

        mask = od_img.to_binary(threshold=tau_thresh,
                                new_img=True)
        if erosion_kernel_size > 0:
            mask.erode(np.ones((erosion_kernel_size,
                             erosion_kernel_size), dtype=np.uint8))
        if dilation_kernel_size > 0:
            mask.dilate(np.ones((dilation_kernel_size,
                              dilation_kernel_size), dtype=np.uint8))
        return mask


    def update_doas_buff(self, doas_dict):
        """
        Updates doas buffer
        :param doas_dict:  dict     Must contain column_density and time keys, otherwise discarded. std_err is optional
        :return:
        """
        if 'column_density' not in doas_dict or 'time' not in doas_dict:
            print('Encountered unexpected value for doas_dict in update_doas_buff(), buffer was not updated')
            return

        # We either place the image in its position in the buffer, or if the buffer is already full we have to rearrange
        # the arrays and then put the new values at the end
        if self.idx_current_doas < self.doas_buff_size:
            idx = self.idx_current_doas
        else:
            idx = -1
            self.column_densities[:-1] = self.column_densities[1:]
            self.cd_times[:-1] = self.cd_times[1:]
            self.std_errs[:-1] =self.std_errs[1:]

        # Update buffers with new values
        self.column_densities[idx] = doas_dict['column_densities']
        self.cd_times[idx] = doas_dict['time']

        try:
            self.std_errs[idx] = doas_dict['std_err']
        except KeyError:
            self.std_errs[idx] = np.nan

        # Increment doas index
        self.idx_current_doas += 1

    def make_doas_results(self, times, column_densities, stds=None, species='SO2'):
        """
        Makes pydoas DOASResults object from timeseries
        :param times:   arraylike           Datetimes of column densities
        :param column_densities: arraylike  Column densities
        :param stds:    arraylike           Standard errors in the column density values
        :param species: str                 Gas species
        """
        doas_results = pydoas.analysis.DoasResults(column_densities, index=times, fit_errs=stds, species_id=species)
        return doas_results

    def doas_fov_search(self, img_stack, doas_results, plot=True):
        """
        Performs FOV search for doas
        :param img_stack:
        :param doas_results:
        :return:
        """
        s = pyplis.doascalib.DoasFOVEngine(img_stack, doas_results)
        s.maxrad = self.maxrad_doas   # Set maximum radius of FOV to close to that expected from optical calculations
        s.g2dasym = False                       # Allow only circular FOV (not eliptical)
        self.calib_pears = s.perform_fov_search(method='pearson')
        self.calib_pears.fit_calib_data(polyorder=1, through_origin=True)
        self.doas_fov_x, self.doas_fov_y = self.calib_pears.fov.pixel_position_center(abs_coords=True)
        self.doas_fov_extent = self.calib_pears.fov.pixel_extend(abs_coords=True)
        self.got_cal_doas = True     # Flag that we now have a calibration

        # Plot results if requested, first checking that we have the tkinter frame generated
        if plot:
            if not self.fig_doas_fov.in_frame:
                self.fig_doas_fov.generate_frame()  # Generating the frame will create the plot automatically
            else:
                self.fig_doas_fov.update_plot()

    def make_stack(self):
        """
        Generates image stack from self.img_tau_buff (tau images)
        :return stack:  ImgStack        Stack with all loaded images
        """
        # Create empty pyplis ImgStack
        stack = pyplis.processing.ImgStack(self.cam_specs.pix_num_y, self.cam_specs.pix_num_x, self.idx_current,
                                           np.float32, 'tau', camera=self.cam, img_prep={'is_tau': True})

        # Add all images of the current image buffer to stack
        # (only take images from the buffer stack up to the current index - the images which have been loaded thusfar)
        for i in range(self.idx_current):
            stack.add_img(self.img_tau_buff[:, :, i], self.img_tau_buff_time[i])

        stack.img_prep['pyrlevel'] = 0

        return stack

    def make_aa_list(self):
        """Makes pyplis ImgList from current img_list (set using get_img_list())"""
        full_paths = [os.path.join(self.img_dir, f) for f in self.img_list]
        self.pyplis_img_list = pyplis.imagelists.ImgList(full_paths, cam=self.cam)

    def model_background(self, mode=None, params=None, plot=True):
        """
        Models plume background for image provided.
        """
        self.vigncorr_A = pyplis.Img(self.img_A.img / self.vign_A)
        self.vigncorr_B = pyplis.Img(self.img_B.img / self.vign_B)
        self.vigncorr_A.edit_log['vigncorr'] = True
        self.vigncorr_B.edit_log['vigncorr'] = True

        # Create a warped version - required for light dilution work
        self.vigncorr_B_warped = pyplis.Img(self.img_reg.register_image(self.vigncorr_A.img, self.vigncorr_B.img))
        self.vigncorr_B_warped.edit_log['vigncorr'] = True

        if self.auto_param_bg and params is None:
            # Find reference areas using vigncorr, to avoid issues caused by sensor smudges etc
            params = pyplis.plumebackground.find_sky_reference_areas(self.vigncorr_A)
        self.plume_bg.update(**params)

        if mode in self.BG_CORR_MODES:
            self.plume_bg.mode = mode

        # Get PCS line if we have one
        if len(self.PCS_lines) < 1:
            pcs_line = None
        else:
            pcs_line = self.PCS_lines[0]

        # Get tau_A and tau_B
        if self.plume_bg.mode == 0:
            # mask for corr mode 0 (i.e. 2D polyfit)
            mask_A = np.ones(self.vigncorr_A.img.shape, dtype=np.float32)
            mask_A[self.vigncorr_A.img < self.polyfit_2d_mask_thresh] = 0
            mask_B = np.ones(self.vigncorr_B.img.shape, dtype=np.float32)
            mask_B[self.vigncorr_B.img < self.polyfit_2d_mask_thresh] = 0

            # First method: retrieve tau image using poly surface fit
            tau_A = self.plume_bg.get_tau_image(self.vigncorr_A,
                                                mode=self.BG_CORR_MODES[0],
                                                surface_fit_mask=mask_A,
                                                surface_fit_polyorder=1)
            tau_B = self.plume_bg.get_tau_image(self.vigncorr_B,
                                                mode=self.BG_CORR_MODES[0],
                                                surface_fit_mask=mask_B,
                                                surface_fit_polyorder=1)
        else:
            tau_A = self.plume_bg.get_tau_image(self.img_A, self.bg_A)
            tau_B = self.plume_bg.get_tau_image(self.img_B, self.bg_B)

        # Plots
        if plot:
            # Close old figures
            try:
                plt.close(self.fig_bg_A)
                plt.close(self.fig_bg_B)
                plt.close(self.fig_bg_ref)
            except AttributeError:
                pass

            if pcs_line is not None:
                self.fig_bg_A = self.plume_bg.plot_tau_result(tau_A, PCS=pcs_line)
                self.fig_bg_B = self.plume_bg.plot_tau_result(tau_B, PCS=pcs_line)
            else:
                self.fig_bg_A = self.plume_bg.plot_tau_result(tau_A)
                self.fig_bg_B = self.plume_bg.plot_tau_result(tau_B)

            self.fig_bg_A.canvas.set_window_title('Background model: Image A')
            self.fig_bg_B.canvas.set_window_title('Background model: Image B')

            self.fig_bg_A.show()
            self.fig_bg_B.show()

            # Reference areas
            self.fig_bg_ref, axes = plt.subplots(1, 1, figsize=(16, 6))
            pyplis.plumebackground.plot_sky_reference_areas(self.vigncorr_A, params, ax=axes)
            axes.set_title("Automatically set parameters")
            self.fig_bg_ref.canvas.set_window_title('Background model: Reference parameters')
            self.fig_bg_ref.show()

        self.tau_A = tau_A
        self.tau_B = tau_B

        return tau_A, tau_B

    def generate_optical_depth(self, plot=True, plot_bg=True):
        """
        Performs the full catalogue of image procesing on a single image pair to generate optical depth image
        Processing beyond this point is left ot another function, since it requires use of a second set of images

        :param img_A: pyplis.image.Img      On-band image
        :param img_B: pyplis.image.Img      Off-band image
        :returns img_aa:    Optical depth image
        """
        # Set last image to img_tau_prev, as it is used in optical flow computation
        self.img_tau_prev = self.img_tau

        # Model sky backgrounds and sets self.tau_A and self.tau_B attributes
        self.model_background(plot=plot_bg)

        # Register off-band image
        img = self.img_reg.register_image(self.tau_A.img, self.tau_B.img)
        self.tau_B_warped = pyplis.image.Img(img)

        # Perform light dilution if we have a correction
        if self.got_light_dil:
            self.lightcorr_A = self.corr_light_dilution(self.vigncorr_A, self.tau_A, band='A')
            self.lightcorr_B = self.corr_light_dilution(self.vigncorr_B_warped, self.tau_B_warped, band='A')

        self.img_tau = pyplis.image.Img(self.tau_A.img - self.tau_B_warped.img)
        self.img_tau.edit_log["is_tau"] = True
        self.img_tau.edit_log["is_aa"] = True
        self.img_tau.meta['start_acq'] = self.img_A.meta['start_acq']

        # Adjust for changing FOV sensitivity if requested
        if self.use_sensitivity_mask:
            self.img_tau.img = self.img_tau.img / self.sensitivity_mask

        if self.got_cal_doas and self.cal_type in [1, 2]:
            #TODO perform calibration here if we have a calibration line.
            pass

        if self.cal_type == 0:
            # TODO perform cell calibration here
            pass

        if plot:
            # TODO should include a tau vs cal flag check, to see whether the plot is displaying AA or ppmm
            self.fig_tau.update_plot(np.array(self.img_tau.img))

    def update_opt_flow_settings(self, **settings):
        """
        Updates optical flow object's settings
        :param settings:
        :return:
        """
        for key in settings:
            setattr(self.opt_flow.settings, key, settings[key])

            # If we have the roi_rad setting we need to set roi_rad_abs too
            if key == 'roi_abs':
                self.opt_flow.settings.roi_rad_abs = settings[key]

    def generate_opt_flow(self, plot=False):
        """
        Generates optical flow vectors for current and previous image
        :return:
        """
        # Update optical flow object to contain previous tau image and current image
        # This includes updating_contrast_range() and prep_images()
        self.opt_flow.set_images(self.img_tau_prev, self.img_tau)

        # Calculate optical flow vectors
        self.flow = self.opt_flow.calc_flow()

        # Generate plume speed array
        dist_img, _, _ = self.meas.meas_geometry.compute_all_integration_step_lengths(pyrlevel=0)
        self.velo_img = pyplis.image.Img(self.opt_flow.to_plume_speed(dist_img))

        if plot:
            self.fig_tau.update_plot(self.img_tau.img)
            if self.fig_opt.in_frame:
                self.fig_opt.update_plot()

        return self.flow, self.velo_img

    def process_pair(self, img_path_A=None, img_path_B=None, plot=True, plot_bg=False):
        """
        Processes full image pair when passed images (need to think about how to deal with dark images)

        :param img_path_A: str  path to on-band image
        :param img_path_B: str  path to corresponding off-band image
        :param plot: bool       defines whether the loaded images are plotted
        :param plot_bg: bool    defines whether the background models are plotted
        :return:
        """

        # Can pass None to this function for img paths, and then the current images will be processed
        if img_path_A is not None:
            # Load in images
            self.load_img(img_path_A, band='A', plot=plot)
        if img_path_B is not None:
            self.load_img(img_path_B, band='B', plot=plot)

        # Generate optical depth image
        self.generate_optical_depth(plot=plot, plot_bg=plot_bg)

        # Only update buffer if this is a new image (if it is a new image the path will be given to it)
        # Really we only want to update the image buffer when processing a full sequence, which will always pass
        # the img_path parameter to this function, so will update the buffer correctly.
        if img_path_A is not None:
            # Add tau image to buffer and update image time too
            self.update_img_buff(self.img_tau.img, img_path_A)

        # Wind speed and subsequent flux calculation if we aren't in the first image of a sequence
        if not self.first_image:
            self.generate_opt_flow(plot=plot)
            # TODO all of processing if not the first image pair

    def process_sequence(self):
        """Start _process_sequence in a thread, so that this can return after starting and the GUI doesn't lock up"""
        self.process_thread = threading.Thread(target=self._process_sequence, args=())
        self.process_thread.daemon = True
        self.process_thread.start()

    def _process_sequence(self):
        """
        Processes the current image directory
        :param plot_iter: bool      Tells function whether to plot iteratively or not
        :return:
        """
        # Reset important parameters to ensure we start processing correctly
        self.reset_self()

        # Set plot iter for this period, get it from current setting for this attribute
        plot_iter = self.plot_iter

        # Add images to queue to be displayed if the plot_iter requested
        self.img_list = self.get_img_list()

        # Perform calibration work
        if self.cal_type in [0, 2]:
            # TODO Need to create an option in cell calibration frame to change use_cell_bg as an option
            # TODO this bool defines whetehr cal_dir is used for automatically finding the clear sky image for bg modelling
            self.perform_cell_calibration(plot=False, set_bg_img=self.use_cell_bg)

        # Loop through img_list and process data
        self.first_image = True
        for i in range(len(self.img_list)):

            # Always plot the final image
            if i == len(self.img_list) - 1:
                plot_iter = True

            # Process image pair
            self.process_pair(self.img_dir + '\\' + self.img_list[i][0], self.img_dir + '\\' + self.img_list[i][1],
                              plot=plot_iter)

            # Once first image is processed we update the first_image bool
            if i == 0:
                self.first_image = False

            # Wait for defined amount of time to allow plotting of data without freezing up
            time.sleep(self.wait_time)

        if self.cal_type in [1,2]:
            # TODO Edit this test to use proper data (currently uses dummy random values)
            # Current test for performing DOAS FOV search
            stack = self.make_stack()
            doas_results = self.make_doas_results(self.test_doas_times, self.test_doas_cds, stds=self.test_doas_stds)
            self.doas_fov_search(stack, doas_results)

    def start_processing(self):
        """Public access thread starter for _processing"""
        self.process_thread = threading.Thread(target=self._processing, args=())
        self.process_thread.daemon = True
        self.process_thread.start()

    def _processing(self):
        """
        Main processing function for continuous processing
        """
        while True:
            # Get the next images in the list
            img_path_A, img_path_B = self.q.get(block=True)

            # Process the pair
            self.process_pair(img_path_A, img_path_B, plot=self.plot_iter)

            # Attempt to get DOAS calibration point to add to list
            try:
                doas_dict = self.q_doas.get()
                # If we have been passed a processed spectrum, we load it into the buffer
                if 'column_density' in doas_dict and 'time' in doas_dict:
                    self.update_doas_buff(doas_dict)
            except queue.Empty:
                pass

            # TODO After a certain amount of time we need to perform doas calibration (maybe once DOAS buff is full?
            # TODO start of day will be uncalibrated until this point



class ImageRegistration:
    """
    Image registration class for warping the off-band image to align with the on-band image
    """
    def __init__(self):
        self.method = None
        self.got_cv_transform = False  # Defines whether the transform matrix has been generated yet
        self.got_cp_transform = False

        self.warp_matrix_cv = False
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.cv_opts = {'num_it': 500, 'term_eps': 1e-10}

        self.cp_tform = tf.SimilarityTransform()

    # ======================================================
    # OPENCV IMAGE REGISTRATION
    # ======================================================
    def cv_generate_warp_matrix(self, img_A, img_B):
        """Calculate the warp matrix 'warp_matrix_cv' in preparation for image registration"""
        img_A = np.array(img_A, dtype=np.float32)  # Converting image to a 32-bit float for processing (required by OpenCV)
        img_B = np.array(img_B, dtype=np.float32)  # Converting image to a 32-bit float for processing (required by OpenCV)

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = self.cv_opts['num_it']

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = self.cv_opts['term_eps']

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(img_A, img_B, warp_matrix, self.warp_mode, criteria)

        self.got_cv_transform = True
        self.warp_matrix_cv = warp_matrix

    def cv_warp_img(self, img, warp_matrix=None):
        """Register the image and return the aligned image"""
        if warp_matrix is not None:
            self.warp_matrix_cv = warp_matrix

        # Convert image to a 32-bit float for processing (required by OpenCV)
        img = np.array(img, dtype=np.float32)
        sz = img.shape

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            img_warped = cv2.warpPerspective(img, self.warp_matrix_cv, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            img_warped = cv2.warpAffine(img, self.warp_matrix_cv, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Convert bakc to a unit16, to be in keeping with cp_warp_img return
        img_warped = np.uint16(img_warped)
        return img_warped

    # ============================================
    # CONTROL POINT REGISTRATION
    # ============================================
    def cp_tform_gen(self, coord_A, coord_B):
        """Generate CP image registration transform object"""
        self.cp_tform.estimate(coord_A, coord_B)
        self.got_cp_transform = True

    def cp_warp_img(self, img):
        """Perform CP image registration"""
        # Perform warp
        img_warped = tf.warp(img, self.cp_tform, output_shape=img.shape)

        # Scale image back - warp function can change the values in the image for some reason
        max_img = np.amax(img)
        img_warped *= max_img / np.amax(img_warped)
        img_warped = np.float32(img_warped)
        return img_warped

    def register_image(self, img_A, img_B, **kwargs):
        """
        Performs image registration based on the current method set
        :param img_A: np.ndarray        The reference image to be registered to (only used with CV registration)
        :param img_B: np.ndarray        The image to be registered
        :return: warped_B: np.ndarray   Registered image
        """

        if self.method is None:
            # No registration
            warped_B = img_B

        elif self.method.lower() == 'cv':
            if not self.got_cv_transform:
                # Generate transform
                self.cv_generate_warp_matrix(img_A, img_B)

            # Warp image
            warped_B = self.cv_warp_img(img_B)

        elif self.method.lower() == 'cp':
            if not self.got_cp_transform:       # TODO got_cp_transform should be set to false when reset CP button or save CP button is pressed in GUI
                # Get coordinates from kwargs - if they are not present we need to return as we can't perform transform
                cp_setts = [kwargs[key] for key in ['coord_A', 'coord_B'] if key in kwargs]

                # If we don't have a transform and aren't provided the correct values to create one, we return the
                # input image
                if len(cp_setts) != 2:
                    # print('coord_A and coord_B arguments are required to perform CP registration when transform is not already present')
                    warped_B = img_B
                # Else we generate the tform and warp the image
                else:
                    # Generate transform if we have more than one point for coordinates
                    self.cp_tform_gen(*cp_setts)
                    warped_B = self.cp_warp_img(img_B)
            # If we already have a cp transform we are happy with, we can warp the image straight away
            else:
                warped_B = self.cp_warp_img(img_B)

        return warped_B


# ## SCRIPT FUNCTION DEFINITIONS
def create_picam_new_filters(geom_info):
    # Picam camera specs (default)
    cam_specs = CameraSpecs()

    # Start with creating an empty Camera object
    cam = pyplis.setupclasses.Camera(**geom_info)

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
    cam.file_type = "png"

    # File name delimiter for information extraction
    cam.delim = "_"

    # position of acquisition time (and date) string in file name after
    # splitting with delimiter
    cam.time_info_pos = 0

    # datetime string conversion of acq. time string in file name
    cam.time_info_str = "%Y-%m-%dT%H%M%S"

    # position of image filter type acronym in filename
    cam.filter_id_pos = 1

    # position of meas type info
    cam.meas_type_pos = 4

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
    cam.focal_length = cam_specs.estimate_focal_length()

    # Detector geometry
    # Using PiCam binned values (binning 2592 x 1944 to 648 x 486)
    cam.pix_height = cam_specs.pix_size_x  # pixel height in m
    cam.pix_width = cam_specs.pix_size_y  # pixel width in m
    cam.pixnum_x = cam_specs.pix_num_x
    cam.pixnum_y = cam_specs.pix_num_y

    cam._init_access_substring_info()

    cam.io_opts = dict(USE_ALL_FILES=False,
                       SEPARATE_FILTERS=True,
                       INCLUDE_SUB_DIRS=True,
                       LINK_OFF_TO_ON=True)

    # Set the custom image import method
    cam.image_import_method = pyplis.custom_image_import.load_picam_png
    # That's it...
    return cam


def plot_pcs_profiles_4_tau_images(tau0, tau1, tau2, tau3, pcs_line):
    """Plot PCS profiles for all 4 methods."""
    BG_CORR_MODES = [0,    # 2D poly surface fit (without sky radiance image)
                     1,    # Scaling of sky radiance image
                     4,    # Scaling + linear gradient correction in x & y direction
                     6]    # Scaling + quadr. gradient correction in x & y direction

    fig, ax = plt.subplots(1, 1)
    tau_imgs = [tau0, tau1, tau2, tau3]

    for k in range(4):
        img = tau_imgs[k]
        profile = pcs_line.get_line_profile(img)
        ax.plot(profile, "-", label=r"Mode %d: $\phi=%.3f$"
                % (BG_CORR_MODES[k], np.mean(profile)))

    ax.grid()
    ax.set_ylabel(r"$\tau_{on}$", fontsize=20)
    ax.set_xlim([0, pcs_line.length()])
    ax.set_xticklabels([])
    ax.set_xlabel("PCS", fontsize=16)
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12)
    return fig


class UnrecognisedSourceError(BaseException):
    """Error raised for a source which cannot be found online"""
    pass
