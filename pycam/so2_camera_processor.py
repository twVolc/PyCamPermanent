# -*- coding: utf-8 -*-

# PycamUV
"""Setup for pyplis usage, controlling filter initiation etc
Scripts are an edited version of the pyplis example scripts, adapted for use with the PiCam"""
from __future__ import (absolute_import, division)

from pycam.setupclasses import CameraSpecs

import queue
import threading
import pyplis
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import transform as tf
import warnings
from math import log10, floor


class PyplisWorker:
    """
    Main pyplis worker class

    """
    def __init__(self, img_dir=None):
        self.q = queue.Queue()      # Queue object for images. Images are passed in a pair for fltrA and fltrB

        # Pyplis object setup
        self.load_img_func = pyplis.custom_image_import.load_picam_png
        self.cam = create_picam_new_filters({})         # Generate pyplis-picam object
        self.meas = pyplis.setupclasses.MeasSetup()     # Pyplis MeasSetup object (instantiated empty)
        self.img_reg = ImageRegistration()              # Image registration object
        self.plume_bg = pyplis.plumebackground.PlumeBackgroundModel()
        self.plume_bg.surface_fit_pyrlevel = 0
        self.plume_bg.mode = 4      # Plume background mode - default (4) is linear in x and y
        self.cam_specs = CameraSpecs()
        self.BG_CORR_MODES = [0,    # 2D poly surface fit (without sky radiance image)
                              1,    # Scaling of sky radiance image
                              2,
                              3,
                              4,    # Scaling + linear gradient correction in x & y direction
                              5,
                              6,    # Scaling + quadr. gradient correction in x & y direction
                              99]
        self.auto_param_bg = True   # Whether the line parameters for BG modelling are generated automatically
        self.POLYFIT_2D_MASK_THRESH = 100
        self.PCS_lines = []

        # Figure objects
        self.fig_A = None
        self.fig_B = None
        self.fig_tau = None
        self.fig_bg_A = None
        self.fig_bg_B = None
        self.fig_bg_ref = None

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

        # Load background image if we are provided with one
        self.bg_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vign_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vigncorr_A = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.bg_A_path = None

        # Load background image if we are provided with one
        self.bg_B = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.vign_B = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
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

    def update_cam_geom(self, geom_info):
        """Updates camera geometry info by creating a new object

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

        # Update image list
        self.img_list = self.get_img_list()

        # Display first images of sequence
        if len(self.img_list) > 0:
            self.process_pair(self.img_dir + '\\' + self.img_list[0][0],
                              self.img_dir + '\\' + self.img_list[0][1],
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
        dark_img = self.find_dark_img(self.dark_dir, band=band, ss=ss)

        if dark_img is not None:
            img.subtract_dark_image(dark_img)
        else:
            warnings.warn('No dark image provided for background image.\n '
                          'Background image has not been corrected for dark current.')

        # Generate vign image
        vign = img.img / img.img.max()  # NOTE: potentially includes y & x gradients

        # Set variables
        setattr(self, 'bg_{}'.format(band), img)
        setattr(self, 'vign_{}'.format(band), vign)
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

        # Dark subtraction - first extract ss then hunt for dark image
        ss = str(int(img.texp * 10 ** 6))
        dark_img = self.find_dark_img(self.dark_dir, band=band, ss=ss)

        if dark_img is not None:
            img.subtract_dark_image(dark_img)
        else:
            warnings.warn('No dark image found, image has been loaded without dark subtraction')

        # Add to plot queue if requested
        if plot:
            # getattr(self, 'img_{}_q'.format(band)).put([img_path, img])
            getattr(self, 'fig_{}'.format(band)).update_plot(np.array(img.img, dtype=np.uint16), img_path)

        # Finally set object attribute to the loaded pyplis image
        setattr(self, 'img_{}'.format(band), img)

    def find_dark_img(self, img_dir=None, band='on', ss=None):
        """
        Searches for suitable dark image in designated directory. First it filters the images for the correct filter,
        then searches for an image with the same shutter speed defined
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

    def model_background(self, mode=None, params=None, plot=True):
        """
        Models plume background for image provided.
        """
        self.vigncorr_A = pyplis.Img(self.img_A.img / self.vign_A)
        self.vigncorr_B = pyplis.Img(self.img_B.img / self.vign_B)

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
            mask_A[self.vigncorr_A.img < self.POLYFIT_2D_MASK_THRESH] = 0
            mask_B = np.ones(self.vigncorr_B.img.shape, dtype=np.float32)
            mask_B[self.vigncorr_B.img < self.POLYFIT_2D_MASK_THRESH] = 0

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

    def generate_optical_depth(self, img_A, img_B, plot=True, plot_bg=True):
        """
        Performs the full catalogue of image procesing on a single image pair to generate optical depth image
        Processing beyond this point is left ot another function, since it requires use of a second set of images

        :param img_A: pyplis.image.Img      On-band image
        :param img_B: pyplis.image.Img      Off-band image
        :returns img_aa:    Optical depth image
        """
        # Model sky backgrounds
        self.model_background(plot=plot_bg)

        # Register off-band image TODO - maybe I want to register first? As after modelling BG i will be in optical depth not intensity space
        if self.img_reg.method is None:
            self.tau_B_warp = self.tau_B
        elif self.img_reg.method == 'cv':
            self.img_reg.cv_generate_warp_matrix(self.vigncorr_A.img, self.vigncorr_B.img)
            img = self.img_reg.cv_warp_img(self.tau_B.img)
            self.tau_B_warp = pyplis.image.Img(img)

        self.img_tau = pyplis.image.Img(self.tau_A.img - self.tau_B_warp.img)
        self.img_tau.edit_log["is_tau"] = True
        self.img_tau.edit_log["is_aa"] = True

        if plot:
            # TODO update optical depth image
            # TODO should include a tau vs cal flag check, to see whether the plot is displaying AA or ppmm
            getattr(self, 'fig_tau').update_plot(np.array(self.img_tau.img))

        # return img_tau

    def process_pair(self, img_path_A, img_path_B, plot=True, plot_bg=True):
        """
        Processes full image pair when passed images (need to think about how to deal with dark images)

        :param img_path_A: str  path to on-band image
        :param img_path_B: str  path to corresponding off-band image
        :param plot: bool       defines whether the loaded images are plotted
        :param plot_bg: bool    defines whether the background models are plotted
        :return:
        """

        # Load in images TODO think about what to do with providing dark images here -should they already be loaded?
        self.load_img(img_path_A, band='A', plot=plot)
        self.load_img(img_path_B, band='B', plot=plot)

        # Generate optical depth image
        self.img_aa = self.generate_optical_depth(self.img_A, self.img_B, plot=plot, plot_bg=plot_bg)

        # Wind speed and subsequent flux calculation if we aren't in the first image of a sequence
        if not self.first_image:
            pass    # TODO all of processing if not the first image pair

    def process_sequence(self):
        """
        Processes the current image directory
        :param plot_iter: bool      Tells function whether to plot iteratively or not
        :return:
        """
        # Set plot iter for this period, get it from current setting for this attribute
        plot_iter = self.plot_iter

        # Add images to queue to be displayed if the plot_iter requested
        self.img_list = self.get_img_list()

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
            self.process_pair(img_path_A, img_path_B)


class ImageRegistration:
    """
    Image registration class for warping the off-band image to align with the on-band image
    """
    def __init__(self):
        self.method = 'cv'
        self.transformed_B = False  # Defines whether the transform matrix has been generated yet

        self.warp_matrix_cv = False
        self.warp_mode = cv2.MOTION_EUCLIDEAN

        self.cp_tform = tf.SimilarityTransform()

    # ======================================================
    # OPENCV IMAGE REGISTRATION
    # ======================================================
    def cv_generate_warp_matrix(self, img_A, img_B, numIt=500, term_eps=1e-10):
        """Calculate the warp matrix 'warp_matrix_cv' in preparation for image registration"""
        img_A = np.array(img_A, dtype=np.float32)  # Converting image to a 32-bit float for processing (required by OpenCV)
        img_B = np.array(img_B, dtype=np.float32)  # Converting image to a 32-bit float for processing (required by OpenCV)

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = numIt

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = term_eps

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(img_A, img_B, warp_matrix, self.warp_mode, criteria)

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

    def cp_warp_img(self, img):
        """Perform CP image registration"""
        img_warped = tf.warp(img, self.cp_tform, output_shape=img.shape)
        max_img = np.amax(img)
        img_warped *= max_img / np.amax(img_warped)
        img_warped = np.uint16(img_warped)
        return img_warped


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
    cam.focal_length = cam_specs.estimate_focal_length()

    # Detector geometry
    # Using PiCam binned values (binning 2592 x 1944 to 648 x 486)
    cam.pix_height = cam_specs.pix_size_x  # pixel height in m
    cam.pix_width = cam_specs.pix_size_y  # pixel width in m
    cam.pixnum_x = cam_specs.pix_num_x
    cam.pixnum_y = cam_specs.pix_size_y

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
