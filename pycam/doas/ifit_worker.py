# -*- coding: utf-8 -*-

"""
Much like DOASWorker but designed for iFit rather than DOAS retrieval using Esse's code for iFit retrieval.
This could be advantageous since we don't know exactly what the clear-sky spectrum looks like? Although this isn't
critical for getting the gradient of AA vs ppm.m in camera calibration.
This work may also allow light dilution correction following Varnam 2021
"""

import queue
import numpy as np
import datetime
import warnings
import threading
import copy
import os
from itertools import compress
from tkinter import filedialog
from astropy.convolution import convolve
import sys
import os
# Make it possible to import iFit by updating path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'ifit'))

from pycam.directory_watcher import create_dir_watcher
from pycam.setupclasses import SpecSpecs
from pycam.io import load_spectrum, spec_txt_2_npy
from ifit.parameters import Parameters
from ifit.spectral_analysis import Analyser
from pydoas.analysis import DoasResults

try:
    from scipy.constants import N_A
except BaseException:
    N_A = 6.022140857e+23


class IFitWorker:
    """
    Class to control DOAS processing
    General order of play for processing:
    Initiate class,
    get_ref_spectrum()
    set_fit_window()
    shift_spectrum()

    :param q_doas: queue.Queue   Queue where final processed dictionary is placed (should be a PyplisWorker.q_doas)
    """
    def __init__(self, routine=2, species=['SO2'], spec_specs=SpecSpecs(), spec_dir='C:\\', dark_dir=None,
                 q_doas=queue.Queue(), frs_path='./ifit/Ref/sao2010.txt', ref_paths={}):
        self.routine = routine          # Defines routine to be used, either (1) Polynomial or (2) Digital Filtering

        self.spec_specs = spec_specs    # Spectrometer specifications



        # ======================================================================================================================
        # Initial Definitions
        # ======================================================================================================================
        self.ppmm_conversion = 2.7e15   # convert absorption cross-section in cm2/molecule to ppm.m (MAY NEED TO CHANGE THIS TO A DICTIONARY AS THE CONVERSION MAY DIFFER FOR EACH SPECIES?)
        self._conversion_factor = 2.663 * 1e-6  # Conversion for ppm.m into Kg m-2
        MOL_MASS_SO2 = 64.0638  # g/mol
        self.ppmm_conv = (self._conversion_factor * N_A * 1000) / (
                100 ** 2 * MOL_MASS_SO2)  # Conversion for ppm.m to molecules cm-2

        self.shift = 0                  # Shift of spectrum in number of pixels
        self.shift_tol = 0              # Shift tolerance (will process data at multiple shifts defined by tolerance)
        self.stretch = 0                # Stretch of spectrum
        self.stretch_tol = 0            # As shift_tol but for stretch
        self.stretch_adjuster = 0.0001  # Factor to scale stretch (needed if different spectrometers have different pixel resolutions otherwise the stretch applied may be in too large or too small stages)
        self.stretch_resample = 100     # Number of points to resample the spectrum by during stretching
        self._start_stray_pix = None    # Pixel space stray light window definitions
        self._end_stray_pix = None
        self._start_stray_wave = 280    # Wavelength space stray light window definitions
        self._end_stray_wave = 290
        self._start_fit_pix = None  # Pixel space fitting window definitions
        self._end_fit_pix = None
        self._start_fit_wave = 305  # Wavelength space fitting window definitions
        self._end_fit_wave = 320
        self.fit_window = None      # Fitting window, determined by set_fit_window()
        self.fit_window_ref = None  # Placeholder for shifted fitting window for the reference spectrum
        self.wave_fit = True        # If True, wavelength parameters are used to define fitting window

        self.wavelengths = None         # Placeholder for wavelengths attribute which contains all wavelengths of spectra
        self.wavelengths_cut = None     # Wavelengths in fit window
        self._dark_spec = None           # Dark spectrum
        self.dark_dict = {}             # Dictionary holding all dark spectra loaded in
        self._clear_spec_raw = None     # Clear (fraunhofer) spectrum - not dark corrected
        self._plume_spec_raw = None     # In-plume spectrum (main one which is used for calculation of SO2
        self.clear_spec_corr = None     # Clear (fraunhofer) spectrum - typically dark corrected and stray light corrected
        self.plume_spec_corr = None     # In-plume spectrum (main one which is used for calculation of SO2
        self.ref_spec = dict()          # Create empty dictionary for holding reference spectra
        self.ref_spec_interp = dict()   # Empty dictionary to hold reference spectra after sampling to spectrometer wavelengths
        self.ref_spec_conv = dict()     # Empty dictionary to hold reference spectra after convolving with ILS
        self.ref_spec_cut = dict()      # Ref spectrum cut to fit window
        self.ref_spec_ppmm = dict()   # Convolved ref spectrum scaled by ppmm_conversion factor
        self.ref_spec_filter = dict()   # Filtered reference spectrum
        self.ref_spec_fit = dict()      # Ref spectrum scaled by ppmm (for plotting)
        self.ref_spec_types = ['SO2', 'O3', 'Ring'] # List of reference spectra types accepted/expected
        self.ref_spec_used = species    # Reference spectra we actually want to use at this time (similar to ref_spec_types - perhaps one is obsolete (or should be!)
        self.abs_spec = None
        self.abs_spec_cut = None
        self.abs_spec_filt = None
        self.abs_spec_species = dict()  # Dictionary of absorbances isolated for individual species
        self.ILS_wavelengths = None     # Wavelengths for ILS
        self._ILS = None                 # Instrument line shape (will be a numpy array)
        self.processed_data = False     # Bool to define if object has processed DOAS yet - will become true once process_doas() is run

        self.poly_order = 2  # Order of polynomial used to fit residual

        self.start_ca = -2000  # Starting column amount for iterations
        self.end_ca = 20000  # Ending column amount for iterations
        self.vals_ca = np.arange(self.start_ca, self.end_ca+1)  # Array of column amounts to be iterated over
        self.vals_ca_cut_idxs = np.arange(0, len(self.vals_ca), 100)
        self.vals_ca_cut = self.vals_ca[self.vals_ca_cut_idxs]
        self.mse_vals_cut = np.zeros(len(self.vals_ca_cut))
        self.mse_vals = np.zeros(len(self.vals_ca))  # Array to hold mse values

        self.std_err = None
        self.column_density = dict()

        self.filetypes = dict(defaultextension='.png', filetypes=[('PNG', '*.png')])

        # ----------------------------------------------------------------------------------
        # We need to make sure that all images are dark subtracted before final processing
        # Also make sure that we don't dark subtract more than once!
        self.ref_convolved = False  # Bool defining if reference spe has been convolved - speeds up DOAS processing
        self.new_spectra = True
        self.dark_corrected_clear = False
        self.dark_corrected_plume = False
        self.stray_corrected_clear = False    # Bool defining if all necessary spectra have been stray light corrected
        self.stray_corrected_plume = False    # Bool defining if all necessary spectra have been stray light corrected

        self.have_dark = False  # Used to define if a dark image is loaded.
        self.cal_dark_corr = False  # Tells us if the calibration image has been dark subtracted
        self.clear_dark_corr = False  # Tells us if the clear image has been dark subtracted
        self.plume_dark_corr = False  # Tells us if the plume image has been dark subtracted
        # ==============================================================================================================

        # Processing loop attributes
        self.process_thread = None      # Thread for running processing loop
        self.processing_in_thread = False   # Flags whether the object is processing in a thread or in the main thread - therefore deciding whether plots should be updated herein or through pyplisworker
        self.q_spec = queue.Queue()     # Queue where spectra files are placed, for processing herein
        self.q_doas = q_doas
        self.watcher = None
        self.watching_dir = None
        self.watching = False

        self._dark_dir = None
        self.dark_dir = dark_dir        # Directory where dark images are stored
        self.spec_dir = spec_dir        # Directory where plume spectra are stored
        self.spec_dict = {}             # Dictionary containing all spectrum files from current spec_dir

        # Figures
        self.fig_spec = None            # pycam.doas.SpectraPlot object
        self.fig_doas = None            # pycam.doas.DOASPlot object

        # Results object
        self.results = DoasResults([], index=[], fit_errs=[], species_id='SO2')

        # ==============================================================================================================
        # iFit setup
        # ==============================================================================================================
        self.frs_path = frs_path
        self.ref_spec_types = list(ref_paths.keys())


        # Create parameter dictionary
        self.params = Parameters()

        # Add the gases
        for species in ref_paths:
            self.params.add(species, value=ref_paths[species]['value'], vary=True, xpath=ref_paths[species]['path'])

            # Load reference spectrum
            self.load_ref_spec(ref_paths[species]['path'], species)


        # Add background polynomial parameters
        self.params.add('bg_poly0', value=0.0, vary=True)
        self.params.add('bg_poly1', value=0.0, vary=True)
        self.params.add('bg_poly2', value=0.0, vary=True)
        self.params.add('bg_poly3', value=1.0, vary=True)

        # Add intensity offset parameters
        self.params.add('offset0', value=0.0, vary=True)

        # Add wavelength shift parameters
        self.params.add('shift0', value=0.0, vary=True)
        self.params.add('shift1', value=0.1, vary=True)

        # Setup ifit analyser (we will stray correct ourselves
        self.analyser = Analyser(params=self.params,
                                 fit_window=[self.start_fit_wave, self.end_stray_wave],
                                 frs_path=frs_path,
                                 stray_flag=False,      # We stray correct prior to passing spectrum to Analyser
                                 dark_flag=False)       # We dark correct prior to passing the spectrum to Analyser

    def reset_self(self):
        """Some resetting of object, before processing occurs"""
        # Reset results objecct
        self.results = DoasResults([], index=[], fit_errs=[], species_id='SO2')

    @property
    def start_stray_wave(self):
        return self._start_stray_wave

    @start_stray_wave.setter
    def start_stray_wave(self, value):
        self._start_stray_wave = value

        # Set pixel value too, if wavelengths attribute is present
        if self.wavelengths is not None:
            self._start_stray_pix = np.argmin(np.absolute(self.wavelengths - value))

    @property
    def end_stray_wave(self):
        return self._end_stray_wave

    @end_stray_wave.setter
    def end_stray_wave(self, value):
        self._end_stray_wave = value

        # Set pixel value too, if wavelengths attribute is present
        if self.wavelengths is not None:
            self._end_stray_pix = np.argmin(np.absolute(self.wavelengths - value))

    @property
    def start_fit_wave(self):
        return self._start_fit_wave

    @start_fit_wave.setter
    def start_fit_wave(self, value):
        self._start_fit_wave = value

        # Set pixel value too, if wavelengths attribute is present
        if self.wavelengths is not None:
            self._start_fit_pix = np.argmin(np.absolute(self.wavelengths - value))

    @property
    def end_fit_wave(self):
        return self._end_fit_wave

    @end_fit_wave.setter
    def end_fit_wave(self, value):
        self._end_fit_wave = value

        # Set pixel value too, if wavelengths attribute is present
        if self.wavelengths is not None:
            self._end_fit_pix = np.argmin(np.absolute(self.wavelengths - value))

    # --------------------------------------------
    # Set spectra attributes so that whenever they are updated we flag that they have not been dark or stray corrected
    # If we have a new dark spectrum too we need to assume it is to be used for correcting spectra, so all corrected
    # spectra, both dark and stray, become invalid
    @property
    def dark_spec(self):
        return self._dark_spec

    @dark_spec.setter
    def dark_spec(self, value):
        self._dark_spec = value

        # If we have a new dark image all dark and stray corrections become invalid
        self.dark_corrected_clear = False
        self.dark_corrected_plume = False
        self.stray_corrected_clear = False
        self.stray_corrected_plume = False

    @property
    def dark_dir(self):
        return self._dark_dir

    @dark_dir.setter
    def dark_dir(self, value):
        """If dark_dir is changed we need to reset the dark_dict which holds preloaded dark specs"""
        self.dark_dict = {}
        self._dark_dir = value

    @property
    def clear_spec_raw(self):
        return self._clear_spec_raw

    @clear_spec_raw.setter
    def clear_spec_raw(self, value):
        self._clear_spec_raw = value
        self.dark_corrected_clear = False
        self.stray_corrected_clear = False

    @property
    def plume_spec_raw(self):
        return self._plume_spec_raw

    @plume_spec_raw.setter
    def plume_spec_raw(self, value):
        self._plume_spec_raw = value
        self.dark_corrected_plume = False
        self.stray_corrected_plume = False

    @property
    def ILS(self):
        """ILS array"""
        return self._ILS

    @ILS.setter
    def ILS(self, value):
        self._ILS = value

        # If new ILS is generated, then must flag that ref spectrum is no longer convolved with up-to-date ILS
        self.ref_convolved = False

    def load_ils(self, path):
        """Load ils from path"""
        self.analyser = Analyser(params=self.params,
                                 fit_window=[self.start_fit_wave, self.end_stray_wave],
                                 frs_path=self.frs_path,
                                 stray_flag=False,      # We stray correct prior to passing spectrum to Analyser
                                 dark_flag=False,
                                 ils_type='File',
                                 ils_path=path)

        # Load ILS
        self.ILS_wavelengths, self.ILS = np.loadtxt(ils_path, unpack=True)
    # -------------------------------------------

    def get_spec_time(self, filename):
        """
        Gets time from filename and converts it to datetime object
        :param filename:
        :return spec_time:
        """
        # Make sure filename only contains file and not larger pathname
        filename = filename.split('\\')[-1].split('/')[-1]

        # Extract time string from filename
        time_str = filename.split('_')[self.spec_specs.file_date_loc]

        # Turn time string into datetime object
        spec_time = datetime.datetime.strptime(time_str, self.spec_specs.file_datestr)

        return spec_time

    def dark_corr_spectra(self):
        """Subtract dark spectrum from spectra"""
        if not self.dark_corrected_clear:
            self.clear_spec_corr = self.clear_spec_raw - self.dark_spec
            self.clear_spec_corr[self.clear_spec_corr < 0] = 0
            self.dark_corrected_clear = True

        if not self.dark_corrected_plume:
            self.plume_spec_corr = self.plume_spec_raw - self.dark_spec
            self.plume_spec_corr[self.plume_spec_corr < 0] = 0
            self.dark_corrected_plume = True

    def stray_corr_spectra(self):
        """Correct spectra for stray light - spectra are assumed to be dark-corrected prior to running this function"""
        # Set the range of stray pixels
        stray_range = np.arange(self._start_stray_pix, self._end_stray_pix + 1)

        # Correct plume spectra (assumed to already be dark subtracted)
        if not self.stray_corrected_plume:
            self.plume_spec_corr = self.plume_spec_corr - np.mean(self.plume_spec_corr[stray_range])
            self.plume_spec_corr[self.plume_spec_corr < 0] = 0
            self.stray_corrected_plume = True

    def load_ref_spec(self, pathname, species):
        """Load raw reference spectrum"""
        self.ref_spec[species] = np.loadtxt(pathname)

        # Remove un-needed data above 400 nm, which may slow down processing
        idxs = np.where(self.ref_spec[species][:, 0] > 400)
        if len(idxs[0]) > 0:
            self.ref_spec[species] = self.ref_spec[species][:np.min(idxs), :]

        # Assume we have loaded a new spectrum, so set this to False - ILS has not been convolved yet
        self.ref_convolved = False

    def get_ref_spectrum(self):
        """Load in reference spectrum"""
        self.wavelengths = None  # Placeholder for wavelengths attribute which contains all wavelengths of spectra
        #
        # --------------------------------

    def conv_ref_spec(self):
        """
        Convolves reference spectrum with instument line shape (ILS)
        after first interpolating to spectrometer wavelengths
        """
        if self.wavelengths is None:
            print('No wavelength data to perform convolution')
            return

        # Need an odd sized array for convolution, so if even we omit the last pixel
        if self.ILS.size % 2 == 0:
            self.ILS = self.ILS[:-1]

        # Loop through all reference species we have loaded and resample their data to the spectrometers wavelengths
        for f in self.ref_spec_used:
            self.ref_spec_interp[f] = np.interp(self.wavelengths, self.ref_spec[f][:, 0], self.ref_spec[f][:, 1])
            self.ref_spec_conv[f] = convolve(self.ref_spec_interp[f], self.ILS)
            self.ref_spec_ppmm[f] = self.ref_spec_conv[f] * self.ppmm_conversion

        # Update bool as we have now performed this process
        self.ref_convolved = True

    def load_calibration_spectrum(self, pathname):
        """Load Calibation image for spectrometer"""
        pass

    def stretch_spectrum(self, ref_key):
        """Stretch/squeeze reference spectrum to improve fit"""
        if self.stretch == 0:
            # If no stretch, extract reference spectrum using fit window and return
            return self.ref_spec_filter[ref_key][self.fit_window_ref]
        else:
            stretch_inc = (self.stretch * self.stretch_adjuster) / self.stretch_resample  # Stretch increment for resampled spectrum

            if self.stretch < 0:
                # Generate the fit window for extraction
                # Must be larger than fit window as a squeeze requires pulling in data outside of the fit window
                if self.fit_window_ref[-1] < (len(self.wavelengths) - 50):
                    extract_window = np.arange(self.fit_window_ref[0], self.fit_window_ref[-1] + 50)
                else:
                    extract_window = np.arange(self.fit_window_ref[0], len(self.wavelengths))
            else:
                extract_window = self.fit_window_ref

            wavelengths = self.wavelengths[extract_window]
            values = self.ref_spec_filter[ref_key][extract_window]

            # Generate new arrays with 'stretch_resample' more data points
            wavelengths_resampled = np.linspace(wavelengths[0], wavelengths[-1], len(wavelengths)*self.stretch_resample)
            values_resample = np.interp(wavelengths_resampled, wavelengths, values)

            num_pts = len(wavelengths_resampled)
            wavelengths_stretch = np.zeros(num_pts)

            # Stretch wavelengths
            for i in range(num_pts):
                wavelengths_stretch[i] = wavelengths_resampled[i] + (i * stretch_inc)

            values_stretch = np.interp(self.wavelengths[self.fit_window_ref], wavelengths_stretch, values_resample)

            return values_stretch

    def load_spec(self):
        """Load spectrum"""
        pass

    def load_dark(self):
        """Load drk images -> co-add to generate single dark image"""
        pass

    def load_dir(self, spec_dir=None, prompt=True, plot=True):
        """Load spectrum directory
        :param spec_dir str     If provided this is the path to the spectra directory. propmt is ignored if spec_dir is
                                specified.
        :param: prompt  bool    If true, a dialogue box is opened to request directory load. Else, self.spec_dir is used
        :param: plot    bool    If true, the first spectra are plotted in the GUI"""

        if spec_dir is not None:
            self.spec_dir = spec_dir
        elif prompt:
            spec_dir = filedialog.askdirectory(title='Select spectrum sequence directory', initialdir=self.spec_dir)

            if len(spec_dir) > 0 and os.path.exists(spec_dir):
                self.spec_dir = spec_dir
            else:
                raise ValueError('Spectrum directory not recognised: {}'.format(spec_dir))
        else:
            if self.spec_dir is None:
                raise ValueError('Spectrum directory not recognised: {}'.format(self.spec_dir))

        # Update first_spec flag TODO possibly not used in DOASWorker, check
        self.first_spec = True

        # Get list of all files in directory
        self.spec_dict = self.get_spec_list()

        # Set current spectra to first in lists
        if len(self.spec_dict['clear']) > 0:
            self.wavelengths, self.clear_spec_raw = load_spectrum(os.path.join(self.spec_dir,
                                                                               self.spec_dict['clear'][0]))
        if len(self.spec_dict['plume']) > 0:
            self.wavelengths, self.plume_spec_raw = load_spectrum(os.path.join(self.spec_dir,
                                                                               self.spec_dict['plume'][0]))
        if len(self.spec_dict['dark']) > 0:
            ss_id = self.spec_specs.file_ss.replace('{}', '')
            ss = self.spec_dict['plume'][0].split('_')[self.spec_specs.file_ss_loc].replace(ss_id, '')
            self.dark_spec = self.find_dark_spectrum(self.dark_dir, ss)

        # Update plots if requested
        if plot:
            self.fig_spec.update_clear()
            self.fig_spec.update_dark()
            self.fig_spec.update_plume()

    def get_spec_list(self):
        """
        Gets list of spectra files from current spec_dir
        Assumes all .npy files in the directory are spectra...
        :returns: sd    dict    Dictionary containing all filenames
        """
        # Setup empty dictionary sd
        sd = {}

        # Get all files into associated list/dictionary entry
        sd['all'] = [f for f in os.listdir(self.spec_dir) if self.spec_specs.file_ext in f]
        sd['all'].sort()
        sd['plume'] = [f for f in sd['all']
                       if self.spec_specs.file_type['meas'] + self.spec_specs.file_ext in f]
        sd['plume'].sort()
        sd['clear'] = [f for f in sd['all']
                       if self.spec_specs.file_type['clear'] + self.spec_specs.file_ext in f]
        sd['clear'].sort()
        sd['dark'] = [f for f in sd['all']
                      if self.spec_specs.file_type['dark'] + self.spec_specs.file_ext in f]
        sd['dark'].sort()
        return sd

    def find_dark_spectrum(self, spec_dir, ss):
        """
        Searches for suitable dark spectrum in designated directory by finding one with the same shutter speed as
        passed to function.
        :return: dark_spec
        """
        # Ensure ss is a string
        ss = str(ss)

        # Fast dictionary look up for preloaded dark spectra
        if ss in self.dark_dict.keys():
            dark_spec = self.dark_dict[ss]
            return dark_spec

        # List all dark images in directory
        dark_list = [f for f in os.listdir(spec_dir)
                     if self.spec_specs.file_type['dark'] in f and self.spec_specs.file_ext in f]

        # Extract ss from each image and round to 2 significant figures
        ss_str = self.spec_specs.file_ss.replace('{}', '')
        ss_list = [int(f.split('_')[self.spec_specs.file_ss_loc].replace(ss_str, '')) for f in dark_list]

        ss_idx = [i for i, x in enumerate(ss_list) if x == int(ss)]
        ss_spectra = [dark_list[i] for i in ss_idx]

        if len(ss_spectra) < 1:
            return None

        # If we have images, we loop through them to create a coadded image
        dark_full = np.zeros([self.spec_specs.pix_num, len(ss_spectra)])
        for i, ss_spectrum in enumerate(ss_spectra):
            # Load image. Coadd.
            wavelengths, dark_full[:, i] = load_spectrum(os.path.join(spec_dir, ss_spectrum))

        # Coadd images to creat single image
        dark_spec = np.mean(dark_full, axis=1)

        # Update lookup dictionary for fast retrieval of dark image later
        self.dark_dict[ss] = dark_spec

        return dark_spec

    def save_dark(self, filename):
        """Save dark spectrum"""
        if self.wavelengths is None or self.dark_spec is None:
            raise ValueError('One or both attributes are NoneType. Cannot save.')
        if len(self.wavelengths) != len(self.dark_spec):
            raise ValueError('Arrays are not the same length. Cannot save.')

        np.savetxt(filename, np.transpose([self.wavelengths, self.dark_spec]),
                   header='Dark spectrum\nWavelength [nm]\tIntensity [DN]')

    def save_clear_raw(self, filename):
        """Save clear spectrum"""
        if self.wavelengths is None or self.clear_spec_raw is None:
            raise ValueError('One or both attributes are NoneType. Cannot save.')
        if len(self.wavelengths) != len(self.clear_spec_raw):
            raise ValueError('Arrays are not the same length. Cannot save.')

        np.savetxt(filename, np.transpose([self.wavelengths, self.clear_spec_raw]),
                   header='Raw clear (Fraunhofer) spectrum\n'
                          '-Not dark-corrected\nWavelength [nm]\tIntensity [DN]')

    def save_plume_raw(self, filename):
        if self.wavelengths is None or self.plume_spec_raw is None:
            raise ValueError('One or both attributes are NoneType. Cannot save.')
        if len(self.wavelengths) != len(self.plume_spec_raw):
            raise ValueError('Arrays are not the same length. Cannot save.')

        np.savetxt(filename, np.transpose([self.wavelengths, self.plume_spec_raw]),
                   header='Raw in-plume spectrum\n'
                          '-Not dark-corrected\nWavelength [nm]\tIntensity [DN]')

    def gen_abs_specs(self):
        """
        Generate absorbance spectra for individual species and match to measured absorbance
        :return:
        """
        # Iterate through reference spectra
        for spec in self.ref_spec_used:

            # Retrieve all ref spectra other than that held in spec
            other_species = [f for f in self.ref_spec_used if f is not spec]

            # Subtract contributions of these other spectra to absorption
            self.abs_spec_species[spec] = self.abs_spec_cut
            for i in other_species:
                self.abs_spec_species[spec] = self.abs_spec_species[spec] - self.ref_spec_fit[i]

        # Setup total absorbance spectrum as 'Total' key
        self.abs_spec_species['Total'] = self.abs_spec_cut

        # Calculate final residual by subtracting all absorbing species
        self.abs_spec_species['residual'] = self.abs_spec_cut
        for spec in self.ref_spec_used:
            self.abs_spec_species['residual'] = self.abs_spec_species['residual'] - self.ref_spec_fit[spec]

    def process_doas(self):
        """Handles the order of DOAS processing"""
        # Check we have all of the correct spectra to perform processing
        if self.plume_spec_raw is None or self.wavelengths is None:
            raise SpectraError('Require plume spectra for iFit processing')

        if self.ref_spec_types[0] not in self.ref_spec.keys():
            raise SpectraError('No SO2 reference spectrum present for processing')

        # If the stray region hasn't been converted to pixel space we just set it to itself to implement pixel mapping
        if self._start_stray_pix is None or self._end_stray_pix is None:
            self.start_stray_wave = self.start_stray_wave
            self.end_stray_wave = self.end_stray_wave

        # Same for fit window
        if self._start_fit_pix is None or self._end_fit_pix is None:
            self.start_fit_wave = self.start_fit_wave
            self.end_fit_wave = self.end_fit_wave

        if not self.dark_corrected_plume:
            if self.dark_spec is None:
                print('Warning! No dark spectrum present, processing without dark subtraction')

                # Set raw spectra to the corrected spectra, ignoring that they have not been dark corrected
                self.plume_spec_corr = self.plume_spec_raw
            else:
                self.dark_corr_spectra()

        # Correct spectra for stray light
        if not self.stray_corrected_plume:
            self.stray_corr_spectra()

        # Convolve reference spectrum with the instrument lineshape
        # TODO COnfirm this still works - prior to the shift_tol incorporation these lines came AFTER self.set_fit_windows()
        # TODO I think it won't make a difference as they aren't directly related. But I should confirm this
        if not self.ref_convolved:
            self.conv_ref_spec()

        # Run processing
        fit = self.analyser.fit_spectrum([self.wavelengths, self.plume_spec_corr])

        # Unpack results
        for species in self.ref_spec_types:
            self.column_density[species] = fit.meas_od[species]
            self.abs_spec_species[species] = fit.synth_od[species]
        self.abs_spec_species['Total'] = fit.fit
        self.abs_spec_species['residual'] = fit.resid
        self.std_err = np.sqrt(np.mean(np.square(fit.resid)))   # RMSE from residual

        # Set flag defining that data has been fully processed
        self.processed_data = True

    def process_dir(self, spec_dir=None, plot=False):
        """
        Processes the current directory, or the directory specified by spec_dir
        :param spec_dir:
        :return:
        """
        if spec_dir is not None:
            self.load_dir(spec_dir=spec_dir, plot=False)

        cds = []
        times = []

        # Loop through directory plume images processing them
        for i in range(len(self.spec_dict['plume'])):
            self.wavelengths, self.plume_spec_raw = load_spectrum(os.path.join(self.spec_dir,
                                                                               self.spec_dict['plume'][i]))

            # Get dark spectrum
            ss_id = self.spec_specs.file_ss.replace('{}', '')
            ss = self.spec_dict['plume'][i].split('_')[self.spec_specs.file_ss_loc].replace(ss_id, '')
            self.dark_spec = self.find_dark_spectrum(self.dark_dir, ss)

            # Process spectrum
            self.process_doas()

            # Update results
            times.append(self.get_spec_time(self.spec_dict['plume'][i]))
            cds.append(self.column_density['SO2'])

        print(times)
        print(cds)

    def add_doas_results(self, doas_dict):
        """
        Add column densities to DoasResults object, which should be already created
        Also controls removal of old doas points, if we wish to
        :param doas_dict:   dict       Containing at least keys 'column_density' and 'time'
        """
        # If we have a low value we can assume it is in ppm.m (arbitrary deifnition but it should hold for almost every
        # case). So then we need to convert to molecules/cm
        # TODO looks like this function doesn't check if the std_err is present. The KeyError at the bottom I think
        # TODO is wrong as the try clause doesn't have a key. Probably need to catch this in the if statements
        if abs(doas_dict['column_density']) < 100000:
            cd = doas_dict['column_density'] * self.ppmm_conv
            cd_err = doas_dict['std_err'] * self.ppmm_conv
        else:
            cd = doas_dict['column_density']
            cd_err = doas_dict['std_err']

        # Faster append method - seems to work
        self.results.loc[doas_dict['time']] = cd
        try:
            self.results.fit_errs.append(cd_err)
        except KeyError:
            self.results.fit_errs.append(np.nan)

    def rem_doas_results(self, time_obj, inplace=False):
        """
        Remove DOAS result values from time series
        :param time_obj:  datetime.datetime     Any value equal to or earlier than this time will be removed
        :param inplace:     bool
                            If False, the new cut array is returned. If True, the self.results array is overwritten
                            to contain the new cut data
        :return:
        """
        indices = [x for x in self.results.index if x < time_obj]
        fit_err_idxs = self.results.index >= time_obj
        fit_errs = np.array(list(compress(self.results.fit_errs, fit_err_idxs)))
        if inplace:
            self.results.drop(indices, inplace=True)
            self.results.fit_errs = fit_errs
            results = None
        else:
            results = DoasResults(self.results.drop(indices, inplace=False))
            results.fit_errs = fit_errs

        return results

    def make_doas_results(self, times, column_densities, stds=None, species='SO2'):
        """
        Makes pydoas DOASResults object from timeseries
        :param times:   arraylike           Datetimes of column densities
        :param column_densities: arraylike  Column densities
        :param stds:    arraylike           Standard errors in the column density values
        :param species: str                 Gas species
        """
        # If we have a low value we can assume it is in ppm.m (arbitrary deifnition but it should hold for almost every
        # case). So then we need to convert to molecules/cm
        if abs(np.mean(column_densities)) < 100000:
            column_densities = column_densities * self.ppmm_conv
            if stds is not None:
                stds = stds * self.ppmm_conv
        doas_results = DoasResults(column_densities, index=times, fit_errs=stds, species_id=species)
        return doas_results

    def start_processing_threadless(self):
        """
        Process spectra already in a directory, without entering a thread - this means that the _process_loop
        function can update the plots wihtout going through the main thread in PyplisWorker
        """
        # Flag that we are running processing outside of thread
        self.processing_in_thread = False

        # Reset self
        self.reset_self()

        # Get all files
        spec_files = [f for f in os.listdir(self.spec_dir) if '.npy' in f]

        # Sort files alphabetically (which will sort them by time due to file format)
        spec_files.sort()

        # Extract clear spectra if they exist. If not, the first file is assumed to be the clear spectrum
        clear_spec = [f for f in spec_files if self.spec_specs.file_type['clear'] + '.npy' in f]
        plume_spec = [f for f in spec_files if self.spec_specs.file_type['meas'] + '.npy' in f]

        # Loop through all files and add them to queue
        for file in clear_spec:
            self.q_spec.put(self.spec_dir + file)
        for file in plume_spec:
            self.q_spec.put(self.spec_dir + file)

        # Add the exit flag at the end, to ensure that the process_loop doesn't get stuck waiting on the queue forever
        self.q_spec.put('exit')

        # Begin processing
        self._process_loop()

    def start_processing_thread(self):
        """Public access thread starter for _processing"""
        # Reset self
        self.reset_self()

        self.processing_in_thread = True
        self.process_thread = threading.Thread(target=self._process_loop, args=())
        self.process_thread.daemon = True
        self.process_thread.start()

    def _process_loop(self):
        """
        Main process loop for doas
        :return:
        """
        # Setup which we don't need to repeat once in the loop (optimising the code a little)
        ss_str = self.spec_specs.file_ss.replace('{}', '')

        first_spec = True       # First spectrum is used as clear spectrum

        while True:
            # Blocking wait for new file
            pathname = self.q_spec.get(block=True)

            # Close thread if requested with 'exit' command
            if pathname == 'exit':
                break

            # Extract filename and create datetime object of spectrum time
            filename = pathname.split('\\')[-1].split('/')[-1]
            spec_time = self.get_spec_time(filename)

            # Extract shutter speed
            ss_full_str = filename.split('_')[self.spec_specs.file_ss_loc]
            ss = int(ss_full_str.replace(ss_str, ''))

            # Find dark spectrum with same shutter speed
            self.dark_spec = self.find_dark_spectrum(self.dark_dir, ss)

            # Load spectrum
            self.wavelengths, self.plume_spec_raw = load_spectrum(pathname)

            self.process_doas()

            # Gather all relevant information and spectra and pass it to PyplisWorker
            processed_dict = {'processed': True,             # Flag whether this is a complete, processed dictionary
                              'time': spec_time,
                              'filename': pathname,             # Filename of processed spectrum
                              'dark': self.dark_spec,           # Dark spectrum used (raw)
                              'clear': self.clear_spec_raw,     # Clear spectrum used (raw)
                              'plume': self.plume_spec_raw,     # Plume spectrum used (raw)
                              'abs': self.abs_spec_cut,         # Absorption spectrum
                              'ref': self.abs_spec_species,     # Reference spectra (scaled)
                              'std_err': self.std_err,          # Standard error of fit
                              'column_density': self.column_density}    # Column density

            # Update results object
            self.add_doas_results(processed_dict)

            # Pass data dictionary to PyplisWorker queue (might not need to do this if I hold all the data here
            if self.processing_in_thread:
                self.q_doas.put(processed_dict)

            # Or if we are not in a processing thread we update the plots directly here
            # else:
            # TODO check this works, but I think I can update whether processing_in_thread or not - always do it here?
            self.fig_spec.update_dark()
            if first_spec:
                self.fig_spec.update_clear()
            else:
                self.fig_spec.update_plume()

                # Update doas plot
                self.fig_doas.update_plot()

            if first_spec:
                # Now that we have processed first_spec, set flag to False
                first_spec = False

    def start_watching(self, directory):
        """
        Setup directory watcher for images - note this is not for watching spectra - use DOASWorker for that
        Also starts a processing thread, so that the images which arrive can be processed
        """
        if self.watching:
            print('Already watching for spectra: {}'.format(self.watching_dir))
            print('Please stop watcher before attempting to start new watch. '
                  'This isssue may be caused by having manual acquisitions running alongside continuous watching')
            return
        self.watcher = create_dir_watcher(directory, True, self.directory_watch_handler)
        self.watcher.start()
        self.watching_dir = directory
        self.watching = True
        print('Watching {} for new spectra'.format(self.watching_dir[-30:]))

        # Start the processing thread
        self.start_processing_thread()

    def stop_watching(self):
        """Stop directory watcher and end processing thread"""
        self.watcher.stop()
        self.watching = False

        # Stop the processing thread
        self.q_spec.put('exit')

    def directory_watch_handler(self, pathname, t):
        """Handles new spectra passed from watcher"""
        # Pass path to queue
        self.q_spec.put(pathname)

    # =================================================
    # IFIT functions
    # =================================================
    def update_analyser(self):
        """
        Updates ifit analyser
        This should be called everytime the fit window is changed.
        Instrument is stray-light corrected prior to analyser to ignore this flag now
        """
        self.analyser.fit_window = [self.start_fit_wave, self.end_fit_wave]


class SpectraError(Exception):
    """
    Error raised if correct spectra aren't present for processing
    """
    pass


if __name__ == '__main__':
    # Calibration paths
    ils_path = './calibration/2019-07-03_302nm_ILS.txt'
    frs_path = './ifit/Ref/sao2010.txt'
    ref_paths = {'SO2': {'path': './ifit/Ref/SO2_293K.txt', 'value': 1.0e16},  # Value is the inital estimation of CD
                 'O3': {'path': './ifit/Ref/O3_243K.txt', 'value': 1.0e19},
                 'Ring': {'path': './ifit/Ref/Ring.txt', 'value': 0.1}
                 }

    # Spectra path
    spec_path = 'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\Data\\Spectra\\test_data'

    # Create ifit object
    ifit_worker = IFitWorker(frs_path=frs_path, ref_paths=ref_paths, dark_dir=spec_path)
    ifit_worker.load_ils(ils_path)  # Load ILS

    # Process directory
    ifit_worker.process_dir(spec_dir=spec_path)
