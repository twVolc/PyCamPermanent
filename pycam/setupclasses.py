# -*- coding: utf-8 -*-

"""
Setup classes for defining default parameters or loading in parameters from files:
> PiCam attributes
> Spectrometer attributes
"""


class CameraSpecs:
    """
    Object containing information on camera setup and acquisition settings

    Parameters
    ----------
    filename : str
        path to configuration file (*.txt) to read in camera parameters. If no file is provided the internal defaults
        are use
    """
    def __init__(self, filename=None):
        self.filename = filename    # Filename for loading specifications

        # Load configuration from text file if it is provided, otherwise setup default parameters
        if isinstance(self.filename, str):
            self.load_specs(self.filename)
        else:
            self._default_specs()

    def _default_specs(self):
        """
        Define camera default specs > binned picam setup
        """
        # Camera specs
        self.pix_size_x = 5.6e-6    # Pixel width in m
        self.pix_size_y = 5.6e-6    # Pixel height in m
        self.pix_num_x = 648        # Number of pixels in horizontal
        self.pix_num_y = 486        # Number of pixels in vertical
        self.fov_x = 28             # Field of view in x
        self.fov_y = 24             # FOV in y

        # Filename info
        self.file_ext = '.png'                                  # File extension for images
        self.file_datestr = "%Y-%m-%dT%H%M%S"                   # Date/time format spec in filename
        self.file_filterids = {'on': 'fltrA', 'off': 'fltrB'}   # Filter identifiers in filename
        self.file_ss = '%iss'                                   # Shutter speed format spec
        self.file_ssunits = 1e-6                                # Shutter speed units relative to seconds
        self.file_imgtype = {'meas': 'Plume', 'dark': 'Dark', 'cal': 'ppm'}

        # Acquisition settings
        self.shutter_speed = None       # Camera shutter speeds (us)
        self.framerate = None           # Camera framerate (Hz)
        self.analog_gain = 1            # Camera analog gain
        self.auto_ss = True             # Bool for requesting autoated shutter speed adjustment
        self.img = None                 # Image array
        self.img_filename = None            # Image filename

    def load_specs(self, filename):
        """
        Load camera specifications from file

        Parameters
        ----------
        filename : str
            path to configuration (*.txt) file
        """
        self.filename = filename

    def save_specs(self, filename):
        """
            Save camera specifications to file

            Parameters
            ----------
            filename : str
                path to configuration (*.txt) file
        """
        pass


class SpecSpecs:
    """
    Object containing information on spectrometer setup and acquisition settings

    Parameters
    ----------
    filename : str
        path to configuration file (*.txt) to read in camera parameters. If no file is provided the internal defaults
        are use
    """
    def __init__(self, filename=None):
        self.filename = filename    # Filename for loading specifications

        # Load configuration from text file if it is provided, otherwise setup default parameters
        if isinstance(self.filename, str):
            self.load_specs(self.filename)
        else:
            self._default_specs()

    def _default_specs(self):
        """
        Define spectrometer default specs > Flame-S
        """
        # Spectrometer specs
        self.model = "Flame-S"      # Spectrometer model
        self.fov = None             # Field of view fo spectrometer
        self.pix_num = None         # Number of pixels
        self.ILS = None             # Number array holding instrument line shape (possibly don't hold this here?)

        # Acquisition settings
        self.integration_time = None    # Integration time
        self.coadd = None               # Number of spectra to coadd
        self.wavelengths = None         # Wavelengths (nm)
        self.spectrum = None            # Spectrum
        self.spectrum_filename = None   # Filename for spectrum

    def load_specs(self, filename):
        """
            Load spectrometer specifications from file

            Parameters
            ----------
            filename : str
                path to configuration (*.txt) file
        """
        self.filename = filename
        # Add loading functionality here

    def save_specs(self, filename):
        """
            Save spectrometer specifications to file

            Parameters
            ----------
            filename : str
                path to configuration (*.txt) file
        """
        pass




