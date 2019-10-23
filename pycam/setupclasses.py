# -*- coding: utf-8 -*-

"""
Setup classes for defining default parameters or loading in parameters from files:
> PiCam attributes
> Spectrometer attributes
"""
import warnings
import numpy as np
from .utils import check_filename


class CameraSpecs:
    """Object containing information on camera setup and acquisition settings

    Parameters
    ----------
    filename : str
        path to configuration file (*.txt) to read in camera parameters. If no file is provided the internal defaults
        are use
    """
    def __init__(self, filename=None):
        self.filename = filename    # Filename for loading specifications

        # Hidden variables
        self._bit_depth = None  # Hidden bit depth holder
        self._max_DN = None     # Maximum digital number of images (associated with bit depth)
        self._shutter_speed = None
        self._ss_idx = None

        self.attr_to_io = {'int': ['pix_num_x', 'pix_num_y', '_bit_depth', '_shutter_speed'],
                           'float': ['pix_size_x', 'pix_size_y', 'fov_x', 'fov_y', 'framerate', 'analog_gain',
                                     'min_saturation', 'max_saturation', 'file_ss_units'],
                           'str': ['save_path', 'file_ext', 'file_datestr', 'file_ss'],
                           'dict': ['file_filterids', 'file_img_type'],
                           'bool': ['auto_ss']
                           }


        # Setup default specs to start
        self._default_specs()

        # Load configuration from text file if it is provided, otherwise default parameters are kept
        if isinstance(self.filename, str):
            self.load_specs(self.filename)

    def _default_specs(self):
        """Define camera default specs > binned picam setup"""
        # Camera specs
        self.pix_size_x = 5.6e-6    # Pixel width in m
        self.pix_size_y = 5.6e-6    # Pixel height in m
        self.pix_num_x = 648        # Number of pixels in horizontal
        self.pix_num_y = 486        # Number of pixels in vertical
        self.fov_x = 28             # Field of view in x
        self.fov_y = 24             # FOV in y
        self.bit_depth = 10         # Bit depth of images (set by property

        # Filename info
        self.save_path = '.\\Images\\'
        self.file_ext = '.png'                                  # File extension for images
        self.file_datestr = "%Y-%m-%dT%H%M%S"                   # Date/time format spec in filename
        self.file_filterids = {'on': 'fltrA', 'off': 'fltrB'}   # Filter identifiers in filename
        self.file_ss = '%iss'                                   # Shutter speed format spec
        self.file_ss_units = 1e-6                               # Shutter speed units relative to seconds
        self.file_img_type = {'meas': 'Plume', 'dark': 'Dark', 'cal': 'ppm'}

        # Pre-defined list of shutter speeds (used for auto shutter speed setting)
        self.ss_list = np.concatenate((np.arange(10 ** 3, 10 ** 4, 10 ** 3),
                                       np.arange(10 ** 4, 10 ** 5, 10 ** 4),
                                       np.arange(10 ** 5, 10 ** 6, 5 * 10 ** 4),
                                       np.arange(10 ** 6, 6 * 10 ** 6, 5 * 10 ** 5)))

        # Acquisition settings
        self.shutter_speed = 50000  # Camera shutter speeds (us)
        self.framerate = 0.25       # Camera framerate (Hz)
        self.analog_gain = 1        # Camera analog gain
        self.auto_ss = True         # Bool for requesting automated shutter speed adjustment
        self.min_saturation = 0.4   # Minimum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.max_saturation = 0.9   # Maximum saturation accepted before adjusting shutter speed (if auto_ss is True)

    @property
    def bit_depth(self):
        return self._bit_depth

    @bit_depth.setter
    def bit_depth(self, value):
        """Update _max_DN when bit_depth is defined (two are intrinsically linked)"""
        self._bit_depth = value
        self._max_DN = (2 ** self.bit_depth) - 1

    @property
    def shutter_speed(self):
        return self._shutter_speed

    @shutter_speed.setter
    def shutter_speed(self, value):
        """Update ss_idx to nearest shutter_speed value in ss_list"""
        self._shutter_speed = value
        self._ss_idx = np.argmin(np.abs(self.ss_list - self.shutter_speed))

    @property
    def ss_idx(self):
        return self._ss_idx

    @ss_idx.setter
    def ss_idx(self, value):
        """Update shutter speed to value in ss_list defined by ss_idx when ss_idx is changed
        Accesses hidden variable _shutter_speed directly to avoid causing property method being called"""
        self._ss_idx = value
        self._shutter_speed = self.ss_list[self.ss_idx]

    def extract_info(self, line):
        """Extracts information from line of text based on typical text format for picam files"""
        return line.split('=')[1].split()[0]

    def load_specs(self, filename):
        """Load camera specifications from file

        Parameters
        ----------
        filename : str
            path to configuration (*.txt) file
        """
        # Run check to ensure filename is as expected
        check_filename(filename, 'txt')

        self.filename = filename

        with open(self.filename, 'r') as f:
            # Flag for if we are currently reading in a dictionary
            dict_open = False

            # Loop through every line of the file to check for keywords
            for line in f:

                # Ignore lines beginning with #
                if line[0] == '#':
                    continue

                # Dictionary reading (works for string values only)
                if 'DICT=' in line:

                    # Extract dictionary attribute name and start clean dictionary
                    dict_attr = self.extract_info(line)
                    setattr(self, dict_attr, dict())

                    # Flag that a dictionary is now open
                    dict_open = True

                    continue    # Don't attempt to read any more of line

                if dict_open:

                    # Finish reading of dictionary
                    if 'DICT_END' in line:
                        dict_open = False

                    # Extract dictionary key and set it to the specified value
                    else:
                        vals = line.split('=')
                        getattr(self, dict_attr)[vals[0]] = vals[1].split()[0]

                else:
                    # Extract attribute name
                    attr = line.split('=')[0]

                    # Check name against attributes stored in attr_to_io and correctly assign value
                    # (test for hidden variables too)
                    if attr in self.attr_to_io['int'] or ('_' + attr) in self.attr_to_io['int']:
                        setattr(self, attr, int(self.extract_info(line)))

                    elif attr in self.attr_to_io['float'] or ('_' + attr) in self.attr_to_io['float']:
                        setattr(self, attr, float(self.extract_info(line)))

                    elif attr in self.attr_to_io['str'] or ('_' + attr) in self.attr_to_io['str']:
                        setattr(self, attr, self.extract_info(line))

                    elif attr in self.attr_to_io['bool'] or ('_' + attr) in self.attr_to_io['bool']:
                        val = self.extract_info(line)
                        if val == 'True' or val == 1:
                            setattr(self, attr, True)
                        elif val == 'False' or val == 0:
                            setattr(self, attr, False)
                        else:
                            warnings.warn('Unexpected value for {}: {}. Setting to default {}'
                                          .format(attr, val, getattr(CameraSpecs(), attr)))
                            setattr(self, attr, getattr(CameraSpecs(), attr))

    def save_specs(self, filename):
        """Save camera specifications to file

        Parameters
        ----------
        filename : str
            path to save configuration (*.txt) file
        """
        # Run check to ensure filename is as expected
        check_filename(filename, 'txt')

        self.filename = filename

        with open(filename, 'w') as f:
            # Write header
            f.write('# -*- coding: utf-8 -*-\n')
            f.write('# File holding camera specifications\n')

            # Loop through object attributes and save them if they aren't None
            for attr in self.__dict__:
                if attr in self.attr_to_io['float'] or attr in self.attr_to_io['int'] or attr in self.attr_to_io['str']\
                        or attr in self.attr_to_io['dict'] or attr in self.attr_to_io['bool']:

                    # If we are saving a hidden variable (due to property decorator) remove the preceding underscore
                    if attr[0] == '_':
                        attr = attr[1:]

                    # Get attribute from object
                    attr_val = getattr(self, attr)

                    # Attribute is ignored if set to None
                    if attr_val is None:
                        pass

                    # If the attribute is a dictionary we need to loop through the dict and save each value
                    elif isinstance(attr_val, dict):

                        # Write attribute name to file
                        f.write('DICT={}\n'.format(attr))

                        # Loop through dictionary and keys and values to file
                        for key in attr_val:
                            f.write('{}={}\n'.format(key, attr_val[key]))

                        # End dictionary write with this
                        f.write('DICT_END\n')

                    # Save everything else in simple format
                    else:
                        f.write('{}={}\n'.format(attr, attr_val))


class SpecSpecs:
    """Object containing information on spectrometer setup and acquisition settings

    Parameters
    ----------
    filename : str
        path to configuration file (*.txt) to read in camera parameters. If no file is provided the internal defaults
        are use
    """
    def __init__(self, filename=None):
        self.filename = filename    # Filename for loading specifications

        # Hidden variables
        self._bit_depth = None  # Hidden bit depth holder
        self._max_DN = None     # Maximum digital number of images (associated with bit depth)
        self._int_time_idx = None

        self._default_specs()       # Load default specs to start

        # Load configuration from text file if it is provided
        if isinstance(self.filename, str):
            self.load_specs(self.filename)


    def _default_specs(self):
        """Define spectrometer default specs > Flame-S"""
        # Spectrometer specs
        self.model = "Flame-S"      # Spectrometer model
        self.file_ext = '.spec'
        self.file_spec_type = {'meas': 'Plume', 'dark': 'Dark'}
        self.fov = None             # Field of view fo spectrometer
        self.ILS = None             # Number array holding instrument line shape (possibly don't hold this here?)

        self.pix_num = 2048     # Number of pixels
        self.bit_depth = 16     # Bit depth of spectrometer detector

        # Acquisition settings
        self.start_int_time = 100       # Starting integration time
        self.start_coadd = 1            # Number of spectra to coadd
        self.framerate = 1              # Framerate of acquisitions (Hz)
        self.wavelengths = None         # Wavelengths (nm)
        self.spectrum = None            # Spectrum
        self.spectrum_filename = None   # Filename for spectrum

        self.auto_int = True        # Bool for requesting automated integration time adjustment
        self.min_saturation = 0.4   # Minimum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.max_saturation = 0.9   # Maximum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.saturation_range = [300, 330]  # Range of wavelengths used in checking integration time

        # Predefined list of integration times for automatic exposure adjustment
        self.int_list = np.concatenate((np.arange(1, 10, 1),
                                        np.arange(10, 50, 5),
                                        np.arange(50, 100, 10),
                                        np.arange(100, 500, 50),
                                        np.arange(500, 1000, 100),
                                        np.arange(10 ** 3, 10 ** 4, 500)))

    def load_specs(self, filename):
        """Load spectrometer specifications from file

        Parameters
        ----------
        filename : str
            path to configuration (*.txt) file
        """
        self.filename = filename
        # Add loading functionality here

    def save_specs(self, filename):
        """Save spectrometer specifications to file

        Parameters
        ----------
        filename : str
            path to configuration (*.txt) file
        """
        pass

    @property
    def bit_depth(self):
        return self._bit_depth

    @bit_depth.setter
    def bit_depth(self, value):
        """Update _max_DN when bit_depth is defined (two are intrinsically linked)"""
        self._bit_depth = value
        self._max_DN = (2 ** self.bit_depth) - 1
