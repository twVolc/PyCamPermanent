# -*- coding: utf-8 -*-

"""
Setup classes for defining default parameters or loading in parameters from files:
> PiCam attributes
> Spectrometer attributes
"""
import warnings
import numpy as np
from .utils import check_filename


pycam_details = {
    'version': '0.1.0',
    'date': '26 February 2020'
    }


class FileLocator:
    """Defines locations of important files"""
    PYCAM_ROOT_PI = '/home/pi/pycam'                               # Root to pycam on Pis from home directory
    PYCAM_ROOT_WINDOWS = 'C:/Users/tw9616/Documents/PostDoc/Permanent Camera/PyCamPermanent/pycam'  # Root to pycam on Windows machine

    CONF_DIR = PYCAM_ROOT_PI + '/conf/'                         # Directory holding configuration files
    CONF_DIR_WINDOWS = PYCAM_ROOT_WINDOWS + '/conf/'            # Directory holding configuration files on Windows

    CONFIG = CONF_DIR + 'config.txt'                            # Main configuration file
    CONFIG_WINDOWS = CONF_DIR_WINDOWS + 'config.txt'            # Main configuration file on Windows machine
    CONFIG_CAM = CONF_DIR + 'cam_specs.txt'                     # Camera configuration file
    CONFIG_SPEC = CONF_DIR + 'spec_specs.txt'                   # Spectrometer configuration file
    SCHEDULE_FILE = CONF_DIR_WINDOWS + 'witty_schedule.wpi'     # Schedule script for local storage
    SCHEDULE_FILE_PI = '/home/pi/wittypi/schedule.wpi'          # Schedule script on pi
    SCRIPT_SCHEDULE_FILE = 'script_schedule.txt'                # Filename for start/stop pycam script schedule
    CRON_PYCAM = CONF_DIR_WINDOWS + 'cron/'
    CRON_PYCAM_PI = CONF_DIR + 'cron/'
    SCRIPT_SCHEDULE = CRON_PYCAM + SCRIPT_SCHEDULE_FILE         # Script schedule for starting/stopping software
    SCRIPT_SCHEDULE_PI = CRON_PYCAM_PI + SCRIPT_SCHEDULE_FILE
    CRON_DIR = '/etc/cron.d/'
    CRON_FILE = CRON_DIR + 'script_schedule'                    # Location for start/stop pycam in crontab

    NET_COMM_FILE = CONF_DIR + 'network_comm.txt'               # Network file for communicating acquisitions
    NET_TRANSFER_FILE = CONF_DIR + 'network_transfer.txt'       # Network file for transferring data

    IMG_SPEC_PATH = PYCAM_ROOT_PI + '/Images/'                  # Image and spectra path on main Pi
    IMG_SPEC_PATH_WINDOWS = PYCAM_ROOT_WINDOWS + '/Images/'     # Image and spectra path on Windows machine

    SCRIPTS = PYCAM_ROOT_PI + '/scripts/'

    LOG_PATH = PYCAM_ROOT_PI + '/logs/'
    ERROR_LOG = LOG_PATH + 'error.log'

    # GUI
    GUI_SETTINGS = './gui_settings.txt'
    GUI_SETTINGS_DEFAULT = './gui_settings_default.txt'

    CAM_GEOM = CONF_DIR_WINDOWS + '/cam_geom/'                  # Camera geometry settings directory
    DEFAULT_GEOM = CAM_GEOM + 'default.txt'                     # File containing default geometry file

    PROCESS_DEFAULTS = CONF_DIR_WINDOWS + 'processing_setting_defaults.txt'

    # Data
    DATA = PYCAM_ROOT_WINDOWS + '/Data/'
    SAVED_OBJECTS = DATA + 'saved_objects/'
    PCS_LINES = SAVED_OBJECTS + 'pcs_lines/'


class ConfigInfo:
    """Defines important attributes related to config files, allowing references to link to this file rather than being
    explicitly coded elsewhere"""
    pi_ip = 'pi_ip'                 # Tag for remote pi ip addresses in config file
    host_ip = 'host_ip'
    local_ip = 'local_ip'           # Tag for local ip address in config file
    port_transfer = 'port_transfer'
    port_comm = 'port_comm'
    port_ext = 'port_ext'

    start_script = 'start_script'
    stop_script = 'stop_script'
    master_script = 'master_script'
    remote_scripts = 'remote_scripts'
    local_scripts = 'local_scripts'
    cam_script = 'cam_script'       # Tag for defining camera script to be run on pi
    spec_script = 'spec_script'
    cam_specs = 'cam_specs'
    temp_log = 'temp_log'

    local_data_dir = 'local_data_dir'


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
                           'dict': ['file_filterids', 'file_type'],
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
        self.fov_y = 22             # FOV in y
        self.bit_depth = 10         # Bit depth of images (set by property

        # Filename info
        self.save_path = '.\\Images\\'
        self.file_ext = '.png'                                  # File extension for images
        self.file_datestr = "%Y-%m-%dT%H%M%S"                   # Date/time format spec in filename
        self.file_filterids = {'on': 'fltrA', 'off': 'fltrB'}   # Filter identifiers in filename
        self.file_ag = '{}ag'                                   # Analog gain format spec
        self.file_ss = '{}ss'                                   # Shutter speed format spec
        self.file_ss_units = 1e-6                               # Shutter speed units relative to seconds
        self.file_type = {'meas': 'Plume',
                          'dark': 'Dark',
                          'cal': 'ppmm',
                          'clear': 'Clear',
                          'test': 'Test',
                          'dark_corr': 'darkcorr'}
        self.file_date_loc = 0      # Location of date string in filename
        self.file_fltr_loc = 1      # Location of filter string
        self.file_gain_loc = 2      # Location of gain string
        self.file_ss_loc = 3        # Shutter speed location in filename
        self.file_type_loc = 4      # Img type location in filename

        # Pre-defined list of shutter speeds (used for auto shutter speed setting)
        self.ss_list = np.concatenate((np.arange(1000, 5000, 500),
                                       np.arange(5000, 10000, 1000),
                                       np.arange(10000, 50000, 5000),
                                       np.arange(50000, 100000, 10000),
                                       np.arange(100000, 500000, 50000),
                                       np.arange(500000, 1000000, 100000),
                                       np.arange(1000000, 6000000, 500000), [6000]))

        # Acquisition settings
        self.shutter_speed = 10000  # Camera shutter speeds (us)
        self.framerate = 0.25       # Camera framerate (Hz)
        self.analog_gain = 1        # Camera analog gain
        self.auto_ss = True         # Bool for requesting automated shutter speed adjustment
        self.min_saturation = 0.7   # Minimum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.max_saturation = 0.9   # Maximum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.saturation_pixels = 100  # Number of pixels checked for saturation
        self.saturation_rows = int(self.pix_num_y / 2)   # rows to extract for saturation check (don't want to check lower rows as snow may be present (if negative, rows start from bottom and work up, postive -top-down)

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

    def estimate_focal_length(self):
        """
        Calculates focal length from FOV and detector dimensions
        Returns: focal length (m)
        """
        fl_x = ((self.pix_num_x * self.pix_size_x) / 2) / np.tan(np.deg2rad(self.fov_x / 2))
        fl_y = ((self.pix_num_y * self.pix_size_y) / 2) / np.tan(np.deg2rad(self.fov_y / 2))

        # Check focal lengths calculated from 2 dimensions are roughly equal (within 5%)
        if fl_x < 0.95 * fl_y or fl_x > 1.05 * fl_y:
            raise ValueError('Focal lengths calculated from x and y dimensions do not agree within a reasonable range')
        else:
            # Calculate average focal length and return
            fl = np.mean([fl_x, fl_y])
            return fl

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
                        if val == 'True' or val == '1':
                            setattr(self, attr, True)
                        elif val == 'False' or val == '0':
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
        self.fov = 1                # Field of view fo spectrometer (radius of FOV)
        self.ILS = None             # Number array holding instrument line shape (possibly don't hold this here?)
        self.fiber_diameter = 1e-3  # Diameter of optical fiber
        self.pix_num = 2048         # Number of pixels
        self.bit_depth = 16         # Bit depth of spectrometer detector

        # File information
        self.file_ext = '.npy'  # Spectra saved as numpy array
        self.file_ss = '{}ss'   # Shutter speed format spec
        self.file_type = {'meas': 'Plume', 'dark': 'Dark', 'cal': 'ppmm', 'clear': 'Clear', 'test': 'Test'}
        self.file_datestr = "%Y-%m-%dT%H%M%S"                   # Date/time format spec in filename
        self.file_date_loc = 0
        self.file_ss_loc = 1        # Shutter speed location in filename
        self.file_coadd_loc = 2     # Coadd location in filename
        self.file_type_loc = 3      # File type location in filename


        # Acquisition settings
        self.start_int_time = 100       # Starting integration time
        self.start_coadd = 1            # Number of spectra to coadd
        self.framerate = 1            # Framerate of acquisitions (Hz)
        self.wavelengths = None         # Wavelengths (nm)
        self.spectrum = None            # Spectrum
        self.spectrum_filename = None   # Filename for spectrum

        self.auto_int = True        # Bool for requesting automated integration time adjustment
        self.min_saturation = 0.6   # Minimum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.max_saturation = 0.9   # Maximum saturation accepted before adjusting shutter speed (if auto_ss is True)
        self.saturation_range = [300, 335]  # Range of wavelengths used in checking integration time
        self.saturation_pixels = 2  # Number of pixels to check

        # Predefined list of integration times for automatic exposure adjustment
        self.int_list = np.concatenate((np.arange(0.1, 0.5, 0.05),
                                        np.arange(0.5, 1, 0.1),
                                        np.arange(1, 5, 0.5),
                                        np.arange(5, 10, 1),
                                        np.arange(10, 50, 5),
                                        np.arange(50, 100, 10),
                                        np.arange(100, 500, 50),
                                        np.arange(500, 1000, 100),
                                        np.arange(10 ** 3, 10 ** 4, 500),
                                        np.array([10 ** 4])))

    def estimate_focal_length(self):
        """
        Calculates focal length assuming a single round fiber of defined dimensions
        Returns: focal length (m)
        """
        fl = (self.fiber_diameter / 2) / np.tan(np.deg2rad(self.fov / 2))

        return fl

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
