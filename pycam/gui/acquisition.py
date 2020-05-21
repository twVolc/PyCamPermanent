# -*- coding: utf-8 -*-

"""
Module containing widgets for camera and spectrometer control, by connecting to the instrument via a socket and sneding
messages other comms
"""

import tkinter as tk
import tkinter.ttk as ttk
from pycam.setupclasses import CameraSpecs, SpecSpecs

import pycam.gui.network_cfg as cfg



class TkVariables:
    """
    Sets up a range of tk variables which will be inherited by widget classes
    """

    def __init__(self):
        self._pix_num_x = tk.IntVar()
        self._pix_num_y = tk.IntVar()
        self._fov_x = tk.IntVar()
        self._fov_y = tk.IntVar()
        self._bit_depth = tk.IntVar()
        self._ss_A = tk.IntVar()
        self._ss_B = tk.IntVar()
        self._framerate = tk.DoubleVar()
        self._min_saturation = tk.DoubleVar()
        self._max_saturation = tk.DoubleVar()
        self._saturation_pixels = tk.IntVar()       # Number of pixels to be averaged for saturation check
        self._saturation_rows = tk.IntVar()         # Number of rows to analyse saturation on
        self._saturation_rows_dir = tk.IntVar()     # 1 = top-down, -1 = bottom-up
        self.saturation_rows_dir = 1
        self._wavelength_min = tk.DoubleVar()       # Minimum wavelength for check_saturation range in spectra
        self._wavelength_max = tk.DoubleVar()       # Maximum wavelength for check_saturation range in spectra
        self._coadd = tk.IntVar()
        self._auto_A = tk.IntVar()
        self._auto_B = tk.IntVar()
        self._plume_distance = tk.IntVar()  # NOT SURE I WANT THIS HERE - THIS IS ANALYSIS REALLY

        self.frame_opts = [1.0, 0.5, 0.33, 0.25, 0.2, 0.1]   # Framerate options

    @property
    def pix_num_x(self):
        """Public access attribute to fetch pix_num_x from GUI"""
        return self._pix_num_x.get()

    @pix_num_x.setter
    def pix_num_x(self, value):
        """Public access attribute to fetch pix_num_x from GUI"""
        self._pix_num_x.set(value)

    @property
    def pix_num_y(self):
        """Public access attribute to fetch pix_num_y from GUI"""
        return self._pix_num_y.get()

    @pix_num_y.setter
    def pix_num_y(self, value):
        """Public access attribute to fetch pix_num_y from GUI"""
        self._pix_num_y.set(value)

    @property
    def fov_x(self):
        """Public access attribute to fetch fov_x from GUI"""
        return self._fov_x.get()

    @fov_x.setter
    def fov_x(self, value):
        """Public access attribute to fetch fov_x from GUI"""
        self._fov_x.set(value)

    @property
    def fov_y(self):
        """Public access attribute to fetch fov_y from GUI"""
        return self._fov_y.get()

    @fov_y.setter
    def fov_y(self, value):
        """Public access attribute to fetch fov_y from GUI"""
        self._fov_y.set(value)

    @property
    def bit_depth(self):
        """Public access attribute to fetch bit_depth from GUI"""
        return self._bit_depth.get()

    @bit_depth.setter
    def bit_depth(self, value):
        """Public access attribute to fetch bit_depth from GUI"""
        self._bit_depth.set(value)

    @property
    def ss_A(self):
        """ss is a public attribute which fetches shutter speed from the GUI"""
        return self._ss_A.get()

    @ss_A.setter
    def ss_A(self, value):
        """Allow the tk var to be set via public access of ss"""
        self._ss_A.set(value)

    @property
    def ss_B(self):
        """ss is a public attribute which fetches shutter speed from the GUI"""
        return self._ss_B.get()

    @ss_B.setter
    def ss_B(self, value):
        """Allow the tk var to be set via public access of ss"""
        self._ss_B.set(value)

    @property
    def framerate(self):
        """framerate is a public attribute which fetches framerate from the GUI"""
        return self._framerate.get()

    @framerate.setter
    def framerate(self, value):
        """Allow the tk var to be set via public access of framerate"""
        self._framerate.set(value)

    @property
    def min_saturation(self):
        """Public access attribute to fetch min_saturation from GUI"""
        return self._min_saturation.get()

    @min_saturation.setter
    def min_saturation(self, value):
        """Public access setter of min_saturation"""
        self._min_saturation.set(value)

    @property
    def max_saturation(self):
        """Public access attribute to fetch max_saturation from GUI"""
        return self._max_saturation.get()

    @max_saturation.setter
    def max_saturation(self, value):
        """Public access setter of max_saturation"""
        self._max_saturation.set(value)

    @property
    def saturation_pixels(self):
        """Public access attribute to fetch saturation_pixels from GUI"""
        return self._saturation_pixels.get()

    @saturation_pixels.setter
    def saturation_pixels(self, value):
        """Public access setter of saturation_pixels"""
        self._saturation_pixels.set(value)

    @property
    def saturation_rows(self):
        return self._saturation_rows.get() * self.saturation_rows_dir

    @saturation_rows.setter
    def saturation_rows(self, value):
        self._saturation_rows.set(value)

    @property
    def saturation_rows_dir(self):
        return self._saturation_rows_dir.get()

    @saturation_rows_dir.setter
    def saturation_rows_dir(self, value):
        self._saturation_rows_dir.set(value)

    @property
    def wavelength_min(self):
        """Public access attribute to fetch wavelength_min from GUI"""
        return self._wavelength_min.get()

    @wavelength_min.setter
    def wavelength_min(self, value):
        """Public access setter of wavelength_min"""
        self._wavelength_min.set(value)

    @property
    def wavelength_max(self):
        """Public access attribute to fetch wavelength_max from GUI"""
        return self._wavelength_max.get()

    @wavelength_max.setter
    def wavelength_max(self, value):
        """Public access setter of wavelength_max"""
        self._wavelength_max.set(value)

    @property
    def coadd(self):
        """Public access attribute to fetch coadd from GUI"""
        return self._coadd.get()

    @coadd.setter
    def coadd(self, value):
        """Public access setter of coadd"""
        self._coadd.set(value)

    @property
    def auto_A(self):
        """Public access attribute to fetch auto_A from GUI"""
        return self._auto_A.get()

    @auto_A.setter
    def auto_A(self, value):
        """Public access setter of auto_A"""
        self._auto_A.set(value)

    @property
    def auto_B(self):
        """Public access attribute to fetch auto_B from GUI"""
        return self._auto_B.get()

    @auto_B.setter
    def auto_B(self, value):
        """Public access setter of auto_B"""
        self._auto_B.set(value)

    @property
    def plume_distance(self):
        """Public access attribute to fetch plume_distance from GUI"""
        return self._plume_distance.get()

    @plume_distance.setter
    def plume_distance(self, value):
        """Public access setter of plume_distance"""
        self._plume_distance.set(value)

    def set_cam_defaults(self):
        """Sets up camera defaults using CameraSpecs defaults"""
        cam = CameraSpecs()

        self.auto_A = cam.auto_ss
        self.auto_B = cam.auto_ss
        self.framerate = cam.framerate
        self.min_saturation = cam.min_saturation
        self.max_saturation = cam.max_saturation
        self.saturation_rows = cam.saturation_rows
        self.saturation_pixels = cam.saturation_pixels
        self.pix_num_x = cam.pix_num_x
        self.pix_num_y = cam.pix_num_y
        self.fov_x = cam.fov_x
        self.fov_y = cam.fov_y
        self.bit_depth = cam.bit_depth

    def set_spec_defaults(self):
        """Sets up camera default using SpecSpecs defaults"""
        spec = SpecSpecs()

        self.pix_num_x = spec.pix_num
        self.framerate = spec.framerate
        self.min_saturation = spec.min_saturation
        self.max_saturation = spec.max_saturation
        self.wavelength_min = spec.saturation_range[0]
        self.wavelength_max = spec.saturation_range[1]
        self.saturation_pixels = spec.saturation_pixels
        self.fov_x = spec.fov
        self.bit_depth = spec.bit_depth


class CameraSettingsWidget(TkVariables):
    """
    Hold all widgets required for communicating with the camera

    Parameters
    ----------
    parent: tk.Frame, ttk.Frame
        Parent frame that widget will be placed into
    """
    def __init__(self, parent):
        super().__init__()

        self.parent = parent    # Parent frame
        self.pdx = 2
        self.pdy = 2
        self.frame = ttk.LabelFrame(self.parent, text='Camera Settings', relief=tk.RAISED)  # Main frame

        # Setup camera defaults
        self.set_cam_defaults()

        # Dictionary of all commands and associated attribute names in CameraSettings widget
        self.cmd_dict = {'SSA': 'ss_A',
                         'SSB': 'ss_B',
                         'FRC': 'framerate',
                         'ATA': 'auto_B',
                         'ATB': 'auto_A',
                         'SMN': 'min_saturation',
                         'SMX': 'max_saturation',
                         'PXC': 'saturation_pixels',
                         'RWC': 'saturation_rows'
                         }

        # --------------------------------------------------------------------------------------------------------------
        # GUI layout
        # --------------------------------------------------------------------------------------------------------------
        row = 0

        ttk.Label(self.frame, text='Framerate (Hz):').grid(row=row, column=0, padx=5, pady=5, sticky='e')
        option_menu = ttk.OptionMenu(self.frame, self._framerate, self.frame_opts[3], *self.frame_opts)
        option_menu.config(width=4)
        option_menu.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
        row += 1

        # -------------------------
        # Setup shutter speed frame
        # -------------------------
        self.ss_frame = ttk.LabelFrame(self.frame, text=u'Shutter speed (\u03bcs)', relief=tk.GROOVE)
        self.ss_frame.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        ttk.Label(self.ss_frame, text='Filter A:').grid(row=0, column=0, padx=self.pdx, pady=self.pdy)
        ttk.Label(self.ss_frame, text='Filter B:').grid(row=1, column=0, padx=self.pdx, pady=self.pdy)
        ttk.Entry(self.ss_frame, width=7, textvariable=self._ss_A).grid(row=0, column=1, padx=self.pdx, pady=self.pdy)
        ttk.Entry(self.ss_frame, width=7, textvariable=self._ss_B).grid(row=1, column=1, padx=self.pdx, pady=self.pdy)
        ttk.Checkbutton(self.ss_frame, text='Auto', variable=self._auto_A).grid(row=0, column=2,
                                                                                padx=self.pdx, pady=self.pdy)
        ttk.Checkbutton(self.ss_frame, text='Auto', variable=self._auto_B).grid(row=1, column=2,
                                                                                padx=self.pdx, pady=self.pdy)
        row += 1

        # ----------------------
        # Saturation level frame
        # ----------------------
        self.sat_frame = ttk.LabelFrame(self.frame, text='Saturation levels', relief=tk.GROOVE)
        self.sat_frame.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        ttk.Label(self.sat_frame, text='Minimum saturation:').grid(row=0, column=0, padx=self.pdx, pady=self.pdy,
                                                                   sticky='e')
        ttk.Label(self.sat_frame, text='Maximum saturation:').grid(row=1, column=0, padx=self.pdx, pady=self.pdy,
                                                                   sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=4, format='%.2f', textvariable=self._min_saturation,
                        from_=0, to=1, increment=0.01)
        s.grid(row=0, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.2f}'.format(self.min_saturation))     # Set intial value to have the right format
        s = ttk.Spinbox(self.sat_frame, width=4, textvariable=self._max_saturation, from_=0.00, to=1.00, increment=0.01)
        s.grid(row=1, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.2f}'.format(self.max_saturation))  # Set intial value to have the right format

        ttk.Label(self.sat_frame, text='Average pixels:').grid(row=2, column=0, padx=self.pdx, pady=self.pdy,sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=4, textvariable=self._saturation_pixels, from_=1, to=9999, increment=1)
        s.grid(row=2, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')

        ttk.Label(self.sat_frame, text='Number of rows:').grid(row=3, column=0, padx=self.pdx, pady=self.pdy,sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=4, textvariable=self._saturation_rows,
                        from_=0, to=self.pix_num_y, increment=1)
        s.grid(row=3, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')

        ttk.Label(self.sat_frame, text='Row direction:').grid(row=4, column=0, padx=self.pdx, pady=self.pdy, sticky='e')
        r = ttk.Radiobutton(self.sat_frame, text='Top-down', variable=self._saturation_rows_dir, value=1)
        r.grid(row=4, column=1, padx=self.pdx, pady=self.pdy, sticky='w')
        r = ttk.Radiobutton(self.sat_frame, text='Bottom-up', variable=self._saturation_rows_dir, value=-1)
        r.grid(row=5, column=1, padx=self.pdx, pady=self.pdy, sticky='w')
        row += 1

        # ----------------
        # Image parameters
        # ----------------
        self.img_param_frame = ttk.LabelFrame(self.frame, text='Imaging Parameters', relief=tk.GROOVE)
        self.img_param_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        ttk.Label(self.img_param_frame, text='Image resolution:').grid(row=0, column=0, padx=5, pady=5, sticky='e')
        ttk.Label(self.img_param_frame, text='x').grid(row=0, column=1, sticky='e')
        e = ttk.Entry(self.img_param_frame, textvariable=self._pix_num_x, width=4)
        e.grid(row=0, column=2, sticky='w')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.img_param_frame, text='  y').grid(row=0, column=3, sticky='e')
        e = ttk.Entry(self.img_param_frame, textvariable=self._pix_num_y, width=4)
        e.grid(row=0, column=4, sticky='e')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.img_param_frame, text='Field of view [°]:').grid(row=1, column=0, padx=5, pady=5, sticky='e')
        ttk.Label(self.img_param_frame, text='x').grid(row=1, column=1, sticky='e')
        e = ttk.Entry(self.img_param_frame, textvariable=self._fov_x, width=4)
        e.grid(row=1, column=2, sticky='w')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.img_param_frame, text='  y').grid(row=1, column=3, sticky='e')
        e = ttk.Entry(self.img_param_frame, textvariable=self._fov_y, width=4)
        e.grid(row=1, column=4, sticky='e')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.img_param_frame, text='Bit depth:').grid(row=2, column=0, padx=5, pady=5, sticky='e')
        e = ttk.Entry(self.img_param_frame, textvariable=self._bit_depth, width=4)
        e.grid(row=2, column=1, columnspan=2, sticky='ew', pady=5)
        e.configure(state=tk.DISABLED)

        row += 1


class SpectrometerSettingsWidget(TkVariables):
    """
    Hold all widgets required for spectrometer settings

    Parameters
    ----------
    parent: tk.Frame, ttk.Frame
        Parent frame that widget will be placed into
    """
    def __init__(self, parent, sock=None):
        super().__init__()

        self.parent = parent  # Parent frame
        self.pdx = 2
        self.pdy = 2
        self.frame = ttk.LabelFrame(self.parent, text='Spectrometer Settings', relief=tk.RAISED)  # Main frame

        # Setup camera defaults
        self.set_spec_defaults()

        # --------------------------------------------------------------------------------------------------------------
        # GUI layout
        # --------------------------------------------------------------------------------------------------------------
        row = 0

        # Framerate
        ttk.Label(self.frame, text='Framerate (Hz):').grid(row=row, column=0, padx=5, pady=5, sticky='e')
        option_menu = ttk.OptionMenu(self.frame, self._framerate, self.frame_opts[3], *self.frame_opts)
        option_menu.config(width=4)
        option_menu.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
        row += 1

        ttk.Label(self.frame, text='Coadd spectra:').grid(row=row, column=0, padx=self.pdx, pady=self.pdy, sticky='e')
        s = ttk.Spinbox(self.frame, width=4, textvariable=self._coadd, from_=0, to=20, increment=1)
        s.grid(row=row, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set(self.coadd)  # Set intial value 
        row += 1

        # Shutter speed
        self.ss_frame = ttk.LabelFrame(self.frame, text='Shutter speed (ms):')
        self.ss_frame.grid(row=row, column=0, columnspan=2, padx=self.pdx, pady=self.pdy, sticky='nsew')
        self.ss = ttk.Entry(self.ss_frame, width=5, textvariable=self._ss_A).grid(row=0, column=0, padx=self.pdx, pady=self.pdy, sticky='ew')
        ttk.Checkbutton(self.ss_frame, text='Auto', variable=self._auto_A).grid(row=0, column=1, padx=self.pdx, pady=self.pdy)
        row += 1

        # ----------------------
        # Saturation level frame
        # ----------------------
        self.sat_frame = ttk.LabelFrame(self.frame, text='Saturation levels', relief=tk.GROOVE)
        self.sat_frame.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        ttk.Label(self.sat_frame, text='Minimum saturation:').grid(row=0, column=0, padx=self.pdx, pady=self.pdy,
                                                                   sticky='e')
        ttk.Label(self.sat_frame, text='Maximum saturation:').grid(row=1, column=0, padx=self.pdx, pady=self.pdy,
                                                                   sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=4, format='%.2f', textvariable=self._min_saturation,
                        from_=0, to=1, increment=0.01)
        s.grid(row=0, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.2f}'.format(self.min_saturation))  # Set intial value to have the right format
        s = ttk.Spinbox(self.sat_frame, width=4, textvariable=self._max_saturation, from_=0.00, to=1.00, increment=0.01)
        s.grid(row=1, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.2f}'.format(self.max_saturation))  # Set intial value to have the right format

        ttk.Label(self.sat_frame, text='Min. wavelength [nm]:').grid(row=2, column=0, padx=self.pdx, pady=self.pdy,
                                                                     sticky='e')
        ttk.Label(self.sat_frame, text='Max. wavelength [nm]:').grid(row=3, column=0, padx=self.pdx, pady=self.pdy,
                                                                     sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=5, format='%.1f', textvariable=self._wavelength_min,
                        from_=300, to=350, increment=0.1)
        s.grid(row=2, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.1f}'.format(self.wavelength_min))  # Set intial value to have the right format
        s = ttk.Spinbox(self.sat_frame, width=5, textvariable=self._wavelength_max, from_=300, to=400, increment=0.1)
        s.grid(row=3, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        s.set('{:.1f}'.format(self.wavelength_max))  # Set intial value to have the right format

        ttk.Label(self.sat_frame, text='Average pixels:').grid(row=4, column=0, padx=self.pdx, pady=self.pdy,
                                                               sticky='e')
        s = ttk.Spinbox(self.sat_frame, width=4, textvariable=self._saturation_pixels, from_=1, to=9999, increment=1)
        s.grid(row=4, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')

        row += 1

        # ----------------
        # Image parameters
        # ----------------
        self.spec_param_frame = ttk.LabelFrame(self.frame, text='Spectrum Parameters', relief=tk.GROOVE)
        self.spec_param_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        ttk.Label(self.spec_param_frame, text='Detector pixels:').grid(row=0, column=0, padx=5, pady=5, sticky='e')
        e = ttk.Entry(self.spec_param_frame, textvariable=self._pix_num_x, width=4)
        e.grid(row=0, column=1, sticky='ew')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.spec_param_frame, text='Field of view [°]:').grid(row=1, column=0, padx=5, pady=5, sticky='e')
        e = ttk.Entry(self.spec_param_frame, textvariable=self._fov_x, width=4)
        e.grid(row=1, column=1, sticky='ew')
        e.configure(state=tk.DISABLED)

        ttk.Label(self.spec_param_frame, text='Bit depth:').grid(row=2, column=0, padx=5, pady=5, sticky='e')
        e = ttk.Entry(self.spec_param_frame, textvariable=self._bit_depth, width=4)
        e.grid(row=2, column=1, sticky='ew')
        e.configure(state=tk.DISABLED)


class CommHandler:
    """
    Handles communication backend between GUI settings and socket connection

    Parameters
    ----------
    cam: CameraSettingsWidget
        Frontend class for camera acquisition settings
    spec: SpectrometerSettingsWidget
        Frontend class for spectrometer acquisition settings
    """
    def __init__(self, cam, spec, frame):
        self.cam = cam
        self.spec = spec
        self.frame = frame

        self.acq_button = ttk.Button(self.frame, text='Update camera', command=self.acq_comm)

    def acq_comm(self):
        """Performs the acquire command, sending all relevant"""




