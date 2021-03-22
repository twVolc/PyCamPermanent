# -*- coding: utf-8 -*-

"""
Contains all classes associated with building figures for the analysis functions of SO2 cameras
"""

from pycam.gui.cfg import gui_setts, fig_face_colour, axes_colour
from pycam.gui.misc import SpinboxOpt, LoadSaveProcessingSettings
from pycam.setupclasses import CameraSpecs, SpecSpecs, FileLocator
from pycam.cfg import pyplis_worker
from pycam.doas.cfg import doas_worker
from pycam.doas.ifit_worker import IFitWorker
from pycam.so2_camera_processor import UnrecognisedSourceError
from pycam.utils import make_circular_mask_line
from pycam.io_py import save_pcs_line, load_pcs_line

from pyplis import LineOnImage, Img
from pyplis.helpers import make_circular_mask, shifted_color_map
from geonum import GeoPoint

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
import matplotlib.widgets as widgets
import matplotlib.patches as patches
import matplotlib.lines as mpllines
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import time
import queue
import threading
from pandas import Series

refresh_rate = 200    # Refresh rate of draw command when in processing thread


class SequenceInfo:
    """
    Generates widget containing squence information, which is displayed at the top of the analysis frame
    """
    def __init__(self, parent, pyplis_work=pyplis_worker, generate_widget=True):
        self.parent = parent
        self.frame = ttk.LabelFrame(self.parent, text='Sequence information')
        self.pyplis_worker = pyplis_worker
        self.pyplis_worker.seq_info = self
        self.date_fmt = '%d-%m-%Y'
        self.time_fmt = '%H:%M:%S'

        self.path_str_length = 50

        self.pdx = 2
        self.pdy = 2

        if generate_widget:
            self.initiate_variables()
            self.generate_widget()

    def initiate_variables(self):
        """Setup tk variables"""
        self._img_dir = tk.StringVar()
        self._num_img_pairs = tk.IntVar()
        self._num_img_tot = tk.IntVar()

        self.img_dir = self.pyplis_worker.img_dir
        self.num_img_tot = self.pyplis_worker.num_img_tot
        self.num_img_pairs = self.pyplis_worker.num_img_pairs
        self.date = '-'
        self.start_time = '-'
        self.end_time = '-'

    def generate_widget(self):
        """Builds widget"""
        row = 0
        label = ttk.Label(self.frame, text='Sequence directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.img_dir_lab = ttk.Label(self.frame, text=self.img_dir_short)
        self.img_dir_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

        row += 1
        label = ttk.Label(self.frame, text='Date:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.date_lab = ttk.Label(self.frame, text=self.date)
        self.date_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

        row += 1
        label = ttk.Label(self.frame, text='Time:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.time_lab = ttk.Label(self.frame, text='{} - {}'.format(self.start_time, self.end_time))
        self.time_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

        row += 1
        label = ttk.Label(self.frame, text='Total images:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.num_img_tot_lab = ttk.Label(self.frame, text=str(self.num_img_tot))
        self.num_img_tot_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

        row += 1
        label = ttk.Label(self.frame, text='Image pairs:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.num_img_pairs_lab = ttk.Label(self.frame, text=str(self.num_img_pairs))
        self.num_img_pairs_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

    @property
    def img_dir(self):
        return self._img_dir.get()

    @img_dir.setter
    def img_dir(self, value):
        self._img_dir.set(value)

    @property
    def img_dir_short(self):
        return '...' + self.img_dir[-self.path_str_length:]

    @property
    def num_img_pairs(self):
        return self._num_img_pairs.get()

    @num_img_pairs.setter
    def num_img_pairs(self, value):
        self._num_img_pairs.set(value)

    @property
    def num_img_tot(self):
        return self._num_img_tot.get()

    @num_img_tot.setter
    def num_img_tot(self, value):
        self._num_img_tot.set(value)

    def update_variables(self):
        """Updates image list variables"""
        self.img_dir = self.pyplis_worker.img_dir
        self.num_img_pairs = self.pyplis_worker.num_img_pairs
        self.num_img_tot = self.pyplis_worker.num_img_tot
        self.date = self.pyplis_worker.time_range[0].strftime(self.date_fmt)
        self.start_time = self.pyplis_worker.time_range[0].strftime(self.time_fmt)
        self.end_time = self.pyplis_worker.time_range[-1].strftime(self.time_fmt)

        self.img_dir_lab.configure(text=self.img_dir_short)
        self.num_img_pairs_lab.configure(text=str(self.num_img_pairs))
        self.num_img_tot_lab.configure(text=str(self.num_img_tot))
        self.date_lab.configure(text=self.date)
        self.time_lab.configure(text='{} - {}'.format(self.start_time, self.end_time))


class ImageSO2(LoadSaveProcessingSettings):
    """
    Main class for generating an image of SO2. It may be calibated [ppm.m] or uncalibrated [apparent absorbance]
    depending on the user's command

    Parameters
    ----------
    parent: tk.Frame, ttk.Frame
        Frame the widget will be placed inside
    image: np.ndarray
        Image array for plotting
    pix_dim: tuple
        [x_dimension, y_dimension] list of resolution for SO2 camera imagery
    """

    def __init__(self, parent, pyplis_work=pyplis_worker, image_tau=None, image_cal=None,
                 pix_dim=(CameraSpecs().pix_num_x, CameraSpecs().pix_num_y)):
        super().__init__()

        # Get root - used for plotting using refresh after in _draw_canv_()
        parent_name = parent.winfo_parent()
        self.root = parent._nametowidget(parent_name)

        self.parent = parent
        self.image_tau = image_tau
        self.image_cal = image_cal
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_tau = self
        self.pyplis_worker.fig_opt.fig_SO2 = self
        self.fig_series = None

        self.pix_num_x = pix_dim[0]
        self.pix_num_y = pix_dim[1]
        self.dpi = gui_setts.dpi
        self.fig_size = gui_setts.fig_SO2
        self.h_ratio = 3
        self.w_ratio = 25

        self.specs = CameraSpecs()

        self.q = queue.Queue()      # Queue for requesting canvas draw (when in processing thread)
        self.lock = threading.Lock()
        self.plot_lag = 0.5             # Lag between images to be plotted (don't plot every image, it freezes the GUI)
        self.draw_time = time.time()

        self.max_lines = 5  # Maximum number of ICA lines
        # ------------------------------------------------------------------------------------------
        # TK variables Setup

        # ICA lines
        self._num_ica = tk.IntVar()
        self.num_ica = 1
        self._current_ica = tk.IntVar()
        self.current_ica = 1
        self._xcorr_ica_old = tk.IntVar()
        self.xcorr_ica_old = 0
        self._xcorr_ica_young = tk.IntVar()
        self.xcorr_ica_young = 0
        self.PCS_lines_list = [None] * self.max_lines           # Pyplis line objects list
        self.ica_plt_list = [None] * self.max_lines             # Plot line objects list
        self.ica_coords = []                                    # Coordinates for most recent line plot
        self.scat_ica_point = None
        colours = 100                                           # Number of colours in cmap
        cmap = cm.get_cmap("jet", colours)
        self.line_colours = [cmap(int(f * (colours / (self.max_lines-1)))) for f in range(self.max_lines)]  # Line colours

        # Colour map
        self.cmaps = ['Reds',
                      'Blues',
                      'Greens',
                      'Oranges',
                      'Greys',
                      'viridis',
                      'plasma',
                      'inferno',
                      'magma',
                      'cividis',
                      'seismic']
        self._cmap = tk.StringVar()
        self.cmap = self.cmaps[0]
        self._tau_max = tk.DoubleVar()      # Maximum value of tau for plotting colourbar (used in spinbox)
        self.tau_max = 1
        self._auto_tau = tk.IntVar()        # Variable for defining automatic plot tau levels
        self.auto_tau = 1
        self._ppmm_max = tk.IntVar()        # Maximum value of calibated ppmm for plotting colourbar (used in spinbox)
        self.ppmm_max = 1000
        self._auto_ppmm = tk.IntVar()       # Variable for defining automatic plot tau levels
        self.auto_ppmm = 1
        self._disp_cal = tk.IntVar()        # Flag for if we are displaying the calibrated image of tau image
        self.disp_cal = 0

        # Optical flow plotting option
        self._plt_flow = tk.IntVar()
        self.plt_flow = 1

        # Interactive mode
        self._interactive_mode = tk.IntVar()
        self.interactive_mode = 0

        # Load any saved variables
        self.initiate_variables()
        # -----------------------------------------------------------------------------------------------

        # Generate main frame for figure
        self.frame = ttk.Frame(self.parent, relief=tk.RAISED, borderwidth=2)

        if self.image_tau is None:
            self.image_tau = np.random.random([self.pix_num_y, self.pix_num_x]) * self.specs._max_DN

        # Generate frame options
        self._build_options()

        # Generate analysis options frame
        self._build_analysis()

        # Generate figure
        self._build_fig_img()

        self.frame_opts.grid(row=0, column=0, pady=5, padx=2, sticky='nsew')
        self.frame_analysis.grid(row=0, column=2, pady=5, padx=2, sticky='nsew')
        self.frame_fig.grid(row=1, column=0, columnspan=3, pady=5, padx=2)

        self.frame.columnconfigure(2, weight=1)

    def initiate_variables(self):
        """Initiates variables to be loaded"""
        self.vars = {'amb_roi': list}

        self.load_defaults()

    def gather_vars(self):
        """
        Update pyplis_worker for all attributes
        :return:
        """
        self.pyplis_worker.ambient_roi = self.amb_roi

        # Update full line list
        self.pyplis_worker.PCS_lines_all = self.PCS_lines_list

        # If there is a xcorr line then we exclude it from the PCS_lines list on pyplis_worker, otherwise set the full
        # list
        if self.xcorr_ica_old > 0:
            self.pyplis_worker.PCS_lines = self.PCS_lines_list[:self.xcorr_ica_old - 1] + self.PCS_lines_list[self.xcorr_ica_old:]
        else:
            self.pyplis_worker.PCS_lines = self.PCS_lines_list

        # Set the cross-correlation line
        if self.xcorr_ica_old > 0:
            self.pyplis_worker.cross_corr_lines['old'] = self.PCS_lines_list[self.xcorr_ica_old - 1]
        else:
            self.pyplis_worker.cross_corr_lines['old'] = None

        # Set the young cross-correlation line (this line is not excluded from the total emission rate calculation -
        # only the old cross-correlation line is) It is therefore assumed that this young line is part of the set of
        # line that constrain the volcanic edifice in all directions
        if self.xcorr_ica_young > 0:
            self.pyplis_worker.cross_corr_lines['young'] = self.PCS_lines_list[self.xcorr_ica_young - 1]
        else:
            self.pyplis_worker.cross_corr_lines['young'] = None

        # Update lines (will get error on start up as fig_series isn't yet assigned. After this it works fine)
        try:
            self.fig_series.update_lines()
        except AttributeError:
            pass

    @property
    def num_ica(self):
        return self._num_ica.get()

    @num_ica.setter
    def num_ica(self, value):
        self._num_ica.set(value)

    @property
    def current_ica(self):
        return self._current_ica.get()

    @current_ica.setter
    def current_ica(self, value):
        self._current_ica.set(value)

    @property
    def xcorr_ica_old(self):
        return self._xcorr_ica_old.get()

    @xcorr_ica_old.setter
    def xcorr_ica_old(self, value):
        self._xcorr_ica_old.set(value)

    @property
    def xcorr_ica_young(self):
        return self._xcorr_ica_young.get()

    @xcorr_ica_young.setter
    def xcorr_ica_young(self, value):
        self._xcorr_ica_young.set(value)

    @property
    def cmap(self):
        return getattr(cm, self._cmap.get())

    @cmap.setter
    def cmap(self, value):
        self._cmap.set(value)

    @property
    def tau_max(self):
        return self._tau_max.get()

    @tau_max.setter
    def tau_max(self, value):
        self._tau_max.set(value)

    @property
    def auto_tau(self):
        return self._auto_tau.get()

    @auto_tau.setter
    def auto_tau(self, value):
        self._auto_tau.set(value)

    @property
    def ppmm_max(self):
        return self._ppmm_max.get()

    @ppmm_max.setter
    def ppmm_max(self, value):
        self._ppmm_max.set(value)

    @property
    def auto_ppmm(self):
        return self._auto_ppmm.get()

    @auto_ppmm.setter
    def auto_ppmm(self, value):
        self._auto_ppmm.set(value)

    @property
    def disp_cal(self):
        return self._disp_cal.get()

    @disp_cal.setter
    def disp_cal(self, value):
        self._disp_cal.set(value)

    @property
    def plt_flow(self):
        return self._plt_flow.get()

    @plt_flow.setter
    def plt_flow(self, value):
        self._plt_flow.set(value)

    @property
    def interactive_mode(self):
        return self._interactive_mode.get()

    @interactive_mode.setter
    def interactive_mode(self, value):
        self._interactive_mode.set(value)

    def _build_fig_img(self):
        """Build figure"""
        # Main frame for figure and all associated widgets
        self.frame_fig = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)

        # Create figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.fig_size, dpi=self.dpi,
                                    gridspec_kw={'height_ratios': [self.h_ratio, 1], 'width_ratios': [self.w_ratio, 1]})
        self.axes[1, 1].axis('off')  # Make bottom-right subplot blank
        self.fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05, wspace=0.00)

        self.ax = self.axes[0, 0]
        self.ax.set_aspect(1)

        self.ax_xsect = self.axes[1, 0]
        self.ax_xsect.grid()

        # Figure colour
        self.fig.set_facecolor(fig_face_colour)
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')
        for child in self.ax_xsect.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax_xsect.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Image display
        self.img_disp = self.ax.imshow(self.image_tau, cmap=self.cmap, interpolation='none', vmin=0,
                                       vmax=self.specs._max_DN, aspect='equal')
        self.ax.set_title(r'SO$_2$ image', color=axes_colour)

        # Colorbar
        # divider = make_axes_locatable(self.ax)
        # self.ax_divider = divider.append_axes("right", size="5%", pad=0.05)
        # self.cbar = plt.colorbar(self.img_disp, cax=self.ax_divider)
        self.cbar = plt.colorbar(self.img_disp, cax=self.axes[0, 1])
        self.cbar.outline.set_edgecolor(axes_colour)
        self.cbar.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Plot optical flwo if it is requested at start
        self.plt_opt_flow(draw=False)

        # Draw ambient roi
        crop_X = self.amb_roi[2] - self.amb_roi[0]
        crop_Y = self.amb_roi[3] - self.amb_roi[1]
        self.rect = self.ax.add_patch(patches.Rectangle((self.amb_roi[0], self.amb_roi[1]),
                                                        crop_X, crop_Y, edgecolor='black', fill=False, linewidth=1))

        # Finalise canvas and gridding
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.img_canvas.draw()
        self.img_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Bind click event to figure
        self.line_draw = self.fig.canvas.callbacks.connect('button_press_event', self.ica_draw)

        # Setup thread-safe plot update
        self.__draw_canv__()

    def _build_analysis(self):
        """Build analysis options"""
        self.frame_analysis = ttk.LabelFrame(self.frame, text='Analysis')

        # Number of lines
        row = 0
        label = ttk.Label(self.frame_analysis, text='Num. ICAs:')
        label.grid(row=row, column=0, sticky='w', padx=2)
        self.ica_spin = ttk.Spinbox(self.frame_analysis, textvariable=self._num_ica, from_=1, to=self.max_lines,
                                    increment=1, command=self.update_ica_num)
        self.ica_spin.grid(row=row, column=1, sticky='ew', padx=2, pady=2)

        # Line to edit
        row += 1
        label = ttk.Label(self.frame_analysis, text='Edit ICA:')
        label.grid(row=row, column=0, sticky='w', padx=2)
        self.ica_edit_spin = ttk.Spinbox(self.frame_analysis, textvariable=self._current_ica, from_=1, to=self.num_ica,
                                         increment=1)
        self.ica_edit_spin.grid(row=row, column=1, sticky='ew', padx=2, pady=2)

        # Flip ICA normal button
        self.ica_flip_butt = ttk.Button(self.frame_analysis, text='Flip ICA normal', command=self.flip_ica_normal)
        self.ica_flip_butt.grid(row=row, column=2, sticky='nsew', padx=2, pady=2)

        # Cross-correlation line (older and younger)
        row += 1
        label = ttk.Label(self.frame_analysis, text='Cross-correlation ICA [young]:')
        label.grid(row=row, column=0, sticky='w', padx=2)
        self.x_corr_spin_young = ttk.Spinbox(self.frame_analysis, textvariable=self._xcorr_ica_young,
                                             from_=0, to=self.num_ica, increment=1, command=self.gather_vars)
        self.x_corr_spin_young.grid(row=row, column=1, sticky='ew', padx=2, pady=2)


        row += 1
        label = ttk.Label(self.frame_analysis, text='Cross-correlation ICA [old]:')
        label.grid(row=row, column=0, sticky='w', padx=2)
        self.x_corr_spin_old = ttk.Spinbox(self.frame_analysis, textvariable=self._xcorr_ica_old,
                                           from_=0, to=self.num_ica, increment=1, command=self.gather_vars)
        self.x_corr_spin_old.grid(row=row, column=1, sticky='ew', padx=2, pady=2)

    def _build_options(self):
        """Builds options widget"""
        self.frame_opts = ttk.LabelFrame(self.frame, text='Figure options')

        # Colour maps
        row = 0
        label = ttk.Label(self.frame_opts, text='Colour map:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.opt_menu = ttk.OptionMenu(self.frame_opts, self._cmap, self._cmap.get(), *self.cmaps,
                                       command=self.change_cmap)
        self.opt_menu.config(width=8)
        self.opt_menu.grid(row=0, column=1, padx=2, pady=2, sticky='ew')

        # Tau or calibrated display
        row += 1
        label = ttk.Label(self.frame_opts, text='Display:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.disp_tau_rad = ttk.Radiobutton(self.frame_opts, text='\u03C4', variable=self._disp_cal, value=0,
                                            command=lambda: self.update_plot(self.image_tau, self.image_cal))
        self.disp_tau_rad.grid(row=row, column=1, padx=2, pady=2, sticky='w')
        self.disp_cal_rad = ttk.Radiobutton(self.frame_opts, text='ppm⋅m', variable=self._disp_cal, value=1,
                                            command=lambda: self.update_plot(self.image_tau, self.image_cal))
        self.disp_cal_rad.grid(row=row, column=2, padx=2, pady=2, sticky='w')
        if self.image_cal is None:
            self.disp_cal_rad.configure(state=tk.DISABLED)

        # Colour level tau
        row += 1
        label = ttk.Label(self.frame_opts, text='\u03C4 max.:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.spin_max = ttk.Spinbox(self.frame_opts, width=4, textvariable=self._tau_max, from_=0, to=9, increment=0.01,
                                    command=self.scale_img)
        self.spin_max.set('{:.2f}'.format(self.tau_max))
        self.spin_max.grid(row=row, column=1, padx=2, pady=2, sticky='ew')
        self.auto_tau_check = ttk.Checkbutton(self.frame_opts, text='Auto', variable=self._auto_tau,
                                              command=self.scale_img)
        self.auto_tau_check.grid(row=row, column=2, padx=2, pady=2, sticky='w')

        # Colour level ppmm
        row += 1
        label = ttk.Label(self.frame_opts, text='ppm⋅m max.:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.spin_ppmm_max = ttk.Spinbox(self.frame_opts, width=5, textvariable=self._ppmm_max, from_=0, to=50000,
                                         increment=50, command=self.scale_img)
        self.spin_ppmm_max.grid(row=row, column=1, padx=2, pady=2, sticky='ew')
        self.auto_ppmm_check = ttk.Checkbutton(self.frame_opts, text='Auto', variable=self._auto_ppmm,
                                              command=self.scale_img)
        self.auto_ppmm_check.grid(row=row, column=2, padx=2, pady=2, sticky='w')

        # Optical flow checkbutton
        row += 1
        self.plt_flow_check = ttk.Checkbutton(self.frame_opts, text='Display Optical Flow', variable=self._plt_flow,
                                              command=self.plt_opt_flow)
        self.plt_flow_check.grid(row=row, column=0, columnspan=2, padx=2, pady=2, sticky='w')

        # Interactive mode selector
        row += 1
        selector_frame = ttk.Frame(self.frame_opts, relief=tk.RAISED, borderwidth=2)
        selector_frame.grid(row=row, column=0, columnspan=3, sticky='nsew')
        self.line_draw_sel = ttk.Radiobutton(selector_frame, text='Draw ICA line',
                                             variable=self._interactive_mode, value=0,
                                             command=self.change_interactive_mode)
        self.roi_draw_sel = ttk.Radiobutton(selector_frame, text='Draw ambient ROI',
                                            variable=self._interactive_mode, value=1,
                                            command=self.change_interactive_mode)
        self.line_draw_sel.grid(row=0, column=0, sticky='w', padx=2, pady=2)
        self.roi_draw_sel.grid(row=0, column=1, sticky='w', padx=2, pady=2)

    def change_interactive_mode(self):
        """Changes interactive mode from drawing PCS lines to drawing ambient region"""
        if self.interactive_mode == 0:
            self.rs.disconnect_events()
            self.line_draw = self.fig.canvas.callbacks.connect('button_press_event', self.ica_draw)
        elif self.interactive_mode == 1:
            self.fig.canvas.mpl_disconnect(self.line_draw)
            self.rs = widgets.RectangleSelector(self.ax, self.draw_roi, drawtype='box',
                                                rectprops=dict(facecolor='red', edgecolor='blue', alpha=0.5, fill=True))
        else:
            raise ValueError('Unrecognised interactive_mode for ImageSO2')

    def draw_roi(self, eclick, erelease):
        """
        Draws region of interest for calculating optical flow
        :return:
        """
        try:  # Delete previous rectangle, if it exists
            self.rect.remove()
        except AttributeError:
            pass
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        self.roi_start_y, self.roi_end_y = int(eclick.ydata), int(erelease.ydata)
        self.roi_start_x, self.roi_end_x = int(eclick.xdata), int(erelease.xdata)
        crop_Y = erelease.ydata - eclick.ydata
        crop_X = erelease.xdata - eclick.xdata
        self.rect = self.ax.add_patch(patches.Rectangle((self.roi_start_x, self.roi_start_y),
                                                        crop_X, crop_Y, edgecolor='black', fill=False, linewidth=1))

        # Only update roi_abs if use_roi is true
        self.amb_roi = [self.roi_start_x, self.roi_start_y, self.roi_end_x, self.roi_end_y]
        self.gather_vars()

        self.q.put(1)

    def add_pcs_line(self, line, line_num=None, force_add=True):
        """
        Adds LineOnImage object to plot and updates all relevant objects
        :param line:        LineOnImage     Object to be added to plot
        :param line_num:    int             Index to add line to (starts at 1, not 0). If None we add it to first
                                            available position
        :param force_add    bool            If True we add the line even if there are no spaces left.
                                            We replace the current_ica for it
        """
        # If line number is given we delete it if it exists
        if line_num is not None:
            line_idx = line_num - 1
            if self.PCS_lines_list[line_idx] is not None:
                self.del_ica(line_idx)
        else:
            line_idx = None
            # Line is only added if we don't already
            for i in range(self.max_lines):
                if self.PCS_lines_list[i] is None:
                    line_idx = i

                    # Update spinbox to reflect the fact we have added a new line
                    self.num_ica = i + 1
                    self.update_ica_num()
                    break

            # If there are no space for this line we either force adding it, by overwriting current edit line or we
            # return from the function without adding the line
            if line_idx is None:
                if force_add:
                    line_idx = self.current_ica - 1
                    self.del_ica(line_idx)
                else:
                    print('No space to add PCS line, please use force_add or delete a line before adding another')
                    return

        # Make line
        lbl = "{}".format(line_idx)
        line.line_id = lbl
        line.color = self.line_colours[line_idx]
        self.PCS_lines_list[line_idx] = line
        # Plot pyplis object on figure
        self.PCS_lines_list[line_idx].plot_line_on_grid(ax=self.ax, include_normal=1,
                                                       include_roi_rot=True, label=lbl)

        # Gather variables to update pyplis_worker object
        self.gather_vars()

        # Redraw canvas
        self.img_canvas.draw()

        # Update time series lines
        self.fig_series.update_lines()

    def update_ica_num(self):
        """Makes necessary changes to update the number of ICA lines"""

        # Edit 'to' of ica_edit_spin spinbox and if current ICA is above number of ICAs we update current ICA
        self.ica_edit_spin.configure(to=self.num_ica)
        self.current_ica = self.num_ica

        # Edit cross-correlation ICA
        self.x_corr_spin_old.configure(to=self.num_ica)
        if self.xcorr_ica_old > self.num_ica:       # If the xcorr_ica_old is deleted we set the xcorr back to 0
            self.xcorr_ica_old = 0

        # Edit cross-correlation ICA (young)
        self.x_corr_spin_young.configure(to=self.num_ica)
        if self.xcorr_ica_young > self.num_ica:       # If the xcorr_ica_young is deleted we set the xcorr back to 0
            self.xcorr_ica_young = 0

        # Delete any drawn lines over the new number requested if they are present
        ica_num = self.num_ica
        for ica in self.PCS_lines_list[self.num_ica:]:
            if ica is not None:
                self.del_ica(ica_num)
                # self.PCS_lines_list[ica_num] = None   # I think this isn't required as it is done in del_ica
            ica_num += 1

        # Gather variables
        self.gather_vars()

    def flip_ica_normal(self):
        """Flips the normal vector of the current ICA"""
        ica_idx = self.current_ica - 1

        # Get line currently being edited
        line = self.PCS_lines_list[ica_idx]
        lbl = "{}".format(ica_idx)

        # If current line is not none we find it's orientation and reverse it
        if line is not None:
            self.del_ica(ica_idx, update_all=False)

            if line.normal_orientation == 'right':
                self.PCS_lines_list[ica_idx] = LineOnImage(x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1,
                                                           normal_orientation="left", color=self.line_colours[ica_idx],
                                                           line_id=lbl)

            elif line.normal_orientation == 'left':
                self.PCS_lines_list[ica_idx] = LineOnImage(x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1,
                                                           normal_orientation="right", color=self.line_colours[ica_idx],
                                                           line_id=lbl)

            # Plot pyplis object on figure
            self.PCS_lines_list[ica_idx].plot_line_on_grid(ax=self.ax, include_normal=1,
                                                           include_roi_rot=True, label=lbl)

            # Gather variables to update pyplis_worker object
            self.gather_vars()

            # Redraw canvas
            self.img_canvas.draw()

    def ica_draw(self, event):
        """Collects points for ICA line and then draws it when a complete line is drawn"""
        # Python indices start at 0, so need to set the correct index for list indexing
        PCS_idx = self.current_ica - 1

        if event.inaxes is self.ax:
            idx = len(self.ica_coords)

            # If we are about to overwrite an old line, we first check that the user wants this
            if idx == 1:
                if self.PCS_lines_list[PCS_idx] is not None:
                    resp = messagebox.askokcancel('Overwriting line',
                                                  'You are about to overwrite an existing line.\n'
                                                  'This could affect processing results if it is currently running.')
                    if not resp:
                        return

            # If 2 points are already defined we want to clear these points
            if idx == 2:
                self.ica_coords = []
                idx = 0  # Resetting index for 'point' definition

            # Update ica_coords with new coordinates
            self.ica_coords.append((event.xdata, event.ydata))

            # Remove last click point and scatter current click
            try:
                self.scat_ica_point.remove()
            except:
                pass
            self.scat_ica_point = self.ax.scatter(event.xdata, event.ydata, s=50, marker='x', color='k', lw=1)

            self.ax.set_xlim(0, self.pix_num_x - 1)
            self.ax.set_ylim(self.pix_num_y - 1, 0)

            if idx == 1:
                # Delete scatter point
                try:
                    self.scat_ica_point.remove()
                except:
                    pass

                # Delete previous line if it exists
                if self.PCS_lines_list[PCS_idx] is not None:
                    self.del_ica(PCS_idx, update_all=False)

                # Update pyplis line object and objects in pyplis_worker
                lbl = "{}".format(PCS_idx)
                self.PCS_lines_list[PCS_idx] = LineOnImage(x0=self.ica_coords[0][0],
                                                           y0=self.ica_coords[0][1],
                                                           x1=self.ica_coords[1][0],
                                                           y1=self.ica_coords[1][1],
                                                           normal_orientation="right",
                                                           color=self.line_colours[PCS_idx],
                                                           line_id=lbl)

                # Plot pyplis object on figure
                self.PCS_lines_list[PCS_idx].plot_line_on_grid(ax=self.ax, include_normal=1,
                                                               include_roi_rot=True, label=lbl)

                # Update lines
                self.gather_vars()

                # Extract ICA values and plot them
                self.update_xsect()

            self.img_canvas.draw()
        else:
            print('Clicked outside axes bounds but inside plot window')

    def del_ica(self, line_num, update_all=True):
        """Searches axis for line object relating to pyplis line object and removes it

        Parameters
        ----------
        line_num: int
            Index of line in PCS_lines_list
        :param update_all:  bool
            If True all drawing etc is done, otherwise it isn't (it will be set to False in flip_ica_normal)
        """
        # Get line
        line = self.PCS_lines_list[line_num]

        # Search for line and remove it when the correct one is found
        plt_lines = self.ax.get_lines()
        for l in plt_lines:
            if l._label == line.line_id:
                l.remove()

        # Search for everything else and remove
        children = self.ax.get_children()
        for child in children:
            if isinstance(child, matplotlib.patches.FancyArrow):
                if child._facecolor == line.color:
                    child.remove()
            elif isinstance(child, matplotlib.patches.Polygon):
                if child._original_facecolor == line.color:
                    child.remove()


        # Once removed, set the line to None
        self.PCS_lines_list[line_num] = None

        if update_all:
            # Gather variables
            self.gather_vars()

            # Update xsect_plot
            self.update_xsect()

            # Redraw canvas
            self.img_canvas.draw()

    def change_cmap(self, cmap):
        """Change colourmap of image"""
        # Set cmap to new value
        self.img_disp.set_cmap(getattr(cm, cmap))

        # Update canvas, first rescaling image, as the seismic canvas we use a different scale
        self.scale_img(draw=True)

    def scale_img(self, draw=True):
        """
        Updates cmap scale
        :param draw: bool   Defines whether the image canvas is redrawn after updating cmap
        :return:
        """
        if self.disp_cal:
            # Get vmax either automatically or by defined spinbox value
            if self.auto_ppmm:
                self.vmax_cal = np.nanpercentile(self.image_cal, 99)
            else:
                self.vmax_cal = self.ppmm_max
            if self.cmap.name == 'seismic':
                vmin = -self.vmax_cal
            else:
                vmin = 0
            self.img_disp.set_clim(vmin=vmin, vmax=self.vmax_cal)
            self.cbar.ax.set_title('ppm.m')
        else:
            # Get vmax either automatically or by defined spinbox value
            if self.auto_tau:
                self.vmax_tau = np.nanpercentile(self.image_tau, 99)
            else:
                self.vmax_tau = self.tau_max
            if self.cmap.name == 'seismic':
                vmin = -self.vmax_tau
            else:
                vmin = 0
            self.img_disp.set_clim(vmin=vmin, vmax=self.vmax_tau)
            self.cbar.ax.set_title(r'$\tau$')

        # Set new limits

        if draw:
            # If in processing, the canvas is drawn a lot, so we don't draw it here
            if not self.pyplis_worker.in_processing:
                self.img_canvas.draw()

    def plt_opt_flow(self, draw=True):
        """Plots optical flow onto figure"""
        # Delete old optical flow lines
        for child in self.ax.get_children():
            del_line = False

            if isinstance(child, patches.Circle):
                del_line = True

                # Loop thorugh PCS lines and check their color. If they match the current circle, don't delete
                # We only want to delete optical flow circles
                color = child._original_edgecolor
                for line in self.PCS_lines_list:
                    if line is not None:
                        if color == line.color:
                            del_line = False
                            break

            elif isinstance(child, mpllines.Line2D):
                del_line = True

                # Loop thorugh PCS lines and check their color. If they match the current line, don't delete
                # We only want to delete optical flow line
                x_dat = child.get_xdata()
                y_dat = child.get_ydata()
                try:
                    coords = [[x_dat[0], y_dat[0]], [x_dat[-1], y_dat[-1]]]
                    for line in self.PCS_lines_list:
                        if line is not None:
                            if line.start in coords and line.stop in coords:
                                del_line = False
                                break
                except IndexError:      # Index error means we have part of the ambient ROI rect, so don't delete
                    continue

            if del_line:
                child.remove()

        if self.plt_flow:
            # Update flow_lines
            pyplis_worker.opt_flow.draw_flow(ax=self.ax, in_roi=True)
            self.ax.set_xlim([0, self.pix_num_x])
            self.ax.set_ylim([self.pix_num_y, 0])

        if draw:
            self.q.put(1)
            # self.img_canvas.draw()

    def update_xsect(self):
        """Updates corss-section subplot"""
        # Clear axis
        self.ax_xsect.clear()

        if self.disp_cal:
            for line in self.pyplis_worker.PCS_lines_all:
                if isinstance(line, LineOnImage):
                    line_id = str(int(line.line_id) + 1)
                    self.ax_xsect.plot(line.get_line_profile(self.image_cal), color=line.color, label=line_id)
            self.ax_xsect.set_ylabel('CD [ppm.m]', color=axes_colour)
        else:
            for line in self.pyplis_worker.PCS_lines_all:
                if isinstance(line, LineOnImage):
                    line_id = str(int(line.line_id) + 1)
                    self.ax_xsect.plot(line.get_line_profile(self.image_tau), color=line.color, label=line_id)
            self.ax_xsect.set_ylabel(r'$\tau$', color=axes_colour)

        # Set xsection aspect ratio
        xlims = self.ax_xsect.get_xlim()
        self.ax_xsect.set_xlim([0, xlims[-1]])
        asp = np.diff(self.ax_xsect.get_xlim())[0] / np.diff(self.ax_xsect.get_ylim())[0]
        asp /= np.abs(np.diff(self.ax.get_xlim())[0] / np.diff(self.ax.get_ylim())[0])
        asp /= self.h_ratio
        self.ax_xsect.set_aspect(asp)

        self.ax_xsect.grid(b=True, which='major')
        self.ax_xsect.legend(loc='upper right')

    def update_plot(self, img_tau, img_cal=None, draw=True):
        """
        Updates image figure and all associated subplots
        :param img:     np.ndarray/pyplis.Img   Image array
        :param draw:    bool                    If True, the plot is drawn (use False when calling from a thread)
        :return:
        """
        # Extract arrays if we aree given pyplis.Img objects
        if isinstance(img_tau, Img):
            img_tau = img_tau.img
        if isinstance(img_cal, Img):
            img_cal = np.copy(img_cal.img) / pyplis_worker.ppmm_conv

        self.image_tau = img_tau
        self.image_cal = img_cal

        # Disable radiobutton if no calibrated image is present - we can't plot what isn't there...
        if self.image_cal is None:
            self.disp_cal_rad.configure(state=tk.DISABLED)
            self.disp_cal = 0       # Set this so scale_img knows we are on tau not cal
        else:
            self.disp_cal_rad.configure(state=tk.NORMAL)

        # Update main image display and title
        if self.disp_cal and img_cal is not None:
            self.img_disp.set_data(img_cal)
        else:
            self.img_disp.set_data(img_tau)
        self.scale_img(draw=False)

        with self.lock:
        # Plot optical flow
            self.plt_opt_flow(draw=False)

        with self.lock:
        # Update cross-section plot
            self.update_xsect()

        if draw:
            self.q.put(1)

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                # self.img_canvas.draw()
                # self.cbar.draw_all()

                # If the lock is currently in use (updating images, exit and will return to try again later)
                if self.lock.locked():
                    self.q.put(1)
                else:
                    with self.lock:
                        self.img_canvas.draw()
                        self.cbar.draw_all()
            else:
                return
        except queue.Empty:
            pass
        self.root.after(refresh_rate, self.__draw_canv__)


class TimeSeriesFigure:
    """
    Class for frame holding time series plot and associated functions to update the plot
    """
    def __init__(self, parent, pyplis_work=pyplis_worker, setts=gui_setts):
        self.parent = parent
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_series = self
        self.pyplis_worker.fig_tau.fig_series = self
        self.q = queue.Queue()
        self.settings = setts

        self.frame = tk.Frame(self.parent, relief=tk.RAISED, borderwidth=2)

        # Plot Options
        self.style = 'default'
        self.date_fmt = '%HH:%MM'
        self.plot_styles = {'flow_glob': {'ls': 'solid'},
                            'flow_raw': {'ls': 'dotted'},
                            'flow_histo': {'ls': 'dashed'},
                            'flow_hybrid': {'ls': 'dashdot'}}
        self.colours = self.pyplis_worker.fig_tau.line_colours
        self.marker = '.'

        # Initiate variables
        self.initiate_variables()

        # Build or widgets
        self._build_opts()
        self._build_fig()
        self.opts_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.fig_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)

        # Begin refreshing of plot
        self.__draw_canv__()

    def initiate_variables(self):
        """Initiates tkinter variables used by object"""
        self._line_plot = tk.StringVar()
        self._plot_total = tk.IntVar()      # If 1, the total of all lines is plotted
        self.plot_total = 1
        self.lines = []                     # List holding ids of all lines currently drawn
        self.total_lines = []               # Lines which contribute to the 'total' emission rate

    @property
    def plot_total(self):
        return self._plot_total.get()

    @plot_total.setter
    def plot_total(self, value):
        self._plot_total.set(int(value))

    @property
    def line_plot(self):
        # We decrease the increment of the line by one so that it matches the line_id for pyplis object
        # (the value was icnreased by one before being set, to match the SO2 image - see update_lines())
        try:
            line = '{}'.format(int(self._line_plot.get()) - 1)
        except:
            line = 'None'
        return line

    @line_plot.setter
    def line_plot(self, value):
        self._line_plot.set(value)

    def _build_opts(self):
        """Builds options widget"""
        self.opts_frame = ttk.LabelFrame(self.frame, text='Options')

        lab = ttk.Label(self.opts_frame, text='Plot line:')
        lab.grid(row=0, column=0, sticky='w', padx=2, pady=2)
        self.line_opts = ttk.Combobox(self.opts_frame, textvariable=self._line_plot, justify='left',
                                      state='readonly')
        self.line_opts.bind('<<ComboboxSelected>>', self.update_plot)
        self.line_opts.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)

        self.plt_tot_check = ttk.Checkbutton(self.opts_frame, text='Plot sum of all lines', variable=self._plot_total,
                                             command=self.update_plot)
        self.plt_tot_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=2, pady=2)

        # Update current line options
        self.update_lines(plot=False)

    def _build_fig(self):
        """Builds frame for time series plot"""
        self.fig_frame = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=2)

        self.fig = plt.Figure(figsize=self.settings.fig_series, dpi=self.settings.dpi)
        self.fig.set_facecolor(fig_face_colour)

        self.axes = [None] * 4
        gs = plt.GridSpec(4, 1, height_ratios=[.6, .2, .2, .2], hspace=0.05)
        self.axes[0] = self.fig.add_subplot(gs[0])
        for i in range(3):
            self.axes[i+1] = self.fig.add_subplot(gs[i+1], sharex=self.axes[0])
        for i in range(2):
            self.axes[(2*i) + 1].yaxis.tick_right()
            self.axes[(2*i) + 1].yaxis.set_label_position("right")
            self.axes[i+1].set_xticklabels([])
            plt.setp(self.axes[i + 1].get_xticklabels(), visible=False)
        self.axes[0].xaxis.tick_top()
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self.axes[-1].set_xlabel('Time')

        # Finalise canvas and gridding
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP)

        # Add toolbar so figures can be saved
        toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP)

    def update_lines(self, plot=True):
        """Updates lines available to optionmenu"""
        self.lines = []
        for line in self.pyplis_worker.PCS_lines_all:
            if isinstance(line, LineOnImage):
                # Incremement line by one so it matches the line number in the SO2 image figure
                self.lines.append('{}'.format(int(line.line_id) + 1))

        # Update widget
        self.line_opts['values'] = self.lines
        if len(self.lines) > 0:
            current_line = self.line_plot if self.line_plot in self.lines else self.lines[0]
        else:
            current_line = 'None'
        self.line_opts.set(current_line)

        # If we have more than one line we can open the option to plot the total too
        self.total_lines = [f for f in self.pyplis_worker.PCS_lines if isinstance(f, LineOnImage)]
        if len(self.total_lines) < 2:
            self.plt_tot_check.configure(state=tk.DISABLED)
        else:
            self.plt_tot_check.configure(state=tk.NORMAL)

        # Update plot
        if plot:
            self.update_plot()

    def update_plot(self, draw=True):
        """
        Update time series plot with new data
        :param draw:    bool    Plot is only drawn if draw=True, otherwise plot updates will not be drawn
        """
        # Clear old data
        for ax in self.axes:
            ax.clear()

        if self.line_plot.lower() != 'none':
            for mode in self.pyplis_worker.velo_modes:
                if self.pyplis_worker.velo_modes[mode]:
                    try:
                        if len(self.pyplis_worker.results[self.line_plot][mode]._phi) > 0:
                            line_lab = 'line_{}: {}'.format(int(self.line_plot) + 1, mode)
                            self.plot_bg_roi(marker=self.marker)
                            self.plot_flow_dir(self.line_plot, label=line_lab,
                                               color=self.colours[int(self.line_plot)],
                                               marker='.')
                            self.plot_veff(self.pyplis_worker.results[self.line_plot][mode],
                                           label=line_lab, ls=self.plot_styles[mode]['ls'],
                                           color=self.colours[int(self.line_plot)],
                                           marker=self.marker)
                            self.pyplis_worker.results[self.line_plot][mode].plot(ax=self.axes[0],
                                                                                  ls=self.plot_styles[mode]['ls'],
                                                                                  color=self.colours[
                                                                                      int(self.line_plot)],
                                                                                  lw=1.5,
                                                                                  ymin=0,
                                                                                  date_fmt=self.date_fmt,
                                                                                  label=line_lab,
                                                                                  marker=self.marker
                                                                                  )
                    except KeyError:
                        print('No emission rate analysis data available for {}'.format(self.line_plot))

        # Plot the summed total
        if self.plot_total and len(self.total_lines) > 1:
            for mode in self.pyplis_worker.velo_modes:
                if self.pyplis_worker.velo_modes[mode]:
                    try:
                        if len(self.pyplis_worker.results['total'][mode]._phi) > 0:
                            self.pyplis_worker.results['total'][mode].plot(ax=self.axes[0],
                                                                           ls=self.plot_styles[mode]['ls'],
                                                                           color='black',
                                                                           lw=2,
                                                                           ymin=0,
                                                                           date_fmt=self.date_fmt,
                                                                           label='total: {}'.format(mode),
                                                                           marker=self.marker)
                    except KeyError:
                        print('No emission rate analysis data available for sum of all ICA lines')

        # Adjust ylimits and do general plot tidying
        self.axes[0].autoscale(axis='y')
        lims = self.axes[0].get_ylim()
        self.axes[0].set_ylim((0, lims[1]))
        self.axes[0].legend(loc='upper left')
        self.axes[1].set_ylabel(r"$v_{eff}$ [m/s]")
        self.axes[2].set_ylabel(r"$\varphi\,[^{\circ}$]")
        self.axes[3].set_ylabel(r"$ROI_{BG}\,[cm^{-2}]$")
        for i in range(len(self.axes)):
            self.axes[i].grid(b=True, which='major')
            if i == 1 or i == 2:
                plt.setp(self.axes[i].get_xticklabels(), visible=False)

        # Draw if requested to
        if draw:
            self.q.put(1)

    def plot_veff(self, emission_rates, ax=1, yerr=True, **kwargs):
        """
        Controls plotting effective velocity - pyplis plotting of this doesn't seem to work on my data

        :param emission_rates:  pyplis.fluxcalc.EmissionRates       Emissions rates containing all info
        """
        times = emission_rates.start_acq
        veff, verr = emission_rates.velo_eff, emission_rates.velo_eff_err
        self.axes[ax].plot(times, veff, **kwargs)

        # Plot errors
        if yerr:
            phi_upper = Series(veff + verr, times)
            phi_lower = Series(veff - verr, times)
            kwargs['lw'] = 0    # Plot no lines around fill
            kwargs.pop('marker', None)  # Remove the amrker key as fill between doesn't take this
            self.axes[ax].fill_between(times, phi_lower, phi_upper, alpha=0.1, **kwargs)
        self.axes[ax].autoscale(axis='y')
        self.axes[ax].set_ylim([0, self.axes[ax].get_ylim()[1]])

    def plot_flow_dir(self, line_id, ax=2, yerr=True, **kwargs):
        """Plot predominant wind direction retrieved from flow_hybrid"""
        times = self.pyplis_worker.results[line_id]['flow_histo'].start_acq
        orientations = self.pyplis_worker.results[line_id]['flow_histo']._flow_orient

        self.axes[ax].plot(times, orientations, **kwargs)

        # Plot errors
        if yerr:
            upper = self.pyplis_worker.results[line_id]['flow_histo']._flow_orient_upper
            lower = self.pyplis_worker.results[line_id]['flow_histo']._flow_orient_lower
            kwargs['lw'] = 0
            kwargs.pop('marker', None)  # Remove the amrker key as fill between doesn't take this
            self.axes[ax].fill_between(times, lower, upper, alpha=0.1, **kwargs)
        self.axes[ax].set_ylim([-180, 180])

    def plot_bg_roi(self, ax=3, yerr=True, **kwargs):
        """Plot background region column density and standard deviation"""
        if "color" not in kwargs:
            kwargs["color"] = "r"

        # Plot background timeseries
        self.axes[ax].plot(self.pyplis_worker.ts, self.pyplis_worker.bg_mean, **kwargs)

        # Plot standard deviations
        if yerr:
            mean_arr = np.array(self.pyplis_worker.bg_mean)     # Convert to numpy for element-wise addition/subtraction
            bg_upper = mean_arr + np.array(self.pyplis_worker.bg_std)
            bg_lower = mean_arr - np.array(self.pyplis_worker.bg_std)

            kwargs['lw'] = 0  # Plot no lines around fill
            kwargs.pop('marker', None)      # Remove the amrker key as fill between doesn't take this
            self.axes[ax].fill_between(self.pyplis_worker.ts, bg_lower, bg_upper, alpha=0.1, **kwargs)

        # Set axis y-limits
        self.axes[ax].autoscale(axis='y')
        lim = max(np.abs(self.axes[ax].get_ylim()))
        self.axes[ax].set_ylim([-lim, lim])

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                self.canvas.draw()
            else:
                return
        except queue.Empty:
            pass
        self.frame.after(refresh_rate, self.__draw_canv__)


class GeomSettings:
    """
    Creates frame holding all geometry and associated settings required by pyplis. This frame links to the PyplisWorker
    object to update settings when needed.

    This object should be instantiated on startup, so a conf file needs to do this, and then the generate_frame() method
    should be used at the point at which it is required
    """
    def __init__(self, parent=None, generate_frame=False, geom_path=FileLocator.CAM_GEOM, fig_setts=gui_setts):
        self.parent = parent
        self.frame = None
        self.geom_path = geom_path
        self.filename = None        # Path to file of current settings
        self.dpi = fig_setts.dpi
        self.fig_size = fig_setts.fig_img
        self.q = queue.Queue()

        self.pyplis_worker = pyplis_worker

        self.in_frame = False

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def initiate_variables(self):
        """Initiates object, and builds frame if parent is a tk.Frame"""
        # Tk Variables
        self._lat = tk.StringVar()
        self._lon = tk.StringVar()
        self._altitude = tk.IntVar()
        self._alt_offset = tk.IntVar()
        self._elev = tk.DoubleVar()
        self._azim = tk.DoubleVar()
        self._volcano = tk.StringVar()
        self._lat_ref = tk.StringVar()
        self._lon_ref = tk.StringVar()
        self._altitude_ref = tk.IntVar()

        self.geom_dict = {'lat': None,
                          'lon': None,
                          'altitude': None,
                          'alt_offset': None,
                          'elev': None,
                          'azim': None}    # List of attributes



        # Setting start values of variables
        with open(FileLocator.DEFAULT_GEOM, 'r') as f:
            self.filename = f.readline()
        self.load_instrument_setup(self.filename)

    def generate_frame(self):
        """Generates the GUI for this frame. This method of generating the frame means that
        the object can exist, with variables being instantiated, prior to building the frame"""
        # If we are already in the frame, just lift the frame
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        # Generate top level frame
        self.frame = tk.Toplevel()
        self.frame.title('Measurement geometry configuration')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        self.in_frame = True

        # ----------------------------------------------------------------------
        # Tkinter widgets
        self.frame_geom = ttk.LabelFrame(self.frame, text='Instrument geometry', borderwidth=2)
        self.frame_geom.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        row = 0

        label = ttk.Label(self.frame_geom, text='Latitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame_geom, width=10, textvariable=self._lat)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Longitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame_geom, width=10, textvariable=self._lon)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Altitude [m]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame_geom, textvariable=self._altitude, from_=0, to=8848, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Altitude offset [m]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame_geom, textvariable=self._alt_offset, from_=0, to=9999, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Elevation angle [°]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame_geom, format='%.1f', textvariable=self._elev,
                              from_=-90, to=90, increment=0.1, width=4)
        spinbox.set('{:.1f}'.format(self.elev))
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Azimuth [°]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame_geom, textvariable=self._azim, from_=0, to=359, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_geom, text='Volcano:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame_geom, width=10, textvariable=self._volcano)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame_geom, text='Update settings', command=self.update_geom)
        button.grid(row=row, column=0, padx=2, pady=2, sticky='nsew')

        button = ttk.Button(self.frame_geom, text='Save settings', command=self.save_instrument_setup)
        button.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame_geom, text='Load settings', command=self.load_instrument_setup)
        button.grid(row=row, column=0, padx=2, pady=2, sticky='nsew')

        button = ttk.Button(self.frame_geom, text='Set default', command=self.set_default_instrument_setup)
        button.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame_geom, text='Draw geometry', command=self.draw_geometry)
        button.grid(row=row, column=0, padx=2, pady=2, sticky='nsew')

        # Reference point location widgets
        self.frame_ref = ttk.LabelFrame(self.frame, text='Reference point location')
        self.frame_ref.grid(row=1, column=0, sticky='new', padx=5, pady=5)

        row = 0
        label = ttk.Label(self.frame_ref, text='Latitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame_ref, width=10, textvariable=self._lat_ref)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_ref, text='Longitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame_ref, width=10, textvariable=self._lon_ref)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame_ref, text='Altitude [m]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame_ref, textvariable=self._altitude_ref, from_=0, to=8000, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame_ref, text='Find viewing direction', command=self.find_viewing_direction)
        button.grid(row=row, column=0, columnspan=2, padx=2, pady=2, sticky='nsew')

        self.frame_ref.grid_columnconfigure(1, weight=1)
        self.frame_ref.grid_rowconfigure(1, weight=1)

        # Figure setup
        self.frame_fig = ttk.Frame(self.frame)
        self.frame_fig.grid(row=0, column=1, rowspan=2, sticky='nw')

        self._build_fig()

        self.__draw_canv__()

    def _build_fig(self):
        """Build figure for adjust geometry based on a defined point"""
        # Create figure
        self.fig = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_aspect(1)

        # Figure colour
        self.fig.set_facecolor(fig_face_colour)
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Image display
        self.img_A = self.pyplis_worker.img_A.img
        self.img_disp = self.ax.imshow(self.img_A, cmap=cm.gray, interpolation='none', vmin=0,
                                       vmax=self.pyplis_worker.cam_specs._max_DN, aspect='equal')
        self.ax.set_title('Reference point selection', color=axes_colour)

        # Finalise canvas and gridding (canvases are drawn in _build_fig_vel())
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.img_canvas.get_tk_widget().grid(row=0, column=0)

        # Bind click event to figure
        self.fig.canvas.callbacks.connect('button_press_event', self.set_ref_pt)

        # Draw canvas
        self.q.put(1)

    def set_ref_pt(self, event):
        """Sets reference point from a click"""
        if event.inaxes is self.ax:
            self.pix_x = event.xdata
            self.pix_y = event.ydata

        try:
            self.ref_pt_plot.remove()
        except AttributeError:
            pass

        self.ref_pt_plot = self.ax.scatter(self.pix_x, self.pix_y, c='orange')

        self.q.put(1)

    @property
    def lat(self):
        return self._lat.get()

    @lat.setter
    def lat(self, value):
        self._lat.set(value)

    @property
    def lon(self):
        return self._lon.get()

    @lon.setter
    def lon(self, value):
        self._lon.set(value)

    @property
    def altitude(self):
        return self._altitude.get()

    @altitude.setter
    def altitude(self, value):
        self._altitude.set(value)

    @property
    def alt_offset(self):
        return self._alt_offset.get()

    @alt_offset.setter
    def alt_offset(self, value):
        self._alt_offset.set(value)

    @property
    def elev(self):
        return self._elev.get()

    @elev.setter
    def elev(self, value):
        self._elev.set(value)

    @property
    def azim(self):
        return self._azim.get()

    @azim.setter
    def azim(self, value):
        self._azim.set(value)

    @property
    def volcano(self):
        return self._volcano.get()

    @volcano.setter
    def volcano(self, value):
        self._volcano.set(value)

    @property
    def lat_ref(self):
        return self._lat_ref.get()

    @lat_ref.setter
    def lat_ref(self, value):
        self._lat_ref.set(value)

    @property
    def lon_ref(self):
        return self._lon_ref.get()

    @lon_ref.setter
    def lon_ref(self, value):
        self._lon_ref.set(value)

    @property
    def altitude_ref(self):
        return self._altitude_ref.get()

    @altitude_ref.setter
    def altitude_ref(self, value):
        self._altitude_ref.set(value)

    def gather_geom(self):
        """Gathers all geometry data and adds it to the dictionary"""
        for key in self.geom_dict:
            self.geom_dict[key] = getattr(self, key)

    def update_geom(self):
        """Updates pyplis MeasGeom object with values from this frame"""
        # Update geometry dictionary and pyplis worker camera object
        self.gather_geom()
        self.pyplis_worker.update_cam_geom(self.geom_dict)

        # Update measurement setup with location
        try:
            self.pyplis_worker.measurement_setup(location=self.volcano)
        except UnrecognisedSourceError:
            messagebox.showerror('Source not recognised.',
                                 'The volcano source {} was not recognised. \n'
                                 'Please try a different source name or add '
                                 'source information manually.'.format(self.volcano))

    def save_instrument_setup(self):
        """Saves all of the current instrument geometry settings to a text file, so it can be loaded on start-up"""
        # Ensure that all tkinter variables have been gathered and that current volcano location exists
        # We do not want to save a geometry which isn't fully functional
        self.update_geom()

        # Ask user to define filename for saving geometry settings
        self.filename = filedialog.asksaveasfilename(initialdir=self.geom_path)

        # Ensure filename has .txt format
        self.filename = '{}.txt'.format(self.filename.split('.')[0])

        # Open file object and write all attributes to it
        with open(self.filename, 'w') as f:
            f.write('# Geometry setup file\n')
            f.write('volcano={}\n'.format(self.volcano))
            for key in self.geom_dict:
                f.write('{}={}\n'.format(key, self.geom_dict[key]))

    def load_instrument_setup(self, filepath=None):
        """Loads existing instrument setup"""
        # Get filename to load
        if filepath is None:
            self.filename = filedialog.askopenfilename(initialdir=self.geom_path)

        with open(self.filename, 'r') as f:
            for line in f:
                # Ignore first line
                if line[0] == '#':
                    continue

                # Extract key-value pair, remove the newline character from the value, then recast
                key, value = line.split('=')
                value = value.replace('\n', '')
                if key == 'volcano':
                    setattr(self, key, value)
                elif key == 'altitude' or key == 'alt_offset':
                    setattr(self, key, int(value))
                else:
                    setattr(self, key, float(value))

        # Update geometry settings
        self.update_geom()

    def set_default_instrument_setup(self):
        """Sets default instrument setup to load on startup"""
        # First save the settings
        self.save_instrument_setup()

        # Update default
        with open(FileLocator.DEFAULT_GEOM, 'w') as f:
            f.write(self.filename)

    def draw_geometry(self):
        """Draws geometry through pyplis"""
        self.update_geom()
        self.pyplis_worker.show_geom()

    def find_viewing_direction(self):
        """
        Uses reference point to calculate the viewing direction of camera using pyplis function
        :return:
        """
        if not hasattr(self, 'pix_x'):
            messagebox.showerror('Run error',
                                 'User must define reference point location on the image before viewing direction can be estimated')
            return
        geo_point = GeoPoint(self.lat_ref, self.lon_ref, self.altitude_ref, name=self.volcano)
        elev_cam, az_cam, geom_old, map = self.pyplis_worker.meas.meas_geometry.find_viewing_direction(
            pix_x=self.pix_x, pix_y=self.pix_y, pix_pos_err=10, geo_point=geo_point, draw_result=True)
        map.fig.show()
        self.elev = np.round(elev_cam, 1)
        self.azim = np.round(az_cam, 1)

    def close_frame(self):
        """Closes frame"""
        self.frame.destroy()
        self.in_frame = False

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                self.img_canvas.draw()
            else:
                return
        except queue.Empty:
            pass
        self.frame.after(refresh_rate, self.__draw_canv__)


class PlumeBackground(LoadSaveProcessingSettings):
    """
    Generates plume background image figure and settings to adjust parameters
    """
    def __init__(self, pyplis_work=pyplis_worker, generate_frame=False, gui_setts=gui_setts):
        super().__init__()
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_bg = self
        self.frame = None
        self.in_frame = False
        self.fig_size_tau = gui_setts.fig_bg
        self.dpi = gui_setts.dpi
        self.q = queue.Queue()
        self.canvases = [None] * 3

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def start_draw(self, parent):
        """
        Starts canvas drawing in the main thread (so that when generate frame is run the canvas is already drawing).
        This means that if update_plot/generate_frame is run from a thread there won't be any thread-drawing issues
        """
        self.parent = parent
        self.__draw_canv__()

    def initiate_variables(self):
        """Prepares all tk variables"""
        self.vars = {'bg_mode': int,
                     'auto_param': int,
                     'polyfit_2d_thresh': int,
                     'ref_check_lower': float,  # Used to check background region for presence of gas which would hinder background model
                     'ref_check_upper': float,  # Used with ambient_roi from light dilution frame for calculations
                     'ref_check_mode': int,
                     'auto_bg_cmap': int}

        self._bg_mode = tk.IntVar()
        self._auto_param = tk.IntVar()
        self._polyfit_2d_thresh = tk.IntVar()
        self._ref_check_lower = tk.DoubleVar()
        self._ref_check_upper = tk.DoubleVar()
        self._ref_check_mode = tk.IntVar()
        self._auto_bg_cmap = tk.IntVar()
        self._tau_max = tk.DoubleVar()
        self.tau_max = 0.1
        self._tau_min = tk.DoubleVar()
        self.tau_min = -0.1

        self.load_defaults()

    def generate_frame(self):
        """Generates frame and associated widgets"""
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.close_window)
        self.frame.title('Background intensity settings')

        self.in_frame = True

        # Top frame
        self.top_frame = ttk.Frame(self.frame)
        self.top_frame.grid(row=0, column=0, sticky='nsew')

        # Options widget
        self.opt_frame = ttk.LabelFrame(self.top_frame, text='Settings')
        self.opt_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        # Mode option menu
        row = 0
        label = tk.Label(self.opt_frame, text='Intensity threshold (mode 0):')
        label.grid(row=row, column=0, padx=self.pdx, pady=self.pdy, sticky='w')
        spin = ttk.Spinbox(self.opt_frame, textvariable=self._polyfit_2d_thresh, from_=0,
                           to=pyplis_worker.cam_specs._max_DN, increment=1)
        spin.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)

        row += 1
        label = tk.Label(self.opt_frame, text='Pyplis background model:')
        label.grid(row=row, column=0, padx=self.pdx, pady=self.pdy, sticky='w')
        self.mode_opt = ttk.OptionMenu(self.opt_frame, self._bg_mode, self.bg_mode, *pyplis_worker.BG_CORR_MODES)
        self.mode_opt.grid(row=row, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')

        # Automatic reference areas
        row += 1
        self.auto = ttk.Checkbutton(self.opt_frame, text='Automatic reference areas', variable=self._auto_param)
        self.auto.grid(row=row, column=0, columnspan=2, sticky='w')

        # Reference are check
        row += 1
        ref_check_frame = ttk.LabelFrame(self.opt_frame, text='Reference background ROI', relief=tk.RAISED, borderwidth=3)
        ref_check_frame.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        # ref_check_frame.pack(side=tk.TOP, anchor='nw', fill=tk.BOTH, expand=True, padx=5, pady=5)

        check = ttk.Checkbutton(ref_check_frame, text='Use thresholds to omit images', variable=self._ref_check_mode)
        check.grid(row=0, column=0, columnspan=3, sticky='w')

        lab = ttk.Label(ref_check_frame, text='Threshold lower [molecules/cm2]:')
        lab.grid(row=1, column=0, sticky='w')
        thresh_spin = ttk.Spinbox(ref_check_frame, textvariable=self._ref_check_lower, from_=-100, to=0, increment=1,
                                  width=4)
        thresh_spin.grid(row=1, column=1, sticky='w')
        thresh_spin.set('{}'.format(int(self._ref_check_lower.get())))
        lab = ttk.Label(ref_check_frame, text='e16')
        lab.grid(row=1, column=2, sticky='w')

        lab = ttk.Label(ref_check_frame, text='Threshold upper [molecules/cm2]:')
        lab.grid(row=2, column=0, sticky='w')
        thresh_spin = ttk.Spinbox(ref_check_frame, textvariable=self._ref_check_upper, from_=0, to=100, increment=1,
                                  width=4)
        thresh_spin.grid(row=2, column=1, sticky='w')
        thresh_spin.set('{}'.format(int(self._ref_check_upper.get())))
        lab = ttk.Label(ref_check_frame, text='e16')
        lab.grid(row=2, column=2, sticky='w')

        ref_check_frame.grid_columnconfigure(2, weight=1)

        # Buttons
        row += 1
        butt_frame = ttk.Frame(self.opt_frame)
        butt_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        butt = ttk.Button(butt_frame, text='OK', command=self.close_window)
        butt.grid(row=0, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(butt_frame, text='Set As Defaults', command=lambda: self.set_defaults(parent=self.frame))
        butt.grid(row=0, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(butt_frame, text='Run', command=self.run_process)
        butt.grid(row=0, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # ---------------
        # Figure settings
        # ---------------
        self.fig_sett_frame = ttk.LabelFrame(self.top_frame, text='Figure settings')
        self.fig_sett_frame.grid(row=0, column=1, sticky='nw', padx=5, pady=5)

        row = 0
        check = ttk.Checkbutton(self.fig_sett_frame, text='Auto limits', variable=self._auto_bg_cmap,
                                command=self.set_cmap)
        check.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        row += 1

        ttk.Label(self.fig_sett_frame, text='\u03C4 maximum:').grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.fig_sett_frame, textvariable=self._tau_max, width=4,
                           command=self.set_cmap, from_=0, to=1, increment=0.01, format='%.2f')
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        row += 1

        ttk.Label(self.fig_sett_frame, text='\u03C4 minimum:').grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.fig_sett_frame, textvariable=self._tau_min, width=4, command=self.set_cmap,
                           from_=-1, to=0, increment=0.01, format='%.2f')
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        row += 1

        # Build figures
        self._build_figures()

        # Run current background model to load up figures
        self.run_process(reload_seq=False)

        # I'm not sure why, but the program was crashing after run_process and exiting the mainloop.
        # But a call to mainloop here prevents the crash
        tk.mainloop()

    def _build_figures(self):
        """Build figures of backgrounds"""
        self.bottom_frame = ttk.Frame(self.frame)
        self.bottom_frame.grid(row=1, column=0, sticky='nsew')
        self.frame_tau_A = ttk.Frame(self.bottom_frame, relief=tk.RAISED, borderwidth=3)
        self.frame_tau_A.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.frame_tau_B = ttk.Frame(self.bottom_frame, relief=tk.RAISED, borderwidth=3)
        self.frame_tau_B.grid(row=0, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)

        # Make empty figure if we don't have a figure to use
        if not hasattr(self, 'fig_tau_A'):
            self.fig_tau_A = plt.Figure(figsize=self.fig_size_tau, dpi=self.dpi)
            self.fig_tau_B = plt.Figure(figsize=self.fig_size_tau, dpi=self.dpi)

        self.fig_canvas_A = FigureCanvasTkAgg(self.fig_tau_A, master=self.frame_tau_A)
        self.fig_canvas_A.get_tk_widget().pack(side=tk.TOP)
        self.fig_canvas_A.draw()
        # Add toolbar so figures can be saved
        self.toolbar_A = NavigationToolbar2Tk(self.fig_canvas_A, self.frame_tau_A)
        self.toolbar_A.update()
        # self.fig_canvas_A._tkcanvas.pack(side=tk.TOP)
        self.toolbar_A.pack(side=tk.TOP)

        self.fig_canvas_B = FigureCanvasTkAgg(self.fig_tau_B, master=self.frame_tau_B)
        self.fig_canvas_B.get_tk_widget().pack(side=tk.TOP)
        self.fig_canvas_B.draw()
        # Add toolbar so figures can be saved
        self.toolbar_B = NavigationToolbar2Tk(self.fig_canvas_B, self.frame_tau_B)
        self.toolbar_B.update()
        # self.fig_canvas_B._tkcanvas.pack(side=tk.TOP)
        self.toolbar_B.pack(side=tk.TOP)

        self.canvases[0] = self.fig_canvas_A
        self.canvases[1] = self.fig_canvas_B

        if hasattr(self, 'tau_A'):
            self.update_plots(self.tau_A, self.tau_B)

    @property
    def bg_mode(self):
        return self._bg_mode.get()

    @bg_mode.setter
    def bg_mode(self, value):
        self._bg_mode.set(value)

    @property
    def auto_param(self):
        return self._auto_param.get()

    @auto_param.setter
    def auto_param(self, value):
        self._auto_param.set(value)

    @property
    def polyfit_2d_thresh(self):
        return self._polyfit_2d_thresh.get()

    @polyfit_2d_thresh.setter
    def polyfit_2d_thresh(self, value):
        self._polyfit_2d_thresh.set(value)

    @property
    def ref_check_lower(self):
        return self._ref_check_lower.get() * 10 ** 16

    @ref_check_lower.setter
    def ref_check_lower(self, value):
        self._ref_check_lower.set(value / 10 ** 16)

    @property
    def ref_check_upper(self):
        return self._ref_check_upper.get() * 10 ** 16

    @ref_check_upper.setter
    def ref_check_upper(self, value):
        self._ref_check_upper.set(value / 10 ** 16)

    @property
    def ref_check_mode(self):
        return self._ref_check_mode.get()

    @ref_check_mode.setter
    def ref_check_mode(self, value):
        self._ref_check_mode.set(value)

    @property
    def auto_bg_cmap(self):
        return self._auto_bg_cmap.get()

    @auto_bg_cmap.setter
    def auto_bg_cmap(self, value):
        self._auto_bg_cmap.set(value)

    @property
    def tau_max(self):
        return self._tau_max.get()

    @tau_max.setter
    def tau_max(self, value):
        self._tau_max.set(value)

    @property
    def tau_min(self):
        return self._tau_min.get()

    @tau_min.setter
    def tau_min(self, value):
        self._tau_min.set(value)

    def gather_vars(self):
        # BG mode 7 is separate to the pyplis background models so can't be assigned to plume_bg.mode
        # It is instead assigned to the bg_pycam flag, which overpowers plume_bg.mode
        if self.bg_mode == 7:
            pyplis_worker.bg_pycam = True
        else:
            pyplis_worker.plume_bg.mode = self.bg_mode
            pyplis_worker.bg_pycam = False
        pyplis_worker.auto_param_bg = self.auto_param
        pyplis_worker.polyfit_2d_mask_thresh = self.polyfit_2d_thresh
        pyplis_worker.ref_check_lower = self.ref_check_lower
        pyplis_worker.ref_check_upper = self.ref_check_upper
        pyplis_worker.ref_check_mode = self.ref_check_mode

    def run_process(self, reload_seq=True):
        """Main processing for background modelling and displaying the results"""
        self.gather_vars()
        pyplis_worker.model_background()
        self.frame.attributes('-topmost', 1)
        self.frame.attributes('-topmost', 0)
        if reload_seq:
            pyplis_worker.load_sequence(pyplis_worker.img_dir, plot=True, plot_bg=False)

    def set_cmap(self, draw=True):
        """Sets colourmap of figures"""
        # For each band we adjust the figure scales
        for band in ['A', 'B']:
            fig = getattr(self, 'fig_tau_{}'.format(band))
            img = getattr(self, 'tau_{}'.format(band))
            ax_img = fig.axes[0].get_images()[0]
            ax_horz = fig.axes[2]
            ax_vert = fig.axes[1]

            if self.auto_bg_cmap:
                line_h = ax_horz.lines[0].get_ydata()
                line_v = ax_vert.lines[0].get_xdata()
                vmin_h = np.nanmin(line_h) * 1.05
                vmax_h = np.nanmax(line_h) * 1.05
                vmin_v = np.nanmin(line_v) * 1.05
                vmax_v = np.nanmax(line_v) * 1.05

                vmin_i = np.nanmin(img.img) * 1.05
                vmax_i = np.nanmax(img.img) * 1.05
            else:
                vmin_i = vmin_h = vmin_v = self.tau_min
                vmax_i = vmax_h = vmax_v = self.tau_max

            # Horizontal cross section axes formatting
            ax_horz.set_ylim([vmin_h, vmax_h])
            ax_horz.yaxis.set_major_locator(plt.MaxNLocator(3))
            ticks = ['{:.2f}'.format(x) for x in ax_horz.get_yticks()]
            ax_horz.set_yticklabels(ticks)

            # Vertical cross-section axes formatting
            ax_vert.set_xlim([vmin_v, vmax_v])
            ax_vert.xaxis.set_major_locator(plt.MaxNLocator(3))
            ticks = ['{:.2f}'.format(x) for x in ax_vert.get_xticks()]
            ax_vert.set_xticklabels(ticks)

            # Image colourmap setting
            cmap = shifted_color_map(vmin_i, vmax_i)
            ax_img.set_clim([vmin_i, vmax_i])
            ax_img.set_cmap(cmap)

        if draw:
            self.q.put(1)

    def update_plots(self, tau_A, tau_B):
        """Updates plots"""
        self.tau_A = tau_A
        self.tau_B = tau_B
        fig_A = pyplis_worker.plume_bg.plot_tau_result(tau_A)
        fig_B = pyplis_worker.plume_bg.plot_tau_result(tau_B)

        figs = {'A': fig_A, 'B': fig_B}

        for i, band in enumerate(['A', 'B']):
            fig = figs[band]
            setattr(self, 'fig_tau_{}'.format(band), fig)
            # Adjust figure size
            fig.set_size_inches(self.fig_size_tau[0], self.fig_size_tau[1], forward=True)
            fig.set_facecolor(fig_face_colour)
            ax_img = fig.axes[0]

            # If we are in_frame then we update the plot. Otherwise we leave it and when generate_frame is next called
            # the updated plot will be built
            if self.in_frame:
                fig_canvas = getattr(self, 'fig_canvas_{}'.format(band))
                fig_canvas.get_tk_widget().destroy()

                fig_canvas = FigureCanvasTkAgg(fig, master=getattr(self, 'frame_tau_{}'.format(band)))
                setattr(self, 'fig_canvas_{}'.format(band), fig_canvas)
                fig_canvas.get_tk_widget().pack(side=tk.TOP)
                self.canvases[i] = fig_canvas

                # Add toolbar so figures can be saved
                getattr(self, 'toolbar_{}'.format(band)).pack_forget()
                toolbar = NavigationToolbar2Tk(fig_canvas, getattr(self, 'frame_tau_{}'.format(band)))
                toolbar.update()
                # fig_canvas._tkcanvas.pack(side=tk.TOP)
                toolbar.pack(side=tk.TOP)
                setattr(self, 'toolbar_{}'.format(band), toolbar)
                ax_img.set_aspect('auto', anchor='C')

        # Set colourmaps
        self.set_cmap(draw=False)

        if self.in_frame:
            self.q.put(1)

    def close_window(self):
        """Restore current settings"""
        if pyplis_worker.bg_pycam:
            self.bg_mode = 7
        else:
            self.bg_mode = pyplis_worker.plume_bg.mode
        self.auto_param = pyplis_worker.auto_param_bg
        self.polyfit_2d_thresh = pyplis_worker.polyfit_2d_mask_thresh
        self.ref_check_lower = pyplis_worker.ref_check_lower
        self.ref_check_upper = pyplis_worker.ref_check_upper
        self.ref_check_mode = pyplis_worker.ref_check_mode
        self.frame.destroy()
        self.in_frame = False

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            if self.in_frame:
                update = self.q.get(block=False)
                if update == 1:
                    if self.in_frame:
                        for canvas in self.canvases:
                            try:
                                canvas.draw()
                            except AttributeError:
                                pass
                else:
                    pass
        except queue.Empty:
            pass
        self.parent.after(refresh_rate, self.__draw_canv__)


class ProcessSettings(LoadSaveProcessingSettings):
    """
    Object for holding the current processing settings, such as update plots iteratively, method of retrievals etc

    To add a new variable:
    1. Add it as a tk variable in initiate variables
    2. Add it to the vars dictionary, along with its type
    3. Add its associated widgets
    4. Add its associated property with get and set options
    5. Add its action to gather_vars()
    6. Add its reset to close_window()
    7. Add its default value to processing_setting_defaults.txt

    :param parent: tk.Frame     Parent frame
    :param generate_frame: bool   Defines whether the frame is generated on instantiation
    """
    def __init__(self, parent=None, generate_frame=False):
        super().__init__()
        self.parent = parent
        self.frame = None
        self.in_frame = False

        self.path_str_length = 50
        self.path_widg_length = self.path_str_length + 2

        # Generate GUI if requested
        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def initiate_variables(self):
        """
        Initiates tk variables to startup values
        :return:
        """
        # List of all variables to be read in and saved
        self.vars = {'plot_iter': int,
                     'bg_A': str,
                     'bg_B': str,
                     'dark_img_dir': str,
                     'dark_spec_dir': str,
                     'cell_cal_dir': str,
                     'cal_type_int': int,        # 0 = cell, 1 = doas, 2 = cell + doas
                     'use_sensitivity_mask': int,
                     'use_light_dilution': int,
                     'min_cd': float,
                     'buff_size': int,
                     'save_opt_flow': int       # If True, optical flow is saved to buffer (takes up more space)
                     }

        self._plot_iter = tk.IntVar()
        self._bg_A = tk.StringVar()
        self._bg_B = tk.StringVar()
        self._dark_img_dir = tk.StringVar()
        self._dark_spec_dir = tk.StringVar()
        self._cell_cal_dir = tk.StringVar()
        self._cal_type = tk.StringVar()
        self.cal_opts = ['Cell', 'DOAS', 'Cell + DOAS']
        self._use_sensitivity_mask = tk.IntVar()
        self._use_light_dilution = tk.IntVar()
        self._min_cd = tk.DoubleVar()
        self._buff_size = tk.IntVar()
        self._save_opt_flow = tk.IntVar()

        # Load defaults from file
        self.load_defaults()

    def generate_frame(self):
        """
        Builds tkinter frame for settings
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.close_window)
        self.frame.title('Post-processing settings')
        self.in_frame = True

        path_frame = ttk.LabelFrame(self.frame, text='Setup paths')
        path_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        row = 0

        # Background img A directory
        label = ttk.Label(path_frame, text='On-band background:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.bg_A_label = ttk.Label(path_frame, text=self.bg_A_short, width=self.path_widg_length, anchor='e')
        self.bg_A_label.grid(row=row, column=1, sticky='e', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(path_frame, text='Choose file', command=lambda: self.get_bg_file('A'))
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Background img B directory
        label = ttk.Label(path_frame, text='Off-band background:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.bg_A_label = ttk.Label(path_frame, text=self.bg_B_short, width=self.path_widg_length, anchor='e')
        self.bg_A_label.grid(row=row, column=1, sticky='e', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(path_frame, text='Choose file', command=lambda: self.get_bg_file('B'))
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Dark directory
        label = ttk.Label(path_frame, text='Dark image directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.dark_img_label = ttk.Label(path_frame, text=self.dark_dir_short, width=self.path_widg_length, anchor='e')
        self.dark_img_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(path_frame, text='Choose Folder', command=self.get_dark_img_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Dark spec directory
        label = ttk.Label(path_frame, text='Dark spectrum directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.dark_spec_label = ttk.Label(path_frame, text=self.dark_spec_dir_short, width=self.path_widg_length, anchor='e')
        self.dark_spec_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(path_frame, text='Choose Folder', command=self.get_dark_spec_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Cell calibration directory
        label = ttk.Label(path_frame, text='Cell calibration directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.cell_cal_label = ttk.Label(path_frame, text=self.cell_cal_dir_short, width=self.path_widg_length,
                                         anchor='e')
        self.cell_cal_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(path_frame, text='Choose Folder', command=self.get_cell_cal_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Processing
        settings_frame = ttk.LabelFrame(self.frame, text='Processing parameters', borderwidth=5)
        settings_frame.grid(row=1, column=0, sticky='nsw', padx=5, pady=5)
        row = 0

        # Optical flow to buffer checkbutton
        self.opt_check = ttk.Checkbutton(settings_frame, text='Save optical flow to buffer',
                                         variable=self._save_opt_flow)
        self.opt_check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        row += 1

        # Minimum column density used in analysis
        lab = ttk.Label(settings_frame, text='Buffer size [images]:')
        lab.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        buff_spin = ttk.Spinbox(settings_frame, textvariable=self._buff_size, from_=1, to=2000, increment=10,
                                  width=4)
        buff_spin.grid(row=row, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Minimum column density used in analysis
        lab = ttk.Label(settings_frame, text='Min. CD analysed [molecules/cm²]:')
        lab.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        ans_frame = ttk.Frame(settings_frame)
        ans_frame.grid(row=row, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        thresh_spin = ttk.Spinbox(ans_frame, textvariable=self._min_cd, from_=0, to=100, increment=1,
                                  width=4)
        thresh_spin.grid(row=0, column=0, sticky='nsew')
        thresh_spin.set('{}'.format(int(self._min_cd.get())))
        lab = ttk.Label(ans_frame, text='e16')
        lab.grid(row=0, column=1, sticky='e')
        ans_frame.grid_columnconfigure(0, weight=1)
        row += 1

        # Calibration type
        label = ttk.Label(settings_frame, text='Calibration method:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.cal_type_widg = ttk.OptionMenu(settings_frame, self._cal_type, self.cal_type, *self.cal_opts,
                                            command=self.update_sens_mask)
        self.cal_type_widg.configure(width=15)
        self.cal_type_widg.grid(row=row, column=1, sticky='e', padx=self.pdx, pady=self.pdy)
        row += 1

        # Use sensitivity mask checkbutton
        self.sens_check = ttk.Checkbutton(settings_frame, text='Use sensitivity mask',
                                          variable=self._use_sensitivity_mask)
        self.sens_check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        row += 1

        # Use light dilution checkbutton
        self.dil_check = ttk.Checkbutton(settings_frame, text='Light dilution correction',
                                          variable=self._use_light_dilution)
        self.dil_check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        row += 1

        # Plot iteratively checkbutton
        self.plot_check = ttk.Checkbutton(settings_frame, text='Update plots iteratively', variable=self._plot_iter)
        self.plot_check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        row += 1
        self.update_sens_mask()

        self.butt_frame = ttk.Frame(self.frame)
        self.butt_frame.grid(row=2, columnspan=4, sticky='nsew')

        # Save/set buttons
        butt = ttk.Button(self.butt_frame, text='OK', command=self.save_close)
        butt.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(self.butt_frame, text='Cancel', command=self.close_window)
        butt.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(self.butt_frame, text='Apply', command=self.gather_vars)
        butt.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(self.butt_frame, text='Set As Defaults', command=self.set_defaults)
        butt.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

    @property
    def plot_iter(self):
        return self._plot_iter.get()

    @plot_iter.setter
    def plot_iter(self, value):
        self._plot_iter.set(value)

    @property
    def dark_img_dir(self):
        return self._dark_img_dir.get()

    @dark_img_dir.setter
    def dark_img_dir(self, value):
        self._dark_img_dir.set(value)
        if hasattr(self, 'dark_img_label') and self.in_frame:
            self.dark_img_label.configure(text=self.dark_dir_short)

    @property
    def dark_dir_short(self):
        """Returns shorter label for dark directory"""
        return '...' + self.dark_img_dir[-self.path_str_length:]

    @property
    def dark_spec_dir(self):
        return self._dark_spec_dir.get()

    @dark_spec_dir.setter
    def dark_spec_dir(self, value):
        self._dark_spec_dir.set(value)
        if hasattr(self, 'dark_spec_label') and self.in_frame:
            self.dark_spec_label.configure(text=self.dark_spec_dir_short)

    @property
    def dark_spec_dir_short(self):
        """Returns shorter label for dark directory"""
        return '...' + self.dark_spec_dir[-self.path_str_length:]

    @property
    def cell_cal_dir(self):
        return self._cell_cal_dir.get()

    @cell_cal_dir.setter
    def cell_cal_dir(self, value):
        self._cell_cal_dir.set(value)
        if hasattr(self, 'cell_cal_label') and self.in_frame:
            self.cell_cal_label.configure(text=self.cell_cal_dir_short)

    @property
    def cell_cal_dir_short(self):
        """Returns shorter label for dark directory"""
        return '...' + self.cell_cal_dir[-self.path_str_length:]

    @property
    def cal_type(self):
        return self._cal_type.get()

    @cal_type.setter
    def cal_type(self, value):
        self._cal_type.set(value)

    @property
    def cal_type_int(self):
        return self.cal_opts.index(self.cal_type)

    @cal_type_int.setter
    def cal_type_int(self, value):
        self.cal_type = self.cal_opts[value]

    @property
    def use_sensitivity_mask(self):
        return self._use_sensitivity_mask.get()

    @use_sensitivity_mask.setter
    def use_sensitivity_mask(self, value):
        self._use_sensitivity_mask.set(value)

    @property
    def use_light_dilution(self):
        return self._use_light_dilution.get()

    @use_light_dilution.setter
    def use_light_dilution(self, value):
        self._use_light_dilution.set(value)

    @property
    def bg_A(self):
        return self._bg_A.get()

    @bg_A.setter
    def bg_A(self, value):
        self._bg_A.set(value)

    @property
    def bg_A_short(self):
        """Returns shorter label for bg_A file"""
        return '...' + self.bg_A[-self.path_str_length:]

    @property
    def bg_B(self):
        return self._bg_B.get()

    @bg_B.setter
    def bg_B(self, value):
        self._bg_B.set(value)

    @property
    def bg_B_short(self):
        """Returns shorter label for bg_B file"""
        return '...' + self.bg_B[-self.path_str_length:]

    @property
    def min_cd(self):
        return self._min_cd.get() * 10 ** 16

    @min_cd.setter
    def min_cd(self, value):
        self._min_cd.set(value / 10 ** 16)

    @property
    def buff_size(self):
        return self._buff_size.get()

    @buff_size.setter
    def buff_size(self, value):
        self._buff_size.set(value)

    @property
    def save_opt_flow(self):
        return self._save_opt_flow.get()

    @save_opt_flow.setter
    def save_opt_flow(self, value):
        self._save_opt_flow.set(value)

    def update_sens_mask(self, val=None):
        """Updates sensitivity mask depending on currently selected calibration option"""
        if self.cal_type == 'Cell':
            self.sens_check.configure(state=tk.NORMAL)
            self.use_sensitivity_mask = 1
        elif self.cal_type == 'Cell + DOAS':
            self.sens_check.configure(state=tk.DISABLED)
            self.use_sensitivity_mask = 1
        elif self.cal_type == 'DOAS':
            self.sens_check.configure(state=tk.DISABLED)
            self.use_sensitivity_mask = 0

    def get_dark_img_dir(self):
        """Gives user options for retrieving dark image directory"""
        dark_img_dir = filedialog.askdirectory(initialdir=self.dark_img_dir)

        # Pull frame back to the top, as otherwise it tends to hide behind the main frame after closing the filedialog
        self.frame.lift()

        if len(dark_img_dir) > 0:
            self.dark_img_dir = dark_img_dir

    def get_dark_spec_dir(self):
        """Gives user options for retrieving dark spectrum directory"""
        dark_spec_dir = filedialog.askdirectory(initialdir=self.dark_spec_dir)

        # Pull frame back to the top, as otherwise it tends to hide behind the main frame after closing the filedialog
        if self.in_frame:
            self.frame.lift()

        if len(dark_spec_dir) > 0:
            self.dark_spec_dir = dark_spec_dir

    def get_cell_cal_dir(self, set_var=False):
        """
        Gives user options for retrieving cell calibration directory
        :param set_var: bool
            If true, this will set the pyplis_worker value automatically. This means that this function can be used
            from outside of the process_settings widget and the directory will automatically be updated, without
            requiring the OK click from the settings widget which usually instigates gather_vars. This is used by the
            menu widget 'Load cell directory' submenu
        """
        cell_cal_dir = filedialog.askdirectory(initialdir=self.cell_cal_dir)

        # Pull frame back to the top, as otherwise it tends to hide behind the main frame after closing the filedialog
        if self.in_frame:
            self.frame.lift()

        if len(cell_cal_dir) > 0:
            self.cell_cal_dir = cell_cal_dir

        # Update pyplis worker value if requested (done when using submenu selection
        if set_var:
            pyplis_worker.cell_cal_dir = self.cell_cal_dir

    def get_bg_file(self, band):
        """Gives user options for retreiving dark directory"""
        bg_file = filedialog.askopenfilename(initialdir=self.dark_img_dir)

        # Pull frame back to the top, as otherwise it tends to hide behind the main frame after closing the filedialog
        if self.in_frame:
            self.frame.lift()

        if len(bg_file) > 0:
            setattr(self, 'bg_{}'.format(band), bg_file)
            getattr(self, 'bg_{}_label'.format(band)).configure(text=getattr(self, 'bg_{}_short'.format(band)))

    def gather_vars(self):
        """
        Gathers all variables and sets associated objects to the values
        :return:
        """
        pyplis_worker.plot_iter = self.plot_iter
        pyplis_worker.dark_dir = self.dark_img_dir       # Load dark_dir prior to bg images - bg images require dark dir
        pyplis_worker.cell_cal_dir = self.cell_cal_dir
        pyplis_worker.cal_type = self.cal_type_int
        pyplis_worker.use_sensitivity_mask = bool(self.use_sensitivity_mask)
        pyplis_worker.use_light_dilution = bool(self.use_light_dilution)
        doas_worker.dark_dir = self.dark_spec_dir
        pyplis_worker.load_BG_img(self.bg_A, band='A')
        pyplis_worker.load_BG_img(self.bg_B, band='B')
        pyplis_worker.min_cd = self.min_cd
        pyplis_worker.img_buff_size = self.buff_size
        pyplis_worker.save_opt_flow = self.save_opt_flow

    def save_close(self):
        """Gathers all variables and then closes"""
        self.gather_vars()
        self.close_window()
        # Reload sequence, to ensure that the updates have been made
        pyplis_worker.load_sequence(pyplis_worker.img_dir, plot=True, plot_bg=False)

    def close_window(self):
        """Closes window"""
        # Reset values if cancel was pressed, by retrieving them from their associated places
        self.plot_iter = self.vars['plot_iter'](pyplis_worker.plot_iter)
        self.bg_A = pyplis_worker.bg_A_path
        self.bg_B = pyplis_worker.bg_B_path
        self.dark_img_dir = pyplis_worker.dark_dir
        self.dark_spec_dir = doas_worker.dark_dir
        self.cell_cal_dir = pyplis_worker.cell_cal_dir
        self.cal_type_int = pyplis_worker.cal_type
        self.use_sensitivity_mask = int(pyplis_worker.use_sensitivity_mask)
        self.use_light_dilution = int(pyplis_worker.use_light_dilution)
        self.min_cd = pyplis_worker.min_cd
        self.buff_size = pyplis_worker.img_buff_size
        self.save_opt_flow = pyplis_worker.save_opt_flow

        self.in_frame = False
        self.frame.destroy()


class DOASFOVSearchFrame(LoadSaveProcessingSettings):
    """
    Frame to control some basic parameters in pyplis doas fov search and display the results if there are any
    """
    def __init__(self, generate=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), spec_specs=SpecSpecs(),
                 fig_setts=gui_setts):
        self.parent = None
        self.cam_specs = cam_specs
        self.spec_specs = spec_specs
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_doas_fov = self
        self.q = queue.Queue()

        self.dpi = fig_setts.dpi
        self.fig_size_doas_calib_img = fig_setts.fig_doas_calib_img
        self.fig_size_doas_calib_fit = fig_setts.fig_doas_calib_fit

        # Correlation image
        self.img_corr = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_y])
        self.fov = None

        # Flag whether the tk widget has been generated
        self.in_frame = False

        if generate:
            self.generate_frame()
            self.initiate_variables()

    def start_draw(self, parent):
        """
        Starts canvas drawing in the main thread (so that when generate frame is run the canvas is already drawing).
        This means that if update_plot/generate_frame is run from a thread there won't be any thread-drawing issues
        """
        self.parent = parent
        self.__draw_canv__()

    def initiate_variables(self):
        """
        Initiate tkinter variables
        :return:
        """
        self.vars = {'remove_doas_mins': int,
                     'doas_recal': int,
                     'doas_fov_recal_mins': int,
                     'doas_fov_recal': int,
                     'max_doas_cam_dif': int,
                     'fix_fov': int,
                     'maxrad_doas': float}          # Maximum radius in pixels, not degrees, so depends on distance!
        self._maxrad_doas = tk.DoubleVar()
        # self.maxrad_doas = self.spec_specs.fov * 1.1
        self._centre_pix_x = tk.IntVar()
        self._centre_pix_y = tk.IntVar()

        # Time [minutes] before a doas point is removed from the calibration (no need to totally redo calibration,
        # unless changing FOV, actually what I want to define is a limit to the time that a DOAS point contributes to
        # the scatter plot - so eseentially as we step forward in time we lose the oldest DOAS points, as they are
        # less likely to represent the current calibration conditions
        self._remove_doas_mins = tk.IntVar()
        self._doas_recal = tk.BooleanVar()          # If False, DOAS data is never removed (until next day). Otherwise data is removed based on remove_doas_mins
        self._doas_fov_recal_mins = tk.IntVar()     # Recalibration time [minutes] for FOV recalibration
        self._doas_fov_recal = tk.BooleanVar()      # Whether or not DOAS FOV should be recalibrated once it has been set (if False, a first calibration will still be run unless fix_fov is True)
        self._max_doas_cam_dif = tk.IntVar()        # Difference (seconds) between camera and DOAS time - any difference larger than this and the data isn't added to the calibration
        
        self._fov_rad = tk.DoubleVar()
        self._fix_fov = tk.BooleanVar()             # Fixes the FOV - no FOV re-calibration at all will occur

        self.load_defaults()

    def gather_vars(self, message=False):
        """Updates pyplis worker settings"""
        self.pyplis_worker.maxrad_doas = self.maxrad_doas

        self.pyplis_worker.remove_doas_mins = self.remove_doas_mins
        self.pyplis_worker.doas_recal = self.doas_recal
        self.pyplis_worker.doas_fov_recal_mins = self.doas_fov_recal_mins
        self.pyplis_worker.doas_fov_recal = self.doas_fov_recal
        self.pyplis_worker.fix_fov = self.fix_fov
        self.pyplis_worker.max_doas_cam_dif = self.max_doas_cam_dif

        if message:
            messagebox.showinfo('Settings updated',
                                'FOV settings have been updated and will be used on the next processing run',
                                parent=self.frame)
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)

    def generate_frame(self):
        """
        Generates frame
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.frame = tk.Toplevel()
        self.frame.title('DOAS FOV alignment')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        self.in_frame = True

        self._build_opts()
        self._build_figures()

        self.frame_ui.grid(row=0, column=0, sticky='nsew')
        self.frame_scat.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.frame_img.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
        self.frame_ui.grid_columnconfigure(1, weight=1)

    def _build_opts(self):
        """Builds FOV options frame"""
        # Frame holding all UI widgets
        self.frame_ui = tk.Frame(self.frame)

        # ---------------------------------------------------------------
        # Options frame
        self.frame_opts = tk.LabelFrame(self.frame_ui, text='Calibration options', relief=tk.RAISED, borderwidth=2)
        self.frame_opts.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        row = 0

        # Maximum accepted radius for FOV search
        lab = ttk.Label(self.frame_opts, text='Maximum FOV radius [°]:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.frame_opts, textvariable=self._maxrad_doas, from_=0.05, to=10.00, increment=0.05,
                           width=5)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        row += 1

        # Maximum accepted DOAS-cam time difference
        lab = ttk.Label(self.frame_opts, text='Max. DOAS-image time diff. [s]:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.frame_opts, textvariable=self._max_doas_cam_dif, from_=0, to=60, increment=1,
                           width=3)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        row += 1

        # Calibration data removal after defined time
        lab = ttk.Label(self.frame_opts, text='Remove data after [minutes]:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.frame_opts, textvariable=self._remove_doas_mins, from_=1, to=999, increment=1,
                           width=5)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        check_frame = ttk.Frame(self.frame_opts, relief=tk.RAISED)
        check_frame.grid(row=row, column=2, sticky='w', padx=2)
        check = ttk.Checkbutton(check_frame, text='On', variable=self._doas_recal)
        check.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        row += 1

        # DOAS FOV calibration rerun after set amount of time. If this time is larger than the buffer size, the
        # calibration will only be able to run the buffer size as a maximum, so some data may be lost.
        lab = ttk.Label(self.frame_opts, text='Recalibrate FOV after [minutes]:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.frame_opts, textvariable=self._doas_fov_recal_mins, from_=1, to=999, increment=1,
                           width=5)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        check_frame = ttk.Frame(self.frame_opts, relief=tk.RAISED)
        check_frame.grid(row=row, column=2, sticky='w', padx=2)
        check = ttk.Checkbutton(check_frame, text='On', variable=self._doas_fov_recal)
        check.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        row += 1

        butt_frame = ttk.Frame(self.frame_opts)
        butt_frame.grid(row=row, column=0, columnspan=3, sticky='nsew')
        butt_frame.grid_columnconfigure(0, weight=1)
        def_butt = ttk.Button(butt_frame, text='Set as defaults', command=lambda: self.set_defaults(parent=self.frame))
        def_butt.grid(row=0, column=0, sticky='e', padx=2, pady=2)
        app_butt = ttk.Button(butt_frame, text='Update settings', command=lambda: self.gather_vars(message=True))
        app_butt.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # FOV info frame
        # Options frame
        self.frame_info = tk.LabelFrame(self.frame_ui, text='FOV parameters', relief=tk.RAISED, borderwidth=2)
        self.frame_info.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        row = 0

        # Warning label
        lab = tk.Label(self.frame_info, relief=tk.RAISED, anchor='w', justify='left',
                       text='WARNING!!! Editing parameters will\n'
                            'cause direct changes to the calibration')
        lab.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=2, pady=2)
        row += 1

        # Centre FOV pixels
        lab = ttk.Label(self.frame_info, text='FOV centre pixel:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        frame_centre = ttk.Frame(self.frame_info)
        frame_centre.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)
        lab = ttk.Label(frame_centre, text='  x')
        lab.grid(row=0, column=0, sticky='e', pady=2)
        spin = ttk.Spinbox(frame_centre, textvariable=self._centre_pix_x, from_=0, to=self.cam_specs.pix_num_x - 1,
                           increment=1, width=5)
        spin.grid(row=0, column=1, sticky='ew', pady=2)
        lab = ttk.Label(frame_centre, text='  y')
        lab.grid(row=0, column=2, sticky='e', pady=2)
        spin = ttk.Spinbox(frame_centre, textvariable=self._centre_pix_y, from_=0, to=self.cam_specs.pix_num_y - 1,
                           increment=1, width=5)
        spin.grid(row=0, column=3, sticky='ew', pady=2)
        row += 1

        # DOAS FOV
        lab = ttk.Label(self.frame_info, text='FOV radius [°]:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(self.frame_info, textvariable=self._fov_rad, from_=0.05, to=90.00,
                           increment=0.05, width=5)
        spin.grid(row=row, column=1, sticky='ew', padx=2, pady=2)
        row += 1

        # Fixing parameters (no calibration will be performed
        check = ttk.Checkbutton(self.frame_info, text='Fix FOV parameters (no calibration run)', variable=self._fix_fov)
        check.grid(row=row, column=0, columnspan=2, sticky='e', padx=2, pady=2)

        # ---------------------------------------------------------------

    def _build_figures(self):
        """Builds figures for DOAS FOV"""
        # Create figure
        self.fig_img = plt.Figure(figsize=self.fig_size_doas_calib_img, dpi=self.dpi)
        self.ax_img = self.fig_img.subplots(1, 1)
        self.ax_img.set_aspect(1)
        self.fig_img.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)

        self.frame_img = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)
        self.img_canvas = FigureCanvasTkAgg(self.fig_img, master=self.frame_img)
        self.img_canvas.get_tk_widget().pack(side=tk.LEFT)
        # Add toolbar so figures can be saved
        toolbar = NavigationToolbar2Tk(self.img_canvas, self.frame_img)
        toolbar.update()
        self.img_canvas._tkcanvas.pack(side=tk.TOP)

        # Create figure
        self.fig_fit = plt.Figure(figsize=self.fig_size_doas_calib_fit, dpi=self.dpi)
        self.ax_fit = self.fig_fit.subplots(1, 1)

        # Finalise canvas and gridding
        self.frame_scat = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)
        self.fit_canvas = FigureCanvasTkAgg(self.fig_fit, master=self.frame_scat)
        self.fit_canvas.get_tk_widget().pack(side=tk.LEFT)
        # Add toolbar so figures can be saved
        toolbar = NavigationToolbar2Tk(self.fit_canvas, self.frame_scat)
        toolbar.update()
        self.fit_canvas._tkcanvas.pack(side=tk.TOP)

        # If there is a calibration already available we can plot it
        if self.pyplis_worker.calib_pears is not None:
            self.update_plot()
        else:
            self.q.put(1)

    @property
    def maxrad_doas(self):
        """Access to tk variable _maxrad_doas. Defines maximum radius of doas FOV"""
        return self._maxrad_doas.get()

    @maxrad_doas.setter
    def maxrad_doas(self, value):
        self._maxrad_doas.set(value)

    @property
    def remove_doas_mins(self):
        """
        Access to tk variable _remove_doas_mins. Defines number of minutes before a DOAS point is removed from
        calibration
        """
        return self._remove_doas_mins.get()

    @remove_doas_mins.setter
    def remove_doas_mins(self, value):
        self._remove_doas_mins.set(value)

    @property
    def doas_recal(self):
        """
        Access to tk variable _doas_recal. Defines whether calibration points should be removed after time defined by
        remove_doas_mins. If False, previous points remain in calibration for whole day (reset each day whatever)
        """
        return self._doas_recal.get()

    @doas_recal.setter
    def doas_recal(self, value):
        self._doas_recal.set(value)

    @property
    def doas_fov_recal_mins(self):
        """
        Access to tk variable _doas_fov_recal_mins. Defines number of minutes DOAS FOV is recalibrated
        """
        return self._doas_fov_recal_mins.get()

    @doas_fov_recal_mins.setter
    def doas_fov_recal_mins(self, value):
        self._doas_fov_recal_mins.set(value)

    @property
    def doas_fov_recal(self):
        """
        Access to tk variable _doas_fov_recal_mins. Defines number of minutes DOAS FOV is recalibrated
        """
        return self._doas_fov_recal.get()

    @doas_fov_recal.setter
    def doas_fov_recal(self, value):
        self._doas_fov_recal.set(value)

    @property
    def centre_pix_x(self):
        return self._centre_pix_x.get()

    @centre_pix_x.setter
    def centre_pix_x(self, value):
        self._centre_pix_x.set(value)

    @property
    def centre_pix_y(self):
        return self._centre_pix_y.get()

    @centre_pix_y.setter
    def centre_pix_y(self, value):
        self._centre_pix_y.set(value)

    @property
    def centre_coords(self):
        """Access to tk variable _maxrad_doas. Defines maximum radius of doas FOV"""
        return [self._centre_pix_x.get(), self._centre_pix_y.get()]

    @centre_coords.setter
    def centre_coords(self, value):
        try:
            if len(value) == 2:
                self._centre_pix_x.set(value[0])
                self._centre_pix_y.set(value[1])
            else:
                raise  IndexError
        except (IndexError, TypeError):
            print('Error when attempting to set DOAS FOV centre pixel. Aborting setting.\n'
                  'Expected list with length 2, got type: {}'.format(type(value)))

    @property
    def fov_rad(self):
        """FOV radius"""
        return self._fov_rad.get()

    @fov_rad.setter
    def fov_rad(self, value):
        self._fov_rad.set(value)

    @property
    def fix_fov(self):
        """If True, FOV is set and no calibration is run"""
        return self._fix_fov.get()

    @fix_fov.setter
    def fix_fov(self, value):
        self._fix_fov.set(value)

    @property
    def max_doas_cam_dif(self):
        return self._max_doas_cam_dif.get()

    @max_doas_cam_dif.setter
    def max_doas_cam_dif(self, value):
        self._max_doas_cam_dif.set(value)

    def update_vars(self):
        """Updates FOV variables based on pyplis worker"""
        try:
            self.centre_coords = (self.pyplis_worker.doas_fov_x, self.pyplis_worker.doas_fov_y)
            self.fov_rad = self.pyplis_worker.doas_fov_extent
        except AttributeError:
            pass

    def update_plot(self, update_scat=True, update_img=True):
        """
        Updates plot
        :return:
        """
        if not self.in_frame:
            self.generate_frame()
            return

        # Update FOV variables
        self.update_vars()

        # Update calibration scatter plot
        if update_scat:
            try:
                self.ax_fit.cla()
                self.pyplis_worker.calib_pears.plot(add_label_str="Pearson", color="b", ax=self.ax_fit)
            except (AttributeError, ValueError):
                pass

        # Update correlation image plot
        if update_img:
            try:
                self.ax_img.images[-1].colorbar.remove()
            except (AttributeError, IndexError):
                pass
            try:
                self.ax_img.cla()
                # Remove nans from corr_img
                self.pyplis_worker.calib_pears.fov.corr_img.img[np.isnan(
                    self.pyplis_worker.calib_pears.fov.corr_img.img)] = 0
                self.pyplis_worker.calib_pears.fov.plot(ax=self.ax_img)
            except AttributeError:
                return

        self.q.put(1)

    def close_frame(self):
        """
        Closes frame
        :return:
        """
        # Ensure all settings match pyplis worker (if Update settings button hasn't been pressed we don't want to update
        # those settings)
        self.maxrad_doas = self.pyplis_worker.maxrad_doas

        self.remove_doas_mins = self.pyplis_worker.remove_doas_mins
        self.doas_recal = self.pyplis_worker.doas_recal
        self.doas_fov_recal_misn = self.pyplis_worker.doas_fov_recal_mins
        self.doas_fov_recal = self.pyplis_worker.doas_fov_recal
        self.fix_fov = self.pyplis_worker.fix_fov
        self.max_doas_cam_dif = self.pyplis_worker.max_doas_cam_dif

        self.in_frame = False
        self.frame.destroy()

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            if self.in_frame:
                update = self.q.get(block=False)
                if update == 1:
                    if self.in_frame:
                        self.img_canvas.draw()
                        self.fit_canvas.draw()
                else:
                    pass
        except queue.Empty:
            pass
        self.parent.after(refresh_rate, self.__draw_canv__)


class CellCalibFrame:
    """
    Frame to control some basic parameters in pyplis doas fov search and display the results if there are any
    """
    def __init__(self, generate=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), spec_specs=SpecSpecs(),
                 fig_setts=gui_setts, process_setts=None):
        self.cam_specs = cam_specs
        self.spec_specs = spec_specs
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_cell_cal = self
        self.process_setts = process_setts

        self.dpi = fig_setts.dpi
        self.fig_size_cell_fit = fig_setts.fig_cell_fit
        self.fig_size_cell_abs = fig_setts.fig_cell_abs
        self.fig_size_sens_mask = fig_setts.fig_sens_mask

        # Correlation image
        self.cell_abs = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_y])      # Cell absorbance
        self.sens_mask = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_y])     # Sensitivity mask

        # Flag whether the tk widget has been generated
        self.in_frame = False

        if generate:
            self.initiate_variables()
            self.generate_frame()

    def initiate_variables(self):
        """
        Initiate tkinter variables
        :return:
        """
        pass

    def generate_frame(self, update_plot=True):
        """
        Generates frame
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.frame = tk.Toplevel()
        self.frame.title('Cell calibration')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        self.in_frame = True

        self.frame_top = ttk.Frame(self.frame)
        self.frame_top.pack(side=tk.TOP, fill=tk.BOTH)

        # Calibration settings
        self.frame_setts = ttk.LabelFrame(self.frame_top, text='Cell calibration settings', borderwidth=5)
        self.frame_setts.pack(side=tk.LEFT, padx=5, pady=5, anchor='nw')

        label = ttk.Label(self.frame_setts, text='Calibration directory:')
        label.grid(row=0, column=0, sticky='w', padx=5)
        self.cal_dir_lab = ttk.Label(self.frame_setts, text=self.cal_dir_short)
        self.cal_dir_lab.grid(row=0, column=1, padx=5)
        change_butt = ttk.Button(self.frame_setts, text='Change directory',
                                 command=lambda: self.process_setts.get_cell_cal_dir(set_var=True))
        change_butt.grid(row=0, column=2, padx=5)

        # Cropped calibration region
        self._radius = tk.IntVar()
        self.radius = 50
        rad_lab = ttk.Label(self.frame_setts, text='Crop radius [pix]:')
        rad_lab.grid(row=1, column=0, padx=5, sticky='w')
        rad_spin = ttk.Spinbox(self.frame_setts, textvariable=self._radius, from_=1, to=self.cam_specs.pix_num_y,
                               increment=10, command=self.draw_circle)
        rad_spin.grid(row=1, column=1, sticky='w')

        self._cal_crop = tk.IntVar()
        self.cal_crop = int(self.pyplis_worker.cal_crop)
        crop_check = ttk.Checkbutton(self.frame_setts, text='Crop calibration region', variable=self._cal_crop,
                                     command=self.run_cal)
        crop_check.grid(row=2, column=0, sticky='w')
        self._sens_crop = tk.IntVar()
        self.sens_crop = int(self.pyplis_worker.crop_sens_mask)
        crop_sens_check = ttk.Checkbutton(self.frame_setts, text='Crop sensitivity mask', variable=self._sens_crop,
                                     command=self.run_cal)
        crop_sens_check.grid(row=3, column=0, sticky='w')

        self._use_cell_bg = tk.IntVar()
        self.use_cell_bg = int(self.pyplis_worker.use_cell_bg)
        use_cell_bg_check = ttk.Checkbutton(self.frame_setts, text='Automatically set background images',
                                          variable=self._use_cell_bg, command=self.run_cal)
        use_cell_bg_check.grid(row=4, column=0, columnspan=2, sticky='w')

        # Create figure
        self.fig_fit = plt.Figure(figsize=self.fig_size_cell_fit, dpi=self.dpi)
        self.ax_fit = self.fig_fit.subplots(1, 1)
        self.ax_fit.set_title('Cell ppm.m vs apparent absorbance')
        self.ax_fit.set_ylabel('Column Density [ppm.m]')
        self.ax_fit.set_xlabel('Apparent Absorbance')
        self.scat = self.ax_fit.scatter([0, 0], [0, 0],
                                        marker='x', color='white', s=50)
        self.line_plt = self.ax_fit.plot([], [], '-', color='white')

        # Create figure
        self.fig_abs = plt.Figure(figsize=self.fig_size_cell_abs, dpi=self.dpi)
        self.ax_abs = self.fig_abs.subplots(1, 1)
        self.ax_abs.set_aspect(1)
        abs_img = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.abs_im = self.ax_abs.imshow(abs_img, interpolation='none', vmin=0, vmax=np.percentile(abs_img, 99),
                                         cmap=cm.plasma)
        divider = make_axes_locatable(self.ax_abs)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        self.cbar_abs = plt.colorbar(self.abs_im, cax=cax)
        # self.cbar_abs.outline.set_edgecolor('white')
        self.cbar_abs.ax.tick_params(axis='both', direction='in', top='on', right='on')

        # Create figure
        self.fig_mask = plt.Figure(figsize=self.fig_size_sens_mask, dpi=self.dpi)
        self.ax_mask = self.fig_mask.subplots(1, 1)
        self.ax_mask.set_aspect(1)
        self.ax_mask.set_title('Sensitivity mask')
        mask_img = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x])
        self.mask_im = self.ax_mask.imshow(mask_img, interpolation='none',
                                           vmin=np.percentile(mask_img, 1),
                                           vmax=np.percentile(mask_img, 99),
                                           cmap=cm.seismic)
        divider = make_axes_locatable(self.ax_mask)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        self.cbar_mask = plt.colorbar(self.mask_im, cax=cax)
        # self.cbar_mask.outline.set_edgecolor('white')
        self.cbar_mask.ax.tick_params(axis='both', direction='in', top='on', right='on')

        # Finalise canvas and gridding
        self.fit_canvas = FigureCanvasTkAgg(self.fig_fit, master=self.frame_top)
        self.fit_canvas.draw()
        self.fit_canvas.get_tk_widget().pack(side=tk.LEFT)

        self.frame_imgs = ttk.Frame(self.frame)
        self.frame_imgs.pack(side=tk.BOTTOM)
        self.abs_canvas = FigureCanvasTkAgg(self.fig_abs, master=self.frame_imgs)
        self.abs_canvas.draw()
        self.abs_canvas.get_tk_widget().pack(side=tk.LEFT)

        self.mask_canvas = FigureCanvasTkAgg(self.fig_mask, master=self.frame_imgs)
        self.mask_canvas.draw()
        self.mask_canvas.get_tk_widget().pack(side=tk.LEFT)

        # If there is a calibration already available we can plot it
        if self.pyplis_worker.got_cal_cell and update_plot:
            self.update_plot()

    @property
    def cal_dir_short(self):
        try:
            val = '...' + self.pyplis_worker.cell_cal_dir[-50:]
        except:
            val = self.pyplis_worker.cell_cal_dir
        return val

    @property
    def cal_crop(self):
        return self._cal_crop.get()

    @cal_crop.setter
    def cal_crop(self, value):
        self._cal_crop.set(value)

    @property
    def sens_crop(self):
        return self._sens_crop.get()

    @sens_crop.setter
    def sens_crop(self, value):
        self._sens_crop.set(value)

    @property
    def use_cell_bg(self):
        return bool(self._use_cell_bg.get())

    @use_cell_bg.setter
    def use_cell_bg(self, value):
        self._use_cell_bg.set(value)

    @property
    def radius(self):
        return self._radius.get()

    @radius.setter
    def radius(self, value):
        self._radius.set(value)

    def generate_cropped_region(self):
        """
        Generates a bool array defining the values of the calibration image which are to be used in calibration
        :return:
        """
        return (make_circular_mask(self.cam_specs.pix_num_y, self.cam_specs.pix_num_x,
                                   self.cam_specs.pix_num_x / 2, self.cam_specs.pix_num_y / 2, radius=self.radius),
                make_circular_mask_line(self.cam_specs.pix_num_y, self.cam_specs.pix_num_x,
                                        self.cam_specs.pix_num_x / 2, self.cam_specs.pix_num_y / 2, radius=self.radius))

    def draw_circle(self, draw=True, run_cal=True):
        """
        Draws crop circle on ax_abs
        :return:
        """
        # try to remove old circle
        try:
            self.circle.remove()
        except AttributeError:
            pass

        self.circle = plt.Circle((self.cam_specs.pix_num_x / 2, self.cam_specs.pix_num_y / 2), self.radius,
                                 color='white', fill=False)
        self.ax_abs.add_artist(self.circle)

        if draw:
            self.abs_canvas.draw()

        # Update calibration if requested, and only if cal_crop is true (otherwise redrawing the circle won't change
        # the calibration result so this step would unnecessarily slow the program)
        if run_cal and self.cal_crop:
            self.run_cal()

    def gather_vars(self):
        """Updates pyplis worker variables"""
        self.pyplis_worker.use_cell_bg = self.use_cell_bg

    def run_cal(self):
        """Runs calibration with current settings"""
        self.pyplis_worker.use_cell_bg = self.use_cell_bg
        self.pyplis_worker.cal_crop = bool(self.cal_crop)
        self.pyplis_worker.crop_sens_mask = bool(self.sens_crop)
        self.pyplis_worker.cal_crop_region, self.pyplis_worker.cal_crop_line_mask = self.generate_cropped_region()
        self.pyplis_worker.cal_crop_rad = self.radius
        self.pyplis_worker.perform_cell_calibration_pyplis(plot=True, load_dat=False)

    def scale_imgs(self, draw=True):
        """
        Scales two images to 99 percentile
        :param draw:
        :return:
        """
        self.abs_im.set_clim(vmin=0,
                             vmax=np.percentile(self.pyplis_worker.cell_tau_dict[self.pyplis_worker.sens_mask_ppmm], 99))

        # For sensitivity mask we want 0 to be the center as we are using the seismic colour map
        vmax_mask = abs(np.percentile(self.pyplis_worker.sensitivity_mask, 99))
        vmax = vmax_mask + 1
        vmin = -vmax_mask + 1
        self.mask_im.set_clim(vmin=vmin, vmax=vmax)

        if draw:
            self.abs_canvas.draw()
            self.mask_canvas.draw()
            self.mask_canvas.draw()

    def update_plot(self, generate_frame=False):
        """
        Updates plot
        :return:
        """
        # If we aren't in the frame we just leave it for now
        if not self.in_frame:
            if generate_frame:
                self.generate_frame(update_plot=False)
            else:
                return

        # Configure label first
        self.cal_dir_lab.configure(text=self.cal_dir_short)

        # # Plot calibration fit line
        # self.scat.remove()
        # self.scat = self.ax_fit.scatter(self.pyplis_worker.cell_cal_vals[:, 1], self.pyplis_worker.cell_cal_vals[:, 0],
        #                                 marker='x', color='white', s=50)
        # max_tau = np.max(self.pyplis_worker.cell_cal_vals[:, 1])
        # self.line_plt.pop(0).remove()
        # self.line_plt = self.ax_fit.plot([0, max_tau],  self.pyplis_worker.cell_pol([0, max_tau]), '-', color='white')

        self.ax_fit.clear()
        self.ax_fit.set_title('Cell CD vs apparent absorbance')
        self.pyplis_worker.cell_calib.plot_all_calib_curves(self.ax_fit)
        # self.pyplis_worker.cell_calib.plot_calib_curve('aa', ax=self.ax_fit)  # Plotting individual curves - not as tidy
        # self.pyplis_worker.cell_calib.plot_calib_curve('on', ax=self.ax_fit)
        # self.pyplis_worker.cell_calib.plot_calib_curve('off', ax=self.ax_fit)

        # # Plot absorbance of 2nd smallest cell
        abs_img = self.pyplis_worker.cell_tau_dict[self.pyplis_worker.sens_mask_ppmm]
        self.abs_im.set_data(abs_img)
        self.ax_abs.set_title('Cell absorbance: {} ppm.m'.format(self.pyplis_worker.sens_mask_ppmm))
        self.cbar_abs.draw_all()

        # Plot sensitivity mask
        self.mask_im.set_data(self.pyplis_worker.sensitivity_mask)
        self.cbar_mask.draw_all()

        # Set limits for images
        self.scale_imgs(draw=False)

        self.draw_circle(draw=False, run_cal=False)

        self.fit_canvas.draw()
        self.abs_canvas.draw()
        self.mask_canvas.draw()

    def close_frame(self):
        """
        Closes frame
        :return:
        """
        self.in_frame = False
        self.frame.destroy()


class CrossCorrelationSettings(LoadSaveProcessingSettings):
    """
    Cross-correlation settings frame

        To add a new variable:
        1. Add it as a tk variable in initiate variables
        2. Add it to the vars dictionary, along with its type
        3. Add its associated widgets
        4. Add its associated property with get and set options
        5. Add its default value to processing_setting_defaults.txt
    """
    def __init__(self, generate_frame=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), fig_setts=gui_setts):
        self.parent = None
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_cross_corr = self
        self.q = queue.Queue()
        self.cam_specs = cam_specs
        self.fig_setts = fig_setts
        self.dpi = self.fig_setts.dpi
        self.fig_size = self.fig_setts.fig_cross_corr

        self.pdx = 5
        self.pdy = 5

        self.in_frame = False

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def start_draw(self, parent):
        self.parent = parent
        self.__draw_canv__()

    def initiate_variables(self):
        """
        Initiates all tkinter variables
        :return:
        """
        self.vars = {'cross_corr_recal': int
                     }
        self._cross_corr_recal = tk.IntVar()

    def gather_vars(self):
        # Set cross-correlation recalibration
        self.pyplis_worker.cross_corr_recal = self.cross_corr_recal

    def generate_frame(self):
        """
        Generates widget settings frame
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.in_frame = True

        self.frame = tk.Toplevel()
        self.frame.title('Cross-correlation settings')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        row = 0
        self.frame_fig = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)
        self.frame_fig.grid(row=row, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # -------------------------------------------
        # Build figure displaying cross-correlation
        # Make empty figure if we don't have a figure to use
        if not hasattr(self, 'fig'):
            self.fig = plt.Figure(figsize=self.fig_size, dpi=self.dpi)

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP)
        self.fig_canvas.draw()

        # Add toolbar so figures can be saved
        toolbar = NavigationToolbar2Tk(self.fig_canvas, self.frame_fig)
        toolbar.update()
        self.fig_canvas._tkcanvas.pack(side=tk.TOP)
        # -------------------------------------------

    @property
    def cross_corr_recal(self):
        """Time in minutes to rerun cross-correlation analysis"""
        return self._cross_corr_recal.get()

    @cross_corr_recal.setter
    def cross_corr_recal(self, value):
        self._cross_corr_recal.set(value)

    def update_plot(self, ax, info=None):
        """Updates cross-correlation plot"""
        self.ax = ax
        self.fig = ax[0].figure
        # Adjust figure size
        self.fig.set_size_inches(self.fig_size[0], self.fig_size[1], forward=True)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        ax[0].set_ylabel(r'ICA SO$_2$ loading [arbitrary]')
        ax[0].set_xlabel('Time')

        # If we are in_frame then we update the plot. Otherwise we leave it and when generate_frame is next called
        # the updated plot will be built
        if self.in_frame:
            self.fig_canvas.get_tk_widget().destroy()

            self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
            self.fig_canvas.get_tk_widget().pack(side=tk.TOP)
            self.fig_canvas.draw()

            # Add toolbar so figures can be saved
            toolbar = NavigationToolbar2Tk(self.fig_canvas, self.frame_fig)
            toolbar.update()
            self.fig_canvas._tkcanvas.pack(side=tk.TOP)

            self.q.put(1)

    def close_frame(self):
        """
        Closes frame and makes sure current values are correct
        :return:
        """
        self.in_frame = False
        # Close frame
        self.frame.destroy()

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                if self.in_frame:
                    self.fig_canvas.draw()
            else:
                pass
        except queue.Empty:
            pass
        self.parent.after(refresh_rate, self.__draw_canv__)


class OptiFlowSettings(LoadSaveProcessingSettings):
    """
    Optical flow settings frame

        To add a new variable:
        1. Add it as a tk variable in initiate variables
        2. Add it to the vars dictionary, along with its type
        3. Add its associated widgets
        4. Add its associated property with get and set options
        5. Add its default value to processing_setting_defaults.txt
    """
    def __init__(self, generate_frame=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), fig_setts=gui_setts):

        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_opt = self
        self.fig_SO2 = None
        self.q = queue.Queue()
        self.cam_specs = cam_specs
        self.fig_setts = fig_setts
        self.dpi = self.fig_setts.dpi
        self.fig_size = self.fig_setts.fig_SO2
        self.img_tau = None
        self.img_vel = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_x], dtype=np.float)

        self.pdx = 5
        self.pdy = 5

        self.in_frame = False

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def initiate_variables(self):
        """
        Initiates all tkinter variables
        :return:
        """
        self.vars = {'pyr_scale': float,
                     'levels': int,
                     'winsize': int,
                     'iterations': int,
                     'poly_n': int,
                     'poly_sigma': float,
                     'min_length': float,
                     'min_count_frac': float,
                     'hist_dir_gnum_max': int,
                     'hist_dir_binres': int,
                     'hist_sigma_tol': int,
                     'use_roi': int,
                     'roi_abs': list,
                     'flow_glob': int,
                     'flow_raw': int,
                     'flow_histo': int,
                     'flow_hybrid': int,
                     'use_multi_gauss': int,
                     'cross_corr_recal': int
                     }

        self.settings_vars = {'pyr_scale': float,       # Alternative vars dict containing only those values which
                              'levels': int,            # pertain directly to optical flow settings which will be
                              'winsize': int,           # passed straight to the pyplis optical flow object
                              'iterations': int,
                              'poly_n': int,
                              'poly_sigma': float,
                              'min_length': float,
                              'min_count_frac': float,
                              'hist_dir_gnum_max': int,
                              'hist_dir_binres': int,
                              'hist_sigma_tol': int,
                              'use_roi': int,
                              'roi_abs': list
                              }

        self._pyr_scale = tk.DoubleVar()
        self._levels = tk.IntVar()
        self._winsize = tk.IntVar()
        self._iterations = tk.IntVar()
        self._poly_n = tk.IntVar()
        self._poly_sigma = tk.DoubleVar()
        self._min_length = tk.DoubleVar()
        self._min_count_frac = tk.DoubleVar()
        self._hist_dir_gnum_max = tk.IntVar()
        self._hist_dir_binres = tk.IntVar()
        self._hist_sigma_tol = tk.IntVar()
        self._use_roi = tk.IntVar()
        self._use_multi_gauss = tk.IntVar()
        self._cross_corr_recal = tk.IntVar()

        # Flow options
        self._flow_glob = tk.IntVar()
        self._flow_raw = tk.IntVar()
        self._flow_histo = tk.IntVar()
        self._flow_hybrid = tk.IntVar()

        # Load default values
        self.load_defaults()

    def generate_frame(self):
        """
        Generates widget settings frame
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.in_frame = True

        self.frame = tk.Toplevel()
        self.frame.title('Optical flow settings')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        # -------------------------
        # Build optical flow figure
        # -------------------------
        self.frame_fig = tk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)
        self.frame_fig.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        # self.frame.rowconfigure(0, weight=1)
        self._build_fig_img()
        self._build_fig_vel()

        # -----------------
        # Parameter options
        # -----------------
        self.param_frame = ttk.LabelFrame(self.frame, text='Optical flow parameters', borderwidth=5)
        self.param_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        row = 0

        # pyr_scale
        pyr_opt = SpinboxOpt(self.param_frame, name='Pyramid scale (pyr_scale)', var=self._pyr_scale, limits=[0, 1, 0.1], row=row)
        row += 1

        # Levels
        levels = SpinboxOpt(self.param_frame, name='Number of pyramid levels (levels)', var=self._levels, limits=[0, 10, 1], row=row)
        row += 1

        # Levels
        winsize = SpinboxOpt(self.param_frame, name='Averaging window size (winsize)', var=self._winsize,
                             limits=[0, 100, 1], row=row)
        row += 1

        # Levels
        iterations = SpinboxOpt(self.param_frame, name='Number of iterations (iterations)', var=self._iterations, limits=[0, 500, 1], row=row)
        row += 1

        # Levels
        poly_n = SpinboxOpt(self.param_frame, name='Pixel neighbourhood size (poly_n)', var=self._poly_n,
                            limits=[0, 10, 1], row=row)
        row += 1

        # Levels
        poly_sigma = SpinboxOpt(self.param_frame, name='SD of Gaussian smoothing (poly_sigma)', var=self._poly_sigma,
                                limits=[0, 10, 0.1], row=row)
        row += 1

        # ------------------------------
        # Pyplis Analysis options frame
        # ------------------------------
        self.analysis_frame = ttk.LabelFrame(self.frame, text='Analysis parameters', borderwidth=5)
        self.analysis_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        row = 0

        # Minimum vector length
        min_length = SpinboxOpt(self.analysis_frame, name='Minimum vector length analysed', var=self._min_length,
                                limits=[1, 99, 0.1], row=row)
        row += 1

        # Minimum vector length
        min_count_frac = SpinboxOpt(self.analysis_frame, name='Minimum fraction of available vectors',
                                    var=self._min_count_frac, limits=[0, 1, 0.05], row=row)
        row += 1

        # Maximum Gaussian number for histogram fit
        hist_dir_gnum_max = SpinboxOpt(self.analysis_frame, name='Maximum Gaussians for histogram fit',
                                       var=self._hist_dir_gnum_max, limits=[1, 100, 1], row=row)
        row += 1

        # Histogram bin width in degrees
        hist_dir_binres = SpinboxOpt(self.analysis_frame, name='Histogram bin width [deg]', var=self._hist_dir_binres,
                                limits=[1, 180, 1], row=row)
        row += 1

        # Sigma tolerance for mean flow analysis
        hist_dir_binres = SpinboxOpt(self.analysis_frame, name='Sigma tolerance for mean flow analysis',
                                     var=self._hist_sigma_tol, limits=[1, 4, 0.1], row=row)
        row += 1

        # USe ROI
        roi_check = ttk.Checkbutton(self.analysis_frame, text='Use ROI', variable=self._use_roi)
        roi_check.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        gauss_check = ttk.Checkbutton(self.analysis_frame, text='Use Multi-Gauss', variable=self._use_multi_gauss)
        gauss_check.grid(row=row, column=1, sticky='w', padx=2, pady=2)

        # Cross-correlation
        row += 1
        spin = SpinboxOpt(self.analysis_frame, name='Cross-correlation length [minutes]:', var=self._cross_corr_recal,
                          limits=[1, 999, 1], row=row)
        self.xcorr_spin = spin.spin_opt

        # -----------------------------------
        # Flow modes in use
        row += 1
        self.flow_frame = ttk.LabelFrame(self.analysis_frame, text='Flow modes', relief=tk.RAISED, borderwidth=5)
        self.flow_frame.grid(row=row, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        check_glob = ttk.Checkbutton(self.flow_frame, text='Cross-correlation', variable=self._flow_glob)
        check_glob.grid(row=0, column=0, padx=2, pady=2, sticky='w')
        check_glob = ttk.Checkbutton(self.flow_frame, text='Raw', variable=self._flow_raw)
        check_glob.grid(row=0, column=1, padx=2, pady=2, sticky='w')
        check_glob = ttk.Checkbutton(self.flow_frame, text='Histogram', variable=self._flow_histo)
        check_glob.grid(row=0, column=2, padx=2, pady=2, sticky='w')
        check_glob = ttk.Checkbutton(self.flow_frame, text='Hybrid', variable=self._flow_hybrid)
        check_glob.grid(row=0, column=3, padx=2, pady=2, sticky='w')
        # ----------------------------------

        # Set buttons
        butt_frame = ttk.Frame(self.frame)
        butt_frame.grid(row=2, column=0, sticky='nsew')

        # Apply button
        butt = ttk.Button(butt_frame, text='Apply', command=lambda: self.gather_vars(run=True))
        butt.grid(row=0, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # Set default button
        butt = ttk.Button(butt_frame, text='Set As Defaults', command=self.set_defaults)
        butt.grid(row=0, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # Setup thread-safe drawing
        self.__draw_canv__()

    @property
    def pyr_scale(self):
        return self._pyr_scale.get()

    @pyr_scale.setter
    def pyr_scale(self, value):
        self._pyr_scale.set(value)

    @property
    def levels(self):
        return self._levels.get()

    @levels.setter
    def levels(self, value):
        self._levels.set(value)

    @property
    def winsize(self):
        return self._winsize.get()

    @winsize.setter
    def winsize(self, value):
        self._winsize.set(value)

    @property
    def iterations(self):
        return self._iterations.get()

    @iterations.setter
    def iterations(self, value):
        self._iterations.set(value)

    @property
    def poly_n(self):
        return self._poly_n.get()

    @poly_n.setter
    def poly_n(self, value):
        self._poly_n.set(value)

    @property
    def poly_sigma(self):
        return self._poly_sigma.get()

    @poly_sigma.setter
    def poly_sigma(self, value):
        self._poly_sigma.set(value)

    @property
    def min_length(self):
        return self._min_length.get()

    @min_length.setter
    def min_length(self, value):
        self._min_length.set(value)

    @property
    def min_count_frac(self):
        return self._min_count_frac.get()

    @min_count_frac.setter
    def min_count_frac(self, value):
        self._min_count_frac.set(value)

    @property
    def hist_dir_gnum_max(self):
        return self._hist_dir_gnum_max.get()

    @hist_dir_gnum_max.setter
    def hist_dir_gnum_max(self, value):
        self._hist_dir_gnum_max.set(value)

    @property
    def hist_dir_binres(self):
        return self._hist_dir_binres.get()

    @hist_dir_binres.setter
    def hist_dir_binres(self, value):
        self._hist_dir_binres.set(value)

    @property
    def hist_sigma_tol(self):
        return self._hist_sigma_tol.get()

    @hist_sigma_tol.setter
    def hist_sigma_tol(self, value):
        self._hist_sigma_tol.set(value)

    @property
    def use_roi(self):
        return self._use_roi.get()

    @use_roi.setter
    def use_roi(self, value):
        self._use_roi.set(value)

    @property
    def flow_glob(self):
        return self._flow_glob.get()

    @flow_glob.setter
    def flow_glob(self, value):
        self._flow_glob.set(value)

    @property
    def flow_raw(self):
        return self._flow_raw.get()

    @flow_raw.setter
    def flow_raw(self, value):
        self._flow_raw.set(value)

    @property
    def flow_histo(self):
        return self._flow_histo.get()

    @flow_histo.setter
    def flow_histo(self, value):
        self._flow_histo.set(value)

    @property
    def flow_hybrid(self):
        return self._flow_hybrid.get()

    @flow_hybrid.setter
    def flow_hybrid(self, value):
        self._flow_hybrid.set(value)

    @property
    def use_multi_gauss(self):
        return self._use_multi_gauss.get()

    @use_multi_gauss.setter
    def use_multi_gauss(self, value):
        self._use_multi_gauss.set(value)

    @property
    def cross_corr_recal(self):
        """Time in minutes to rerun cross-correlation analysis"""
        return self._cross_corr_recal.get()

    @cross_corr_recal.setter
    def cross_corr_recal(self, value):
        self._cross_corr_recal.set(value)

    def gather_vars(self, run=False):
        """
        Gathers all optical flow settings and updates pyplis worker settings
        :return:
        """
        # Pack all settings into a settings dictionary
        settings = {}
        for key in self.settings_vars:
            settings[key] = getattr(self, key)

        # If not using ROI, we set the roi_abs to full resolution
        if not self.use_roi:
            settings['roi_abs'] = [0, 0, self.cam_specs.pix_num_x, self.cam_specs.pix_num_y]

        # Pass settings to pyplis worker update function
        self.pyplis_worker.update_opt_flow_settings(**settings)
        self.pyplis_worker.use_multi_gauss = self.use_multi_gauss

        # Loop through flow options and set them
        for key in self.pyplis_worker.velo_modes:
            self.pyplis_worker.velo_modes[key] = bool(getattr(self, key))

        # Set cross-correlation recalibration
        self.pyplis_worker.cross_corr_recal = self.cross_corr_recal

        if run:
            self.run_flow()

    def run_flow(self):
        """
        Runs optical flow with current settings and plots the results
        :return:
        """
        # Now run optical flow
        self.pyplis_worker.generate_opt_flow(plot=True)

    def _build_fig_img(self):
        """Build image figure"""
        # Create figure
        self.fig = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_aspect(1)

        # Figure colour
        self.fig.set_facecolor(fig_face_colour)
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Image display
        self.img_tau = self.pyplis_worker.img_tau_prev.img
        self.img_disp = self.ax.imshow(self.img_tau, cmap=cm.Oranges, interpolation='none', vmin=0,
                                       vmax=0.5, aspect='equal')
        self.ax.set_title('Optical flow', color=axes_colour)

        # Draw optical flow
        self.pyplis_worker.opt_flow.draw_flow(ax=self.ax, in_roi=True)
        self.ax.set_xlim([0, self.cam_specs.pix_num_x])
        self.ax.set_ylim([self.cam_specs.pix_num_y, 0])

        # Colorbar
        divider = make_axes_locatable(self.ax)
        self.ax_divider = divider.append_axes("right", size="10%", pad=0.05)
        self.cbar = plt.colorbar(self.img_disp, cax=self.ax_divider)
        self.cbar.outline.set_edgecolor(axes_colour)
        self.cbar.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')


        # Finalise canvas and gridding (canvases are drawn in _build_fig_vel())
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.img_canvas.get_tk_widget().grid(row=0, column=0)

        # Add rectangle crop functionality
        self.rs = widgets.RectangleSelector(self.ax, self.draw_roi, drawtype='box',
                                            rectprops=dict(facecolor='red', edgecolor='blue', alpha=0.5, fill=True))

        # Initial rectangle format
        self.roi_start_x, self.roi_end_x = self.roi_abs[0], self.roi_abs[2]
        self.roi_start_y, self.roi_end_y = self.roi_abs[1], self.roi_abs[3]
        crop_x = self.roi_end_x - self.roi_start_x
        crop_y = self.roi_end_y - self.roi_start_y

        # Draw initial rectangle
        if self.roi_end_x <= self.cam_specs.pix_num_x and self.roi_end_y <= self.cam_specs.pix_num_y:
            self.rect = self.ax.add_patch(patches.Rectangle((self.roi_start_x, self.roi_start_y),
                                                            crop_x, crop_y, edgecolor='green', fill=False, linewidth=3))

    def _build_fig_vel(self):
        """
        Builds figure plotting velocity magnitudes (not directions, although that could be overlayed?)
        :return:
        """
        self.fig_vel = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax_vel = self.fig_vel.subplots(1, 1)
        self.ax_vel.set_aspect(1)

        # Figure colour
        self.fig_vel.set_facecolor(fig_face_colour)
        for child in self.ax_vel.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax_vel.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Image display
        self.img_vel = self.pyplis_worker.velo_img
        # self.img_disp = self.ax.imshow(self.img_tau, cmap=cm.Oranges, interpolation='none', vmin=0,
        #                                vmax=0.5, aspect='auto')
        self.img_vel.show_img(ax=self.ax_vel, tit='Optical flow velocities', cbar=False, cmap='Greens')
        self.img_vel_disp = self.ax_vel.get_images()[0]

        # Colorbar
        divider = make_axes_locatable(self.ax_vel)
        self.ax_vel_divider = divider.append_axes("right", size="10%", pad=0.05)
        self.cbar_vel = plt.colorbar(self.img_vel_disp, cax=self.ax_vel_divider)
        self.cbar_vel.outline.set_edgecolor(axes_colour)
        self.cbar_vel.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Finalise canvas and gridding
        self.vel_canvas = FigureCanvasTkAgg(self.fig_vel, master=self.frame_fig)
        self.vel_canvas.get_tk_widget().grid(row=0, column=1)

        # Draw all canvases
        self.q.put(1)

    def draw_roi(self, eclick, erelease):
        """
        Draws region of interest for calculating optical flow
        :return:
        """
        try:  # Delete previous rectangle, if it exists
            self.rect.remove()
        except AttributeError:
            pass
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        self.roi_start_y, self.roi_end_y = int(eclick.ydata), int(erelease.ydata)
        self.roi_start_x, self.roi_end_x = int(eclick.xdata), int(erelease.xdata)
        crop_Y = erelease.ydata - eclick.ydata
        crop_X = erelease.xdata - eclick.xdata
        self.rect = self.ax.add_patch(patches.Rectangle((self.roi_start_x, self.roi_start_y),
                                                        crop_X, crop_Y, edgecolor='green', fill=False, linewidth=3))

        # Only update roi_abs if use_roi is true
        if self.use_roi:
            self.roi_abs = [self.roi_start_x, self.roi_start_y, self.roi_end_x, self.roi_end_y]

        self.fig.canvas.draw()

    def plt_opt_flow(self):
        """
        Plots optical flow onto figure
        :return:
        """
        pass

    def update_plot(self, draw=True):
        """
        Updates plot
        :param draw:    bool
            If true the draw() function of the canvas is called. Draw should be false if threading the processing
        :return:
        """
        for child in self.ax.get_children():
            if isinstance(child, patches.Circle):
                child.remove()
            elif isinstance(child, mpllines.Line2D):
                child.remove()

        self.img_tau = self.pyplis_worker.img_tau_prev.img

        # Update SO2 image
        self.img_disp.set_data(self.img_tau)
        self.img_disp.set_clim(vmin=0, vmax=self.fig_SO2.vmax_tau)      # Set upper limit using main SO2 image

        # Update flow_lines
        self.pyplis_worker.opt_flow.draw_flow(ax=self.ax, in_roi=True)
        self.ax.set_xlim([0, self.cam_specs.pix_num_x])
        self.ax.set_ylim([self.cam_specs.pix_num_y, 0])

        # Update velocity image
        self.img_vel_disp.set_data(self.pyplis_worker.velo_img.img)
        self.img_vel_disp.set_clim(vmin=0, vmax=np.percentile(self.pyplis_worker.velo_img.img, 99))

        if draw:
            self.q.put(1)

    def close_frame(self):
        """
        Closes frame and makes sure current values are correct
        :return:
        """
        # Set in_frame to False
        self.in_frame = False

        # Ensure current values are correct
        for key in self.settings_vars:
            setattr(self, key, getattr(self.pyplis_worker.opt_flow.settings, key))
        for key in self.pyplis_worker.velo_modes:
            setattr(self, key, self.pyplis_worker.velo_modes[key])
        self.use_multi_gauss = self.pyplis_worker.use_multi_gauss

        # Close drawing function (it is started again on opening the frame
        self.q.put(2)

        # Close frame
        self.frame.destroy()

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                if self.in_frame:
                    self.img_canvas.draw()
                    self.vel_canvas.draw()
            else:
                return
        except queue.Empty:
            pass
        self.frame.after(refresh_rate, self.__draw_canv__)


class LightDilutionSettings(LoadSaveProcessingSettings):
    """
        Light dilution settings frame

            To add a new variable:
            1. Add it as a tk variable in initiate variables
            2. Add it to the vars dictionary, along with its type
            3. Add its associated widgets
            4. Add its associated property with get and set options
            5. Add its default value to processing_setting_defaults.txt
        """

    def __init__(self, generate_frame=False, pyplis_work=pyplis_worker, doas_work=doas_worker,
                 cam_specs=CameraSpecs(), fig_setts=gui_setts):
        self.pyplis_worker = pyplis_work
        self.doas_worker = doas_work
        self.pyplis_worker.fig_dilution = self
        self.q = queue.Queue()
        self.cam_specs = cam_specs
        self.pix_num_x = self.cam_specs.pix_num_x
        self.pix_num_y = self.cam_specs.pix_num_y
        self.fig_setts = fig_setts
        self.dpi = self.fig_setts.dpi
        self.fig_size = self.fig_setts.fig_SO2
        self.fig_scat_size = (self.fig_size[0], self.fig_size[1]-2)
        self.img_A = None
        self.img_B = None
        self.gui = None
        self.init_dir = FileLocator.SPEC_PATH_WINDOWS
        self.dark_spec_path = None
        self.dark_filename = None
        self.clear_spec_path = None
        self.clear_filename = None
        self.max_str_len = 70
        self.path_widg_length = self.max_str_len + 2

        # Set up lines for drawing light dilution extraction and the colours they will be
        self.max_lines = 5
        self.lines_A = [None] * self.max_lines
        self.lines_pyplis = [None] * self.max_lines
        cmap = cm.get_cmap("jet", 100)
        self.line_colours = [cmap(int(f * (100 / (self.max_lines - 1)))) for f in
                             range(self.max_lines)]  # Line colours
        self.line_coords = []

        self.pdx = 5
        self.pdy = 5

        self.in_frame = False

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def add_gui(self, gui):
        """Adds PyCam object as gui (used for Notebook style"""
        self.gui = gui

    def start_draw(self, parent):
        self.parent = parent
        self.__draw_canv__()

    def initiate_variables(self):
        """
        Initiates all tkinter variables
        :return:
        """
        self._current_line = tk.IntVar()
        self.current_line = 1

        self._draw_meth = tk.IntVar()
        self.draw_meth = 1

        self.vars = {'amb_roi': list,
                     'I0_MIN': int,
                     'tau_thresh': float,
                     'dil_recal_time': int}     # Time [minutes] until recalibration of light dilution

        self.amb_roi = [0, 0, 0, 0]
        self._I0_MIN = tk.IntVar()
        self._tau_thresh = tk.DoubleVar()
        self._dil_recal_time = tk.IntVar()

        # Spectrometer light dilution variable
        self._grid_max_ppmm = tk.IntVar()
        self._grid_increment_ppmm = tk.IntVar()

        # Load default values
        self.load_defaults()

    @property
    def I0_MIN(self):
        return self._I0_MIN.get()

    @I0_MIN.setter
    def I0_MIN(self, value):
        self._I0_MIN.set(value)

    @property
    def tau_thresh(self):
        return self._tau_thresh.get()

    @tau_thresh.setter
    def tau_thresh(self, value):
        self._tau_thresh.set(value)

    @property
    def dil_recal_time(self):
        return self._dil_recal_time.get()

    @dil_recal_time.setter
    def dil_recal_time(self, value):
        self._dil_recal_time.set(value)

    @property
    def current_line(self):
        return self._current_line.get()

    @current_line.setter
    def current_line(self, value):
        self._current_line.set(value)

    @property
    def draw_meth(self):
        return self._draw_meth.get()

    @draw_meth.setter
    def draw_meth(self, value):
        self._draw_meth.set(value)

    @property
    def grid_max_ppmm(self):
        return self._grid_max_ppmm.get()

    @grid_max_ppmm.setter
    def grid_max_ppmm(self, value):
        self._grid_max_ppmm.set(value)

    @property
    def grid_increment_ppmm(self):
        return self._grid_increment_ppmm.get()

    @grid_increment_ppmm.setter
    def grid_increment_ppmm(self, value):
        self._grid_increment_ppmm.set(value)

    @property
    def dark_spec_path_short(self):
        try:
            return_str = '...' + self.dark_spec_path[-(self.max_str_len-3):]
        except (IndexError, TypeError):
            return_str = self.dark_spec_path
        return return_str

    @property
    def clear_spec_path_short(self):
        try:
            return_str = '...' + self.clear_spec_path[-(self.max_str_len-3):]
        except (IndexError, TypeError):
            return_str = self.clear_spec_path
        return return_str

    def generate_frame(self):
        """
        Generates widget settings frame
        :return:
        """
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.in_frame = True

        self.frame = tk.Toplevel()
        self.frame.title('Light dilution settings')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)
        self.frame.grid_rowconfigure(1, weight=1)

        # Notebook setup for toggling between camera and sectrometer light dilution settings
        style = self.gui.style
        style.configure('One.TNotebook.Tab', **self.gui.layout_old[0][1])
        self.tabs = ttk.Notebook(self.frame, style='One.TNotebook.Tab')

        self.frame_cam = ttk.Frame(self.tabs)
        self.frame_spec = ttk.Frame(self.tabs)
        self.tabs.add(self.frame_cam, text='Camera')
        self.tabs.add(self.frame_spec, text='Spectrometer')
        self.tabs.pack(side=tk.TOP, fill="both", expand=1)

        self._setup_cam_frame()
        self._setup_spec_frame()

    def _setup_cam_frame(self):
        """Camera frame setup"""
        frame_setts = ttk.LabelFrame(self.frame_cam, text='Settings', borderwidth=3)
        frame_setts.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        lab = ttk.Label(frame_setts, text='Recalibration time [minutes]:')
        lab.grid(row=0, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(frame_setts, textvariable=self._dil_recal_time, from_=0, to=600)
        spin.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)

        # Line settings
        line_frame = tk.Frame(frame_setts, borderwidth=2, relief=tk.RAISED)
        line_frame.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky='nsew')

        row = 0
        lab = ttk.Label(line_frame, text='Edit line:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(line_frame, textvariable=self._current_line, from_=1, to=self.max_lines)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)

        del_butt = ttk.Button(line_frame, text='Delete line', command=lambda: self.del_line(self.current_line - 1))
        del_butt.grid(row=row, column=2, padx=2)

        row += 1
        check_butt_1 = ttk.Radiobutton(line_frame, text='Draw line', variable=self._draw_meth, value=1,
                                       command=self.event_select)
        check_butt_1.grid(row=row, column=0)
        check_butt_2 = ttk.Radiobutton(line_frame, text='Draw ambient region', variable=self._draw_meth, value=0,
                                       command=self.event_select)
        check_butt_2.grid(row=row, column=1)
        self.draw_meth = 1

        right_frame = ttk.Frame(self.frame_cam)
        right_frame.grid(row=0, column=1)
        right_frame.grid_rowconfigure(0, weight=1)

        # # Threshold value for plume mask
        thresh_frame = ttk.LabelFrame(right_frame, text='Image thresholds')
        thresh_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        row = 0
        lab = ttk.Label(thresh_frame, text='Minimum intensity:')
        lab.grid(row=row, column=0, sticky='w', padx=2, pady=2)
        spin = ttk.Spinbox(thresh_frame, textvariable=self._I0_MIN, from_=0, to=self.cam_specs._max_DN, increment=1)
        spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)

        row += 1
        label = ttk.Label(thresh_frame, text='\u03C4 threshold:')
        label.grid(row=row, column=0, sticky='w', padx=2)
        self.ica_edit_spin = ttk.Spinbox(thresh_frame, textvariable=self._tau_thresh, from_=0, to=1,
                                         increment=0.005)
        self.ica_edit_spin.grid(row=row, column=1, sticky='nsew', padx=2, pady=2)

        # Buttons frame
        butt_frame = ttk.Frame(right_frame)
        butt_frame.grid(row=1, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # Apply button
        butt = ttk.Button(butt_frame, text='Apply', command=self.gather_vars)
        butt.grid(row=0, column=0, sticky='ew', padx=2, pady=2)

        # Run button
        butt = ttk.Button(butt_frame, text='Run', command=self.run_dil_corr)
        butt.grid(row=0, column=1, sticky='ew', padx=2, pady=2)

        butt = ttk.Button(butt_frame, text='Save defaults', command=self.set_defaults)
        butt.grid(row=0, column=2, sticky='ew', padx=2, pady=2)

        # -------------------------
        # Build light dilution figure
        # -------------------------
        self.frame_fig = ttk.Frame(self.frame_cam, relief=tk.RAISED, borderwidth=3)
        self.frame_fig.grid(row=1, column=0, columnspan=2, sticky='n', padx=5, pady=5)
        self._build_fig_img()
        self.frame_xtra_figs = ttk.Frame(self.frame_cam, relief=tk.RAISED, borderwidth=3)
        self.frame_xtra_figs.grid(row=0, column=2, rowspan=2, sticky='nsew', padx=5, pady=5)
        self.fig_canvas_xtra = [[None, None], [None, None]]
        # Build figure displaying cross-correlation

        # Make empty figure if we don't have a figure to use
        if not hasattr(self, 'fig_scat_A'):
            self.fig_scat_A = plt.Figure(figsize=self.fig_scat_size, dpi=self.dpi)
            self.fig_scat_B = plt.Figure(figsize=self.fig_scat_size, dpi=self.dpi)

        self.fig_canvas_A = FigureCanvasTkAgg(self.fig_scat_A, master=self.frame_xtra_figs)
        self.fig_canvas_A.get_tk_widget().pack(side=tk.TOP)
        self.fig_canvas_A.draw()
        # Add toolbar so figures can be saved
        self.toolbar_A = NavigationToolbar2Tk(self.fig_canvas_A, self.frame_xtra_figs)
        self.toolbar_A.update()
        # self.fig_canvas_A._tkcanvas.pack(side=tk.TOP)
        self.toolbar_A.pack(side=tk.TOP)

        self.fig_canvas_B = FigureCanvasTkAgg(self.fig_scat_B, master=self.frame_xtra_figs)
        self.fig_canvas_B.get_tk_widget().pack(side=tk.TOP)
        self.fig_canvas_B.draw()
        # Add toolbar so figures can be saved
        self.toolbar_B = NavigationToolbar2Tk(self.fig_canvas_B, self.frame_xtra_figs)
        self.toolbar_B.update()
        # self.fig_canvas_B._tkcanvas.pack(side=tk.TOP)
        self.toolbar_B.pack(side=tk.TOP)

        # self.__draw_canv__()

    def _setup_spec_frame(self):
        """Setup spectrometer frame"""
        # Load spectra
        self.load_spec = ttk.LabelFrame(self.frame_spec, text='Load spectra')
        self.load_spec.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        row = 0
        label = ttk.Label(self.load_spec, text='Dark filename:')
        label.grid(row=row, column=0, padx=self.pdx, pady=self.pdy)
        self.name_dark = ttk.Label(self.load_spec, text=self.dark_spec_path_short, width=self.path_widg_length)
        self.name_dark.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        self.select_dark = ttk.Button(self.load_spec, text='Load Dark Spectrum', command=self.choose_dark_spec)
        self.select_dark.grid(row=row, column=2, sticky='e', padx=self.pdx, pady=self.pdy)
        row += 1
        label = ttk.Label(self.load_spec, text='Clear filename:')
        label.grid(row=row, column=0, padx=self.pdx, pady=self.pdy)
        self.name_clear = ttk.Label(self.load_spec, text=self.clear_spec_path_short, width=self.path_widg_length)
        self.name_clear.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        self.select_clear = ttk.Button(self.load_spec, text='Load Dark Spectrum', command=self.choose_clear_spec)
        self.select_clear.grid(row=row, column=2, sticky='e', padx=self.pdx, pady=self.pdy)

        # TODO Add load grid option - 2 window grids can be loaded and used

        # Options
        self.opt_frame = ttk.LabelFrame(self.frame_spec, text='Grid Settings')
        self.opt_frame.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)

        ttk.Label(self.opt_frame, text='Maximum [ppm.m]:').grid(row=0, column=0, sticky='w', padx=2, pady=2)
        ttk.Label(self.opt_frame, text='Increment [ppm.m]:').grid(row=1, column=0, sticky='w', padx=2, pady=2)
        max_ppmm_spin = ttk.Spinbox(self.opt_frame, textvariable=self._grid_max_ppmm, from_=0, to=20000, increment=10)
        max_ppmm_spin.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        incr_ppmm_spin = ttk.Spinbox(self.opt_frame, textvariable=self._grid_increment_ppmm, from_=0, to=100,
                                     increment=1)
        incr_ppmm_spin.grid(row=1, column=1, sticky='ew', padx=2, pady=2)

    def choose_dark_spec(self):
        """Loads dark spectrum with file prompt"""
        if not isinstance(self.doas_worker, IFitWorker):
            messagebox.showerror('Cannot run light dilution correction',
                                 'Light dilution correction is only available when iFit is used.\n'
                                 'Please switch from DOAS to iFit and then attempt correction')
            return

        dark_spec_path = filedialog.askopenfilename(initialdir=self.init_dir, title='Select dark spectrum',
                                                     filetypes=(("NumPy arrays", "*.npy"),("Text files", "*.txt"),
                                                                ("All files", "*.*")))
        self.load_dark_spec(dark_spec_path)

    def load_dark_spec(self, dark_spec_path):
        """Loads dark spectrum into IFitWorker"""
        filename, ext = os.path.splitext(dark_spec_path)
        if ext == '.npy':
            wavelengths, spectrum = np.load(dark_spec_path)
        elif ext == '.txt':
            wavelengths, spectrum = np.load(dark_spec_path)
        else:
            print('Unrecognised file type for loading clear spectrum')
            return
        self.dark_spec_path = dark_spec_path
        self.dark_filename = os.path.split(dark_spec_path)[-1]
        self.doas_worker.wavelengths = wavelengths
        self.doas_worker.dark_spec = spectrum

    def choose_clear_spec(self):
        """Loads clear spectrum with file prompt"""
        if not isinstance(self.doas_worker, IFitWorker):
            messagebox.showerror('Cannot run light dilution correction',
                                 'Light dilution correction is only available when iFit is used.\n'
                                 'Please switch from DOAS to iFit and then attempt correction')
            return

        clear_spec_path = filedialog.askopenfilename(initialdir=self.init_dir, title='Select clear spectrum',
                                                     filetypes=(("NumPy arrays", "*.npy"),("Text files", "*.txt"),
                                                                ("All files", "*.*")))
        self.load_clear_spec(clear_spec_path)

    def load_clear_spec(self, clear_spec_path):
        """Loads clear spectrum into IFitWorker"""
        filename, ext = os.path.splitext(clear_spec_path)
        if ext == '.npy':
            wavelengths, spectrum = np.load(clear_spec_path)
        elif ext == '.txt':
            wavelengths, spectrum = np.load(clear_spec_path)
        else:
            print('Unrecognised file type for loading clear spectrum')
            return
        self.clear_spec_path = clear_spec_path
        self.clear_filename = os.path.split(clear_spec_path)[-1]
        self.doas_worker.wavelengths = wavelengths
        self.doas_worker.clear_spec_raw = spectrum

    def run_ld_lookup_generator(self):
        """Runs light dilution lookup grid generator"""
        if not isinstance(self.doas_worker, IFitWorker):
            messagebox.showerror('Cannot run light dilution correction',
                                 'Light dilution correction is only available when iFit is used.\n'
                                 'Please switch from DOAS to iFit and then attempt correction')
            return

        # Correct clear spectrum
        self.doas_worker.dark_corr_spectra()
        self.doas_worker.stray_corr_spectra()

        # Run generator
        spec_time = self.doas_worker.get_spec_time(self.clear_filename)
        self.doas_worker.update_grid(self.grid_max_ppmm, self.grid_increment_ppmm)
        self.doas_worker.light_diluiton_curve_generator(self.doas_worker.wavelengths,
                                                        self.doas_worker.clear_spec_corr,
                                                        spec_date=spec_time)

    def _build_fig_img(self):
        """Build figure frame for dilution correction"""
        # Create figure
        self.fig = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_aspect(1)

        # Figure colour
        self.fig.set_facecolor(fig_face_colour)
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(axes_colour)
        self.ax.tick_params(axis='both', colors=axes_colour, direction='in', top='on', right='on')

        # Image display
        self.img_B = self.pyplis_worker.vigncorr_B_warped.img
        self.img_disp = self.ax.imshow(self.img_B, cmap=cm.gray, interpolation='none', vmin=0,
                                       vmax=self.cam_specs._max_DN, aspect='equal')
        self.ax.set_title('Light dilution', color=axes_colour)

        # Finalise canvas and gridding (canvases are drawn in _build_fig_vel())
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.img_canvas.get_tk_widget().grid(row=0, column=0)

        # Bind click event to figure
        self.line_draw = self.fig.canvas.callbacks.connect('button_press_event', self.draw_line)

        # Plot any existing lines
        for i, line in enumerate(self.lines_pyplis):
            if isinstance(line, LineOnImage):
                self.draw_line_obj(line, line_idx=i, draw=False)
        crop_x = self.amb_roi[2] - self.amb_roi[0]
        crop_y = self.amb_roi[3] - self.amb_roi[1]
        self.rect = self.ax.add_patch(patches.Rectangle((self.amb_roi[0], self.amb_roi[1]),
                                                        crop_x, crop_y, edgecolor='green', fill=False, linewidth=3))

        # Draw canvas
        self.q.put(1)

    def event_select(self):
        """Controls plot interactive events"""
        if self.draw_meth:
            self.rs.disconnect_events()
            self.line_draw = self.fig.canvas.callbacks.connect('button_press_event', self.draw_line)
        else:
            self.fig.canvas.mpl_disconnect(self.line_draw)
            self.rs = widgets.RectangleSelector(self.ax, self.draw_roi, drawtype='box',
                                                rectprops=dict(facecolor='red', edgecolor='blue', alpha=0.5, fill=True))

    def draw_line(self, event):
        """Draws line on image for light dilution following click event"""
        # Python indices start at 0, so need to set the correct index for list indexing
        line_idx = self.current_line - 1

        if event.inaxes is self.ax:
            idx = len(self.line_coords)

            # If we are about to overwrite an old line, we first check that the user wants this
            if idx == 1:
                if self.lines_pyplis[line_idx] is not None:
                    resp = messagebox.askokcancel('Overwriting line',
                                                  'You are about to overwrite an existing line.\n'
                                                  'This could affect processing results if it is currently running.',
                                                  parent=self.frame)
                    self.frame.attributes('-topmost', 1)
                    self.frame.attributes('-topmost', 0)
                    if not resp:
                        return

            # If 2 points are already defined we want to clear these points
            if idx == 2:
                self.line_coords = []
                idx = 0  # Resetting index for 'point' definition

            # Update ica_coords with new coordinates
            self.line_coords.append((int(np.round(event.xdata)), int(np.round(event.ydata))))

            # Remove last click point and scatter current click
            try:
                self.scat_ica_point.remove()
            except:
                pass
            self.scat_ica_point = self.ax.scatter(event.xdata, event.ydata, s=50, marker='x', color='k', lw=1)

            self.ax.set_xlim(0, self.pix_num_x - 1)
            self.ax.set_ylim(self.pix_num_y - 1, 0)

            if idx == 1:
                # Delete scatter point
                try:
                    self.scat_ica_point.remove()
                except:
                    pass

                # Delete previous line if it exists
                if self.lines_pyplis[line_idx] is not None:
                    self.del_line(line_idx)

                # Update pyplis line object and objects in pyplis_worker
                lbl = "{}".format(line_idx)
                self.lines_pyplis[line_idx] = LineOnImage(x0=self.line_coords[0][0],
                                                           y0=self.line_coords[0][1],
                                                           x1=self.line_coords[1][0],
                                                           y1=self.line_coords[1][1],
                                                           normal_orientation="right",
                                                           color=self.line_colours[line_idx],
                                                           line_id=lbl)

                self.pyplis_worker.light_dil_lines = self.lines_pyplis

                # Plot pyplis object on figure
                self.lines_A[line_idx] = self.ax.plot([self.line_coords[0][0], self.line_coords[1][0]],
                                [self.line_coords[0][1], self.line_coords[1][1]], color=self.line_colours[line_idx])

                # Auto increment the line we are currently drawing, so you don't have to manually move to next line
                if self.current_line < self.max_lines:
                    self.current_line += 1

            self.q.put(1)

    def draw_line_obj(self, line, line_idx=None, draw=False):
        """
        Draws line from object that is passed to it
        :param line: LineOnImage    Line to be drawn
        """
        if line_idx is None:
            line_idx = self.current_line - 1

        x0, x1 = line.x0, line.x1
        y0, y1 = line.y0, line.y1
        self.lines_A[line_idx] = self.ax.plot([x0, x1],  [y0, y1], color=self.line_colours[line_idx])

        if draw:
            self.q.put(1)

    def add_dil_line(self, line, line_idx=None, force_add=True, run_model=True):
        """Adds light dilution line """
        if line_idx is None:
            line_idx = self.current_line - 1

        # Remove previous line if it is present - only if we are forcing the add
        if self.lines_pyplis[line_idx] is not None:
            if force_add:
                self.del_line(line_idx)
            else:
                return

        # Update pyplis line object and objects in pyplis_worker
        lbl = "{}".format(line_idx)
        self.lines_pyplis[line_idx] = line

        self.pyplis_worker.light_dil_lines = self.lines_pyplis

        # Add line if in frame
        if self.in_frame:
            self.draw_line_obj(line, line_idx=line_idx, draw=True)

        if self.current_line < self.max_lines:
            self.current_line += 1

        # if run_model:
        #     self.run_dil_corr(draw=False)


    def del_line(self, idx):
        """Deletes line"""
        # Get line
        try:
            self.lines_A[idx].pop(0).remove()
            self.lines_A[idx] = None
            self.lines_pyplis[idx] = None
        except AttributeError:
            pass

        # Update pyplis object lines
        self.pyplis_worker.light_dil_lines = self.lines_pyplis

        self.q.put(1)

    def draw_roi(self, eclick, erelease):
        """Draws ROI rectangle for ambient intensity"""
        try:  # Delete previous rectangle, if it exists
            self.rect.remove()
        except AttributeError:
            pass
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        self.roi_start_y, self.roi_end_y = int(eclick.ydata), int(erelease.ydata)
        self.roi_start_x, self.roi_end_x = int(eclick.xdata), int(erelease.xdata)
        crop_Y = erelease.ydata - eclick.ydata
        crop_X = erelease.xdata - eclick.xdata
        self.rect = self.ax.add_patch(patches.Rectangle((self.roi_start_x, self.roi_start_y),
                                                        crop_X, crop_Y, edgecolor='green', fill=False, linewidth=3))

        # Only update roi_abs if use_roi is true
        self.amb_roi = [self.roi_start_x, self.roi_start_y, self.roi_end_x, self.roi_end_y]
        self.pyplis_worker.ambient_roi = self.amb_roi

        self.q.put(1)

    def gather_vars(self):
        """
        Gathers all parameters and sets them to pyplis worker
        :return:
        """
        self.pyplis_worker.ambient_roi = self.amb_roi
        self.pyplis_worker.I0_MIN = self.I0_MIN
        self.pyplis_worker.tau_thresh = self.tau_thresh
        self.pyplis_worker.dil_recal_time = self.dil_recal_time

    def run_dil_corr(self, draw=True):
        """Wrapper for pyplis_worker light dilution correction function"""
        if all(x is None for x in self.lines_pyplis):
            messagebox.showerror('Run error',
                                 'At least one line must be drawn for light dilution modelling to be performed')
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.gather_vars()
        self.pyplis_worker.model_light_dilution(draw=draw)

        # Reload whole sequence to show updated plots
        self.pyplis_worker.load_sequence(self.pyplis_worker.img_dir, plot_bg=False)

    def update_figs(self, figs, draw=True):
        """
        Takes a dictionary of figures and puts them into canvases in current window
        :param figs:    dict    Dictionary of figures
        :param draw:    bool    If True, all plots are drawn
        :return:
        """
        # Loop through figures placing them into the figure array
        for i, band in enumerate(['A', 'B']):
            fig = figs[band]
            setattr(self, 'fig_scat_{}'.format(band), fig)
            # Adjust figure size
            fig.set_size_inches(self.fig_scat_size[0], self.fig_scat_size[1], forward=True)
            ax = fig.axes[0]
            ax.set_ylabel("Terrain radiances ({} band)".format(band))

            # If we are in_frame then we update the plot. Otherwise we leave it and when generate_frame is next called
            # the updated plot will be built
            if self.in_frame:
                fig_canvas = getattr(self, 'fig_canvas_{}'.format(band))
                fig_canvas.get_tk_widget().destroy()

                fig_canvas = FigureCanvasTkAgg(fig, master=self.frame_xtra_figs)
                setattr(self, 'fig_canvas_{}'.format(band), fig_canvas)
                fig_canvas.get_tk_widget().pack(side=tk.TOP)

                # Add toolbar so figures can be saved
                getattr(self, 'toolbar_{}'.format(band)).pack_forget()
                toolbar = NavigationToolbar2Tk(fig_canvas, self.frame_xtra_figs)
                toolbar.update()
                # fig_canvas._tkcanvas.pack(side=tk.TOP)
                toolbar.pack(side=tk.TOP)
                setattr(self, 'toolbar_{}'.format(band), toolbar)

        if draw:
            if figs['basemap'] is not None:
                self.basemap = figs['basemap']
                self.basemap.fig.canvas.set_window_title('Light dilution geometry')
                self.q.put(2)   # For drawing basemap we add to queue

            # If we are not in_frame, but draw was requested, we draw it all
            if not self.in_frame:
                self.generate_frame()
                return

        if self.in_frame:
            self.q.put(1)

    def close_frame(self):
        """
        Closes frame and makes sure current values are correct
        :return:
        """
        # Make sure any non-saved settings are reverted back to pyplis settings
        self.amb_roi = self.pyplis_worker.ambient_roi
        self.I0_MIN = self.pyplis_worker.I0_MIN
        self.tau_thresh = self.pyplis_worker.tau_thresh
        self.dil_recal_time = self.pyplis_worker.dil_recal_time

        self.in_frame = False

        # Close frame
        self.frame.destroy()

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            if update == 1:
                if self.in_frame:
                    self.img_canvas.draw()
            elif update == 2:
                self.basemap.fig.show()
            else:
                pass
        except queue.Empty:
            pass
        self.parent.after(refresh_rate, self.__draw_canv__)