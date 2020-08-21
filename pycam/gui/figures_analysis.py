# -*- coding: utf-8 -*-

"""
Contains all classes associated with building figures for the analysis functions of SO2 cameras
"""

from pycam.gui.cfg import gui_setts
from pycam.setupclasses import CameraSpecs, SpecSpecs, FileLocator
from pycam.cfg import pyplis_worker
from pycam.doas.cfg import doas_worker
from pycam.so2_camera_processor import UnrecognisedSourceError
from pyplis import LineOnImage

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


class SequenceInfo:
    """
    Generates widget containing squence information, which is displayed at the top of the analysis frame
    """
    def __init__(self, parent, generate_widget=True):
        self.parent = parent
        self.frame = ttk.LabelFrame(self.parent, text='Sequence information')

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

        self.img_dir = pyplis_worker.img_dir
        self.num_img_tot = pyplis_worker.num_img_tot
        self.num_img_pairs = pyplis_worker.num_img_pairs

    def generate_widget(self):
        """Builds widget"""
        row = 0
        label = ttk.Label(self.frame, text='Sequence directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.img_dir_lab = ttk.Label(self.frame, text=self.img_dir_short)
        self.img_dir_lab.grid(row=row, column=1, sticky='w', padx=self.pdx, pady=self.pdy)

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
        self.img_dir = pyplis_worker.img_dir
        self.num_img_pairs = pyplis_worker.num_img_pairs
        self.num_img_tot = pyplis_worker.num_img_tot

        self.img_dir_lab.configure(text=self.img_dir_short)
        self.num_img_pairs_lab.configure(text=str(self.num_img_pairs))
        self.num_img_tot_lab.configure(text=str(self.num_img_tot))


class ImageSO2:
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

    def __init__(self, parent, image_tau=None, image_cal=None,
                 pix_dim=(CameraSpecs().pix_num_x, CameraSpecs().pix_num_y)):
        self.parent = parent
        self.image_tau = image_tau
        self.image_cal = image_cal
        setattr(pyplis_worker, 'fig_tau', self)
        self.pix_num_x = pix_dim[0]
        self.pix_num_y = pix_dim[1]
        self.dpi = gui_setts.dpi
        self.fig_size = gui_setts.fig_SO2

        self.specs = CameraSpecs()


        self.max_lines = 5  # Maximum number of ICA lines
        # ------------------------------------------------------------------------------------------
        # TK variables Setup

        # ICA lines
        self._num_ica = tk.IntVar()
        self.num_ica = 1
        self._current_ica = tk.IntVar()
        self.current_ica = 1
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

        # -----------------------------------------------------------------------------------------------

        # Generate main frame for figure
        self.frame = ttk.Frame(self.parent)

        if self.image_tau is None:
            self.image_tau = np.random.random([self.pix_num_y, self.pix_num_x]) * self.specs._max_DN

        # Generate frame options
        self._build_options()

        # Generate analysis options frame
        self._build_analysis()

        # Generate figure
        self._build_fig_img()

        self.frame_opts.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.frame_analysis.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.frame_fig.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.frame.columnconfigure(1, weight=1)

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

    def _build_fig_img(self):
        """Build figure"""
        # Main frame for figure and all associated widgets
        self.frame_fig = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)

        # Create figure
        self.fig = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_aspect(1)

        # Figure colour
        self.fig.set_facecolor('black')
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')

        # Image display
        self.img_disp = self.ax.imshow(self.image_tau, cmap=self.cmap, interpolation='none', vmin=0,
                                       vmax=self.specs._max_DN, aspect='auto')
        self.ax.set_title('SO2 image', color='white')

        # Colorbar
        divider = make_axes_locatable(self.ax)
        self.ax_divider = divider.append_axes("right", size="10%", pad=0.05)
        self.cbar = plt.colorbar(self.img_disp, cax=self.ax_divider)
        self.cbar.outline.set_edgecolor('white')
        self.cbar.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')

        # Plot optical flwo if it is requested at start
        self.plt_opt_flow()

        # Finalise canvas and gridding
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_fig)
        self.img_canvas.draw()
        self.img_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Bind click event to figure
        self.fig.canvas.callbacks.connect('button_press_event', self.ica_draw)

    def _build_analysis(self):
        """Build analysis options"""
        self.frame_analysis = ttk.LabelFrame(self.frame, text='Analysis')

        # Number of lines
        label = ttk.Label(self.frame_analysis, text='Num. ICAs:')
        label.grid(row=0, column=0, sticky='w')
        self.ica_spin = ttk.Spinbox(self.frame_analysis, textvariable=self._num_ica, from_=1, to=self.max_lines,
                                    increment=1, command=self.update_ica_num)
        self.ica_spin.grid(row=0, column=1, sticky='ew')

        # Line to edit
        label = ttk.Label(self.frame_analysis, text='Edit ICA:')
        label.grid(row=1, column=0, sticky='w')
        self.ica_edit_spin = ttk.Spinbox(self.frame_analysis, textvariable=self._current_ica, from_=1, to=self.num_ica,
                                         increment=1)
        self.ica_edit_spin.grid(row=1, column=1, sticky='ew')

        # Flip ICA normal button
        self.ica_flip_butt = ttk.Button(self.frame_analysis, text='Flip ICA normal', command=self.flip_ica_normal)
        self.ica_flip_butt.grid(row=1, column=2, sticky='nsew', padx=2, pady=2)


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

    def update_ica_num(self):
        """Makes necessary changes to update the number of ICA lines"""

        # Edit 'to' of ica_edit_spin spinbox and if current ICA is above number of ICAs we update current ICA
        self.ica_edit_spin.configure(to=self.num_ica)
        if self.current_ica > self.num_ica:
            self.current_ica = self.num_ica

        # Delete any drawn lines over the new number requested if they are present
        ica_num = self.num_ica
        for ica in self.PCS_lines_list[self.num_ica:]:
            if ica is not None:
                self.del_ica(ica_num)
                self.PCS_lines_list[ica_num] = None
            ica_num += 1

    def flip_ica_normal(self):
        """Flips the normal vector of the current ICA"""
        ica_idx = self.current_ica - 1

        # Get line currently being edited
        line = self.PCS_lines_list[ica_idx]
        lbl = "{}".format(ica_idx)

        # If current line is not none we find it's orientation and reverse it
        if line is not None:
            self.del_ica(ica_idx)

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
                    self.del_ica(PCS_idx)

                # Update pyplis line object and objects in pyplis_worker
                lbl = "{}".format(PCS_idx)
                self.PCS_lines_list[PCS_idx] = LineOnImage(x0=self.ica_coords[0][0],
                                                           y0=self.ica_coords[0][1],
                                                           x1=self.ica_coords[1][0],
                                                           y1=self.ica_coords[1][1],
                                                           normal_orientation="right",
                                                           color=self.line_colours[PCS_idx],
                                                           line_id=lbl)

                pyplis_worker.PCS_lines = self.PCS_lines_list

                # Plot pyplis object on figure
                self.PCS_lines_list[PCS_idx].plot_line_on_grid(ax=self.ax, include_normal=1,
                                                               include_roi_rot=True, label=lbl)

                # # Extract ICA values and plotting them
                # self.plot_ica_xsect()

            self.img_canvas.draw()
        else:
            self.messages('Clicked outside axes bounds but inside plot window')

    def del_ica(self, line_num):
        """Searches axis for line object relating to pyplis line object and removes it

        Parameters
        ----------
        line_num: int
            Index of line in PCS_lines_list"""
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

        # Redraw canvas
        self.img_canvas.draw()

    def change_cmap(self, cmap):
        """Change colourmap of image"""
        # Set cmap to new value
        self.img_disp.set_cmap(getattr(cm, cmap))

        # Update canvas
        self.img_canvas.draw()

    def scale_img(self, draw=True):
        """
        Updates cmap scale
        :param draw: bool   Defines whether the image canvas is redrawn after updating cmap
        :return:
        """
        if self.disp_cal:
            # Get vmax either automatically or by defined spinbox value
            if self.auto_ppmm:
                vmax = np.percentile(self.image_tau, 99.99)
            else:
                vmax = self.ppmm_max
        else:
            # Get vmax either automatically or by defined spinbox value
            if self.auto_tau:
                vmax = np.percentile(self.image_tau, 99.99)
            else:
                vmax = self.tau_max

        # Set new limits
        self.img_disp.set_clim(vmin=0, vmax=vmax)

        if draw:
            self.img_canvas.draw()

    def plt_opt_flow(self):
        """Plots optical flow onto figure"""
        pass

    def update_plot(self, img_tau, img_cal=None):
        """
        Updates image figure and all associated subplots
        :param img: np.ndarray  Image array
        :return:
        """
        self.image_tau = img_tau
        self.image_cal = img_cal

        # Disable radiobutton if no calibrated image is present - we can't plot what isn't there...
        if self.image_cal is None:
            self.disp_cal_rad.configure(state=tk.DISABLED)
        else:
            self.disp_cal_rad.configure(state=tk.NORMAL)

        # Update main image display and title
        if self.disp_cal and img_cal is not None:
            self.img_disp.set_data(img_cal)
        else:
            self.img_disp.set_data(img_tau)
        self.scale_img(draw=False)
        self.cbar.draw_all()
        self.img_canvas.draw()


class GeomSettings:
    """
    Creates frame holding all geometry and associated settings required by pyplis. This frame links to the PyplisWorker
    object to update settings when needed.

    This object should be instantiated on startup, so a conf file needs to do this, and then the generate_frame() method
    should be used at the point at which it is required
    """
    def __init__(self, parent=None, geom_path=FileLocator.CAM_GEOM):
        self.parent = parent
        self.frame = None
        self.geom_path = geom_path
        self.filename = None        # Path to file of current settings

        self.pyplis_worker = pyplis_worker

    def initiate_variables(self):
        """Initiates object, and builds frame if parent is a tk.Frame"""
        # Tk Variables
        self._lat = tk.StringVar()
        self._lon = tk.StringVar()
        self._altitude = tk.IntVar()
        self._elev = tk.DoubleVar()
        self._azim = tk.DoubleVar()
        self._volcano = tk.StringVar()

        self.geom_dict = {'lat': None,
                          'lon': None,
                          'altitude': None,
                          'elev': None,
                          'azim': None}    # List of attributes

        # Setting start values of variables
        with open(FileLocator.DEFAULT_GEOM, 'r') as f:
            self.filename = f.readline()
        self.load_instrument_setup(self.filename)

        # If class is instantiated with a Frame we generate the widgets
        if self.parent is not None:
            self.generate_frame(self.parent)

    def generate_frame(self, parent):
        """Generates the GUI for this frame, given the parent frame. This method of generating the frame means that
        the object can exist, with variables being instantiated, prior to building the frame"""
        self.parent = parent
        self.frame = ttk.Frame(self.parent)
        # ----------------------------------------------------------------------
        # Tkinter widgets
        row = 0

        label = ttk.Label(self.frame, text='Latitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame, width=10, textvariable=self._lat)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame, text='Longitude [dec]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame, width=10, textvariable=self._lon)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame, text='Altitude [m]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame, textvariable=self._altitude, from_=0, to=8000, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame, text='Elevation angle [°]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame, format='%.1f', textvariable=self._elev,
                              from_=-90, to=90, increment=0.1, width=4)
        spinbox.set('{:.1f}'.format(self.elev))
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame, text='Azimuth [°]:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        spinbox = ttk.Spinbox(self.frame, textvariable=self._azim, from_=0, to=359, increment=1, width=4)
        spinbox.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        label = ttk.Label(self.frame, text='Volcano:')
        label.grid(row=row, column=0, padx=2, pady=2, sticky='w')
        entry = ttk.Entry(self.frame, width=10, textvariable=self._volcano)
        entry.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame, text='Update settings', command=self.update_geom)
        button.grid(row=row, column=0, padx=2, pady=2, sticky='nsew')

        button = ttk.Button(self.frame, text='Save settings', command=self.save_instrument_setup)
        button.grid(row=row, column=1, padx=2, pady=2, sticky='nsew')

        button = ttk.Button(self.frame, text='Load settings', command=self.load_instrument_setup)
        button.grid(row=row, column=2, padx=2, pady=2, sticky='nsew')

        button = ttk.Button(self.frame, text='Set as default', command=self.set_default_instrument_setup)
        button.grid(row=row, column=3, padx=2, pady=2, sticky='nsew')

        row += 1
        button = ttk.Button(self.frame, text='Draw geometry', command=self.draw_geometry)
        button.grid(row=row, column=0, padx=2, pady=2, sticky='nsew')

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
                elif key == 'altitude':
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


class LoadSaveProcessingSettings:
    """Base Class for loading and saving processing settings methods"""

    def __init__(self):
        self.vars = {}  # Variables dictionary with kay-value pairs containing attribute name and variable type

        self.pdx = 2
        self.pdy = 2

    def gather_vars(self):
        """
        Place holder for inheriting classes. It gathers all tk variables and sets associated variables to their
        values
        """
        pass

    def load_defaults(self):
        """Loads default settings"""
        filename = FileLocator.PROCESS_DEFAULTS

        with open(filename, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue

                # Try to split line into key and value, if it fails this line is not used
                try:
                    key, value = line.split('=')
                except ValueError:
                    continue

                # If the value is one we edit, we extract the value
                if key in self.vars.keys():
                    if self.vars[key] is str:
                        value = value.split('\'')[1]
                    else:
                        value = self.vars[key](value.split('\n')[0].split('#')[0])
                    setattr(self, key, value)

        # Update all objects finally
        self.gather_vars()

    def set_defaults(self):
        """Sets current values as defaults"""
        # First set this variables
        self.gather_vars()

        # Ask user to define filename for saving geometry settings
        filename = FileLocator.PROCESS_DEFAULTS
        filename_temp = filename.replace('.txt', '_temp.txt')

        # Open file object and write all attributes to it
        with open(filename_temp, 'w') as f_temp:
            with open(filename, 'r') as f:
                for line in f:
                    if line[0] == '#':
                        f_temp.write(line)
                        continue

                    # If we can't split the line, then we just write it as it as, as it won't contain anything useful
                    try:
                        key, value = line.split('=')
                    except ValueError:
                        f_temp.write(line)
                        continue

                    # If the value is one we edit, we extract the value
                    if key in self.vars.keys():
                        if self.vars[key] is str:  # If string we need to write the value within quotes
                            f_temp.write('{}={}\n'.format(key, '\'' + getattr(self, key) + '\''))
                        else:
                            f_temp.write('{}={}\n'.format(key, getattr(self, key)))
                    else:
                        f_temp.write(line)

        # Finally, overwrite old default file with new file
        os.replace(filename_temp, filename)


class PlumeBackground(LoadSaveProcessingSettings):
    """
    Generates plume background image figure and settings to adjust parameters
    """
    def __init__(self, generate_frame=False):
        super().__init__()
        self.frame = None

        if generate_frame:
            self.initiate_variables()
            self.generate_frame()

    def initiate_variables(self):
        """Prepares all tk variables"""
        self.vars = {'bg_mode': int,
                     'auto_param': int}

        self._bg_mode = tk.IntVar()
        self._auto_param = tk.IntVar()

        self.load_defaults()

    def generate_frame(self):
        """Generates frame and associated widgets"""
        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.close_window)
        self.frame.title('Background intensity settings')

        # Options widget
        self.opt_frame = ttk.LabelFrame(self.frame, text='Settings')
        self.opt_frame.pack(side=tk.LEFT)

        # Mode option menu
        row = 0
        label = tk.Label(self.opt_frame, text='Pyplis background model:')
        label.grid(row=row, column=0, padx=self.pdx, pady=self.pdy, sticky='w')
        self.mode_opt = ttk.OptionMenu(self.opt_frame, self._bg_mode,
                                      pyplis_worker.plume_bg.mode, *pyplis_worker.BG_CORR_MODES)
        self.mode_opt.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)

        # Automatic reference areas
        row += 1
        self.auto = ttk.Checkbutton(self.opt_frame, text='Automatic reference areas', variable=self._auto_param)
        self.auto.grid(row=row, column=0, columnspan=2, sticky='w')

        # Buttons
        row += 1
        butt_frame = ttk.Frame(self.opt_frame)
        butt_frame.grid(row=row, column=0, columnspan=2)

        butt = ttk.Button(butt_frame, text='Apply', command=self.gather_vars)
        butt.grid(row=0, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(butt_frame, text='Set As Defaults', command=self.set_defaults)
        butt.grid(row=0, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(butt_frame, text='Run', command=self.run_process)
        butt.grid(row=0, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)

        # Run current background model to load up figures
        self.run_process()

        # I'm not sure why, but the program was crashing after run_process and exiting the mainloop.
        # But a call to mainloop here prevents the crash
        tk.mainloop()

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

    def gather_vars(self):
        pyplis_worker.plume_bg.mode = self.bg_mode
        pyplis_worker.auto_param_bg = self.auto_param

    def run_process(self):
        """Main processing for background modelling and displaying the results"""
        self.gather_vars()
        pyplis_worker.model_background()
        self.frame.attributes('-topmost', 1)
        self.frame.attributes('-topmost', 0)

    def close_window(self):
        """Restore current settings"""
        self.bg_mode = pyplis_worker.plume_bg.mode
        self.auto_param = pyplis_worker.auto_param_bg
        self.frame.destroy()


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
                     'use_sensitivity_mask': int
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

        # Load defaults from file
        self.load_defaults()

    def generate_frame(self):
        """
        Builds tkinter frame for settings
        :return:
        """
        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.close_window)
        self.frame.title('Post-processing settings')
        self.in_frame = True

        row = 0

        # Background img A directory
        label = ttk.Label(self.frame, text='On-band background:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.bg_A_label = ttk.Label(self.frame, text=self.bg_A_short, width=self.path_widg_length, anchor='e')
        self.bg_A_label.grid(row=row, column=1, sticky='e', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.frame, text='Choose file', command=lambda: self.get_bg_file('A'))
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Background img B directory
        label = ttk.Label(self.frame, text='Off-band background:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.bg_A_label = ttk.Label(self.frame, text=self.bg_B_short, width=self.path_widg_length, anchor='e')
        self.bg_A_label.grid(row=row, column=1, sticky='e', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.frame, text='Choose file', command=lambda: self.get_bg_file('B'))
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Dark directory
        label = ttk.Label(self.frame, text='Dark image directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.dark_img_label = ttk.Label(self.frame, text=self.dark_dir_short, width=self.path_widg_length, anchor='e')
        self.dark_img_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.frame, text='Choose Folder', command=self.get_dark_img_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Dark spec directory
        label = ttk.Label(self.frame, text='Dark spectrum directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.dark_spec_label = ttk.Label(self.frame, text=self.dark_spec_dir_short, width=self.path_widg_length, anchor='e')
        self.dark_spec_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.frame, text='Choose Folder', command=self.get_dark_spec_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Cell calibration directory
        label = ttk.Label(self.frame, text='Cell calibration directory:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.cell_cal_label = ttk.Label(self.frame, text=self.cell_cal_dir_short, width=self.path_widg_length,
                                         anchor='e')
        self.cell_cal_label.grid(row=row, column=1, padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.frame, text='Choose Folder', command=self.get_cell_cal_dir)
        butt.grid(row=row, column=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Calibration type
        label = ttk.Label(self.frame, text='Calibration method:')
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.cal_type_widg = ttk.OptionMenu(self.frame, self._cal_type, self.cal_type, *self.cal_opts)
        self.cal_type_widg.configure(width=15)
        self.cal_type_widg.grid(row=row, column=1, sticky='e')
        row += 1

        # Plot iteratively checkbutton
        self.plot_check = ttk.Checkbutton(self.frame, text='Use sensitivity mask', variable=self._use_sensitivity_mask)
        self.plot_check.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        # Plot iteratively checkbutton
        self.plot_check = ttk.Checkbutton(self.frame, text='Update plots iteratively', variable=self._plot_iter)
        self.plot_check.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row += 1

        self.butt_frame = ttk.Frame(self.frame)
        self.butt_frame.grid(row=row, columnspan=4, sticky='nsew')

        # Save/set buttons
        butt = ttk.Button(self.butt_frame, text='Cancel', command=self.close_window)
        butt.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

        butt = ttk.Button(self.butt_frame, text='OK', command=self.save_close)
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
        doas_worker.dark_dir = self.dark_spec_dir
        pyplis_worker.load_BG_img(self.bg_A, band='A')
        pyplis_worker.load_BG_img(self.bg_B, band='B')

    def save_close(self):
        """Gathers all variables and then closes"""
        self.gather_vars()
        self.close_window()

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

        self.in_frame = False
        self.frame.destroy()


class DOASFOVSearchFrame:
    """
    Frame to control some basic parameters in pyplis doas fov search and display the results if there are any
    """
    def __init__(self, generate=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), spec_specs=SpecSpecs(),
                 fig_setts=gui_setts):
        self.cam_specs = cam_specs
        self.spec_specs = spec_specs
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_doas_fov = self

        self.dpi = fig_setts.dpi
        self.fig_size_doas_calib_img = fig_setts.fig_doas_calib_img
        self.fig_size_doas_calib_fit = fig_setts.fig_doas_calib_fit

        # Correlation image
        self.img_corr = np.zeros([self.cam_specs.pix_num_y, self.cam_specs.pix_num_y])
        self.fov = None

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
        self._maxrad_doas = tk.DoubleVar()
        self.maxrad_doas = self.spec_specs.fov * 1.1

    def generate_frame(self):
        """
        Generates frame
        :return:
        """
        self.frame = tk.Toplevel()
        self.frame.title('DOAS FOV alignment')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        self.in_frame = True

        # Create figure
        self.fig_img = plt.Figure(figsize=self.fig_size_doas_calib_img, dpi=self.dpi)
        self.ax_img = self.fig_img.subplots(1, 1)
        self.ax_img.set_aspect(1)

        # Create figure
        self.fig_fit = plt.Figure(figsize=self.fig_size_doas_calib_fit, dpi=self.dpi)
        self.ax_fit = self.fig_fit.subplots(1, 1)

        # Finalise canvas and gridding
        self.img_canvas = FigureCanvasTkAgg(self.fig_img, master=self.frame)
        self.img_canvas.draw()
        self.img_canvas.get_tk_widget().pack(side=tk.LEFT)

        self.fit_canvas = FigureCanvasTkAgg(self.fig_fit, master=self.frame)
        self.fit_canvas.draw()
        self.fit_canvas.get_tk_widget().pack(side=tk.LEFT)

        # If there is a calibration already available we can plot it
        if self.pyplis_worker.calib_pears is not None:
            self.update_plot()

    @property
    def maxrad_doas(self):
        """Access to tk variable _maxrad_doas. Defines maximum radius of doas FOV"""
        return self._maxrad_doas.get()

    @maxrad_doas.setter
    def maxrad_doas(self, value):
        self._maxrad_doas.set(value)

    def update_plot(self, img_corr=None, fov=None):
        """
        Updates plot
        :param img_corr:
        :param fov:
        :return:
        """
        if img_corr is None and fov is None:
            self.ax_img.cla()
            self.ax_fit.cla()
            self.pyplis_worker.calib_pears.fov.plot(ax=self.ax_img)
            self.pyplis_worker.calib_pears.plot(add_label_str="Pearson", color="b", ax=self.ax_fit)
        self.img_corr = img_corr
        self.fov = fov

        self.img_canvas.draw()
        self.fit_canvas.draw()

    def close_frame(self):
        """
        Closes frame
        :return:
        """
        self.in_frame = False
        self.frame.destroy()


class CellCalibFrame:
    """
    Frame to control some basic parameters in pyplis doas fov search and display the results if there are any
    """
    def __init__(self, generate=False, pyplis_work=pyplis_worker, cam_specs=CameraSpecs(), spec_specs=SpecSpecs(),
                 fig_setts=gui_setts):
        self.cam_specs = cam_specs
        self.spec_specs = spec_specs
        self.pyplis_worker = pyplis_work
        self.pyplis_worker.fig_cell_cal = self

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

    def generate_frame(self):
        """
        Generates frame
        :return:
        """
        self.frame = tk.Toplevel()
        self.frame.title('Cell calibration')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)

        self.in_frame = True

        # Create figure
        self.fig_abs = plt.Figure(figsize=self.fig_size_cell_abs, dpi=self.dpi)
        self.ax_abs = self.fig_abs.subplots(1, 1)
        self.ax_abs.set_aspect(1)

        # Create figure
        self.fig_mask = plt.Figure(figsize=self.fig_size_sens_mask, dpi=self.dpi)
        self.ax_mask = self.fig_mask.subplots(1, 1)
        self.ax_mask.set_aspect(1)

        # Create figure
        self.fig_fit = plt.Figure(figsize=self.fig_size_cell_fit, dpi=self.dpi)
        self.ax_fit = self.fig_fit.subplots(1, 1)

        # Finalise canvas and gridding
        self.fit_canvas = FigureCanvasTkAgg(self.fig_fit, master=self.frame)
        self.fit_canvas.draw()
        self.fit_canvas.get_tk_widget().pack(side=tk.TOP)

        self.abs_canvas = FigureCanvasTkAgg(self.fig_abs, master=self.frame)
        self.abs_canvas.draw()
        self.abs_canvas.get_tk_widget().pack(side=tk.LEFT)

        self.mask_canvas = FigureCanvasTkAgg(self.fig_mask, master=self.frame)
        self.mask_canvas.draw()
        self.mask_canvas.get_tk_widget().pack(side=tk.LEFT)

        # If there is a calibration already available we can plot it
        if self.pyplis_worker.got_cal_cell:
            self.update_plot()

    def update_plot(self):
        """
        Updates plot
        :return:
        """
        # Clear old axes
        self.ax_fit.cla()
        self.ax_abs.cla()
        self.ax_mask.cla()

        # Plot calibration fit line
        # TODO This line is not plotting consistently. Some issues with matching ppmm to AA in cell_cal_vals. Work out issue.
        self.ax_fit.scatter(self.pyplis_worker.cell_cal_vals[:, 1], self.pyplis_worker.cell_cal_vals[:, 0],
                            marker='x', color='white', s=50)
        max_tau = np.max(self.pyplis_worker.cell_cal_vals[:, 1])
        self.ax_fit.plot([0, max_tau],  self.pyplis_worker.cell_pol([0, max_tau]), '-', color='white')
        self.ax_fit.set_title('Cell ppm.m vs apparent absorbance')
        self.ax_fit.set_ylabel('Column Density [ppm.m]')
        self.ax_fit.set_xlabel('Apparent Absorbance')

        # Plot absorbance of 2nd smallest cell
        # TODO Make function which calculates 95% max value of image, so that we can set the scale of the color bar
        # TODO Currently this image seems to have a very large number, so the scale bar is all wrong - also check this
        abs_im = self.ax_abs.imshow(self.pyplis_worker.cell_tau_dict[self.pyplis_worker.sens_mask_ppmm],
                           interpolation='none', vmin=0)
        self.ax_abs.set_title('Cell absorbance: {} ppm.m'.format(self.pyplis_worker.sens_mask_ppmm))
        divider = make_axes_locatable(self.ax_abs)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = plt.colorbar(abs_im, cax=cax)
        cbar.outline.set_edgecolor('white')
        cbar.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')

        # Plot sensitivity mask
        mask_im = self.ax_mask.imshow(self.pyplis_worker.sensitivity_mask,
                           interpolation='none', vmin=1)
        self.ax_mask.set_title('Sensitivity mask')
        divider = make_axes_locatable(self.ax_mask)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = plt.colorbar(mask_im, cax=cax)
        cbar.outline.set_edgecolor('white')
        cbar.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')

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