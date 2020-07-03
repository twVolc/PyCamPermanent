# -*- codingL utf-8 -*-

"""Holds all classes for generating matplotlib figures for the pycam GUI"""

from pycam.setupclasses import CameraSpecs
import pycam.gui.cfg as cfg
from pycam.cfg import pyplis_worker

import tkinter as tk
import tkinter.ttk as ttk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.spines
import matplotlib.cm as cm

import numpy as np
import threading
import time


class ImageFigure:
    """
    Class for plotting an image and associated widgets, such as cross-sectinal DNs
    """
    def __init__(self, frame, lock=threading.Lock(), name='Image', band='A',
                 image=np.zeros([CameraSpecs().pix_num_y, CameraSpecs().pix_num_x]),
                 start_update_thread=False):
        self.parent = frame
        self.frame = ttk.LabelFrame(self.parent, text=name)
        self.lock = lock
        self.update_thread = None
        self.draw_time = time.time()
        self.plot_lag = 0.15        # Minimum time between successive draw() calls. Prevents GUI freezing

        # Set self to pyplis worker figure object
        setattr(pyplis_worker, 'fig_{}'.format(band), self)

        self.image = image
        self.band = band
        self.specs = CameraSpecs()

        self.pdx = 2
        self.pdy = 2

        self.img_fig_size = cfg.gui_setts.fig_img
        self.dpi = cfg.gui_setts.dpi

        self._row_colour = 'limegreen'  # Colour of cross-section plotting for row
        self._col_colour = 'red'        # Colour of cross-section plotting for column
        self._init_img_row = 50         # Starting row position
        self._init_img_col = 50         # Starting column position
        self._row = tk.IntVar()
        self._col = tk.IntVar()

        # Build figure
        self._build_img_fig()

        # Build cross-section control panel
        self._build_xsect_panel()

        # Grid each frame
        self.xsect_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.fig_frame.grid(row=1, column=0, padx=5, pady=5)

        if start_update_thread:
            self.start_update_thread()

    @property
    def row(self):
        val = self._row.get()
        if val > self.specs.pix_num_y:
            self._row.set(self.specs.pix_num_y)
        return self._row.get()

    @row.setter
    def row(self, value):
        self._row.set(value)

    @property
    def col(self):
        val = self._col.get()
        if val > self.specs.pix_num_x:
            self._col.set(self.specs.pix_num_x)
        return self._col.get()

    @col.setter
    def col(self, value):
        self._col.set(value)

    def _build_img_fig(self):
        """Builds image figure using matplotlib"""
        self.fig_frame = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=3)

        # Generate figure and axes
        self.fig = plt.Figure(figsize=self.img_fig_size, dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, width_ratios=[648, 200], height_ratios=[486, 200])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.set_aspect(1)
        self.plt_col = self.fig.add_subplot(gs[1], sharey=self.ax)
        self.plt_row = self.fig.add_subplot(gs[2], sharex=self.ax)

        plt.setp(self.plt_col.get_yticklabels(), visible=False)
        plt.setp(self.ax.get_xticklabels(), visible=False)

        self.fig.set_facecolor('black')

        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        self.img_disp = self.ax.imshow(self.image, cmap=cm.gray, interpolation='none', vmin=0,
                                       vmax=self.specs._max_DN, aspect='auto')  # FOR GREYSCALE
        self.img_disp_row, = self.ax.plot([0, self.specs.pix_num_x], [self._init_img_row, self._init_img_row],
                                          color=self._row_colour, lw=2)
        self.img_disp_col, = self.ax.plot([self._init_img_col, self._init_img_col], [0, self.specs.pix_num_y],
                                          color=self._col_colour, lw=2)
        self.ax.set_xlim([0, self.specs.pix_num_x - 1])
        self.ax.set_ylim([self.specs.pix_num_y - 1, 0])
        self.ax.set_title('Test Image', color='white')
        self.ax.set_ylabel('Pixel', color='white')

        for child in self.plt_row.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.plt_row.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        for child in self.plt_col.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.plt_col.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        self.plt_row.set_facecolor('black')
        self.plt_col.set_facecolor('black')

        self.pix_row = np.arange(0, self.specs.pix_num_x, 1)
        self.pix_col = np.arange(0, self.specs.pix_num_y, 1)
        self.row_DN = self.image[self._init_img_row, :]
        self.col_DN = self.image[:, self._init_img_row]

        self.line_row, = self.plt_row.plot(self.pix_row, self.row_DN, color=self._row_colour)
        self.line_col, = self.plt_col.plot(self.col_DN, self.pix_col, color=self._col_colour)

        # ------------------------------------------------------
        # Plot settings
        # ------------------------------------------------------
        self.plt_row.set_xlabel('Pixel', color='white')
        self.plt_row.set_ylabel('DN', color='white')
        self.plt_col.set_xlabel('DN', color='white')
        # self.plt_col.set_ylabel('Pixel', color='white')
        # self.plt_row.set_xlim(0, self.imgSizeX)
        self.plt_row.set_ylim(0, self.specs._max_DN)
        self.plt_col.set_xlim(0, self.specs._max_DN)
        self.plt_col.set_ylim(self.specs.pix_num_y, 0)

        self.fig.tight_layout()  # Make plots extend right to edges of figure (or at least to a good fit)
        self.fig.subplots_adjust(hspace=0.1, wspace=486 / 6480)  # Make space between subplots equal

        # -------------------------------------
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        with self.lock:
            self.img_canvas.draw()
        self.img_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def _build_xsect_panel(self):
        """Builds control panel GUI for adjusting x-sections in image"""
        self.xsect_frame = ttk.LabelFrame(self.frame, text="Image cross-sectional DNs", relief=tk.GROOVE,
                                          borderwidth=2)

        # Row spinbox
        row_label = ttk.Label(self.xsect_frame, text="Row:")
        row_label.grid(row=0, column=0, padx=self.pdx, pady=self.pdy, sticky='w')
        self.row = self._init_img_row
        self.img_row = ttk.Spinbox(self.xsect_frame, from_=0, to=self.specs.pix_num_y-1, width=4,
                                   textvariable=self._row)
        self.img_row.grid(row=0, column=1, padx=self.pdx, pady=self.pdy, sticky='w')

        # Column spinbox
        row_col = ttk.Label(self.xsect_frame, text="Column:")
        row_col.grid(row=0, column=2, padx=self.pdx, pady=self.pdy, sticky='w')
        self.col = self._init_img_col
        self.img_col = ttk.Spinbox(self.xsect_frame, from_=0, to=self.specs.pix_num_x-1, width=4,
                                   textvariable=self._col)
        self.img_col.grid(row=0, column=3, padx=self.pdx, pady=self.pdy, sticky='w')

        self.x_sect_butt = ttk.Button(self.xsect_frame, text="Update plot", command=self.x_sect_plot)
        self.x_sect_butt.grid(row=0, column=4, padx=self.pdx, pady=self.pdy)

    def x_sect_plot(self):
        """Updates cross-section plot"""
        # Extract row and column digital numbers
        row_DN = self.image[self.row, :]
        col_DN = self.image[:, self.col]

        # Plot new values on subplots
        self.line_row.set_data(self.pix_row, row_DN)
        self.line_col.set_data(col_DN, self.pix_col)

        # Update main figure cross-section lines
        self.img_disp_row.set_data([0, self.specs.pix_num_x], [self.row, self.row])
        self.img_disp_col.set_data([self.col, self.col], [0, self.specs.pix_num_y])

        # Redraw the canvas to update plot
        with self.lock:
            # Check how long has passed. Only draw if > 0.5s has passed, to ensure that we don't freeze up the GUI
            if time.time() - self.draw_time > self.plot_lag:
                self.img_canvas.draw()
                self.draw_time = time.time()

    def update_plot(self, img, img_path):
        """
        Updates image figure and all associated subplots
        :param img: np.ndarray  Image array
        :param img_path: str    Image name to be set as title
        :return:
        """
        self.image = img
        filename = img_path.split('\\')[-1]     # Extract filename from full path

        # Update main image display and title
        self.img_disp.set_data(img)
        self.ax.set_title(filename)

        # Update subplots - this includes a call to draw() so the figure will be updated after this
        self.x_sect_plot()

    def start_update_thread(self):
        """
        Starts image update thread
        :return:
        """
        self.update_thread = threading.Thread(target=self._img_update_thread, args=())
        self.update_thread.daemon = True
        self.update_thread.start()

    def _img_update_thread(self):
        """
        Gets new images to be displayed in figure and displays them
        :return:
        """
        while True:
            # Get next image and its path (passed to queue as a 2-element list)
            img_path, img_obj = getattr(pyplis_worker, 'img_{}_q'.format(self.band)).get(block=True)

            print(img_path)

            # Get data from the pyplis.image.Img object
            self.update_plot(np.array(img_obj.img, dtype=np.uint16), img_path)
