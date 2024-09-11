# -*- coding: utf-8 -*-

"""Module containing the core classes to create each window frame of the GUI
Each window forms a tab which can be accessed through the 'View' menu."""

from .acquisition import CameraSettingsWidget, SpectrometerSettingsWidget
from .misc import ScrollWindow, MessageWindow
import pycam.gui.cfg as cfg
from pycam.doas.cfg import doas_worker
from pycam.gui.figures_cam import ImageFigure, ImageRegistrationFrame
from pycam.gui.figures_doas import SpectraPlot, DOASPlot, CDSeries, DirIndicator
from pycam.gui.figures_analysis import ImageSO2, SequenceInfo, TimeSeriesFigure

import tkinter as tk
import tkinter.ttk as ttk
import threading


class CameraWind:
    """Window for viewing camera images and associated data

    Parameters
    ----------
    parent: ttk.Frame
        Frame to place CameraWind frame within
    name: str
        Name of widget (mainly for ttk.Notebook use)
    """
    def __init__(self, main_gui, parent, name='Camera'):
        self.main_gui = main_gui
        self.parent = parent
        self.name = name
        self.padx = 5
        self.pady = 5

        self.frame = ttk.Frame(self.parent)
        # self.frame.columnconfigure(0, weight=1)
        # self.frame.rowconfigure(0, weight=1)

        # ========================================================
        # Scroll window
        self.plt_canvas = tk.Canvas(self.frame, borderwidth=0)
        self.plt_canvas_scroll = ScrollWindow(self.frame, self.plt_canvas)
        self.scroll_frame = ttk.Frame(self.plt_canvas_scroll.frame, borderwidth=2)
        self.scroll_frame.pack(expand=True, fill=tk.BOTH, anchor='nw')
        # ==========================================================

        # Get indicator and place it in the top of the frame
        cfg.indicator.generate_indicator(self.scroll_frame)
        cfg.indicator.frames[-1].grid(row=0, column=0, sticky='nw', padx=self.padx, pady=self.pady)

        # Get acquisition settings frame and add it to window
        self.acq_settings = CameraSettingsWidget(self.main_gui, self.scroll_frame)
        self.acq_settings.frame.grid(row=1, column=0, sticky='nw', padx=self.padx, pady=self.pady)

        # Image registration
        self.img_reg_frame = ImageRegistrationFrame(self.main_gui, self.scroll_frame)
        self.img_reg_frame.frame.grid(row=0, column=3, rowspan=2, sticky='new', padx=self.padx, pady=self.pady)

        # Image A widget setup
        draw_lock = threading.Lock()
        self.img_A = ImageFigure(self.main_gui, self.scroll_frame, self.img_reg_frame, lock=draw_lock,
                                 name='Image A', band='A')
        self.img_A.frame.grid(row=0, column=1, rowspan=2, sticky='nw', padx=self.padx, pady=self.pady)

        # Image B widget setup
        self.img_B = ImageFigure(self.main_gui, self.scroll_frame, self.img_reg_frame, lock=draw_lock,
                                 name='Image B', band='B')
        self.img_B.frame.grid(row=0, column=2, rowspan=2, sticky='nw', padx=self.padx, pady=self.pady)

        # Message window
        self.mess_wind = MessageWindow(self.main_gui, self.scroll_frame)
        self.mess_wind.frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=2, pady=2)


class SpecWind:
    """Window for viewing spectrometer measurements and assciated DOAS retrievals"""
    def __init__(self, gui, root, parent, name='Spectrometer'):
        self.gui = gui
        self.root = root
        self.parent = parent
        self.name = name
        self.padx = 5
        self.pady = 5
        self.frame = ttk.Frame(self.parent)

        # Get indicator and place it in the top of the frame
        cfg.indicator.generate_indicator(self.frame)
        cfg.indicator.frames[-1].grid(row=0, column=0, sticky='nw', padx=self.padx, pady=self.pady)

        # Get acquisition settings frame and add it to window
        self.acq_settings = SpectrometerSettingsWidget(self.gui, self.frame)
        self.acq_settings.frame.grid(row=1, column=0, sticky='nw', padx=self.padx, pady=self.pady)

        self.dir_indicator = DirIndicator(self.gui, self.frame, doas_worker)
        # ---------------------------------------
        # Plots
        # ---------------------------------------
        self.scroll_frame = ttk.Frame(self.frame)
        self.scroll_frame.grid(row=1, column=1, rowspan=3, sticky='nsew')
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(1, weight=1)

        # Canvas for scrolling through plots
        self.plt_canvas = tk.Canvas(self.scroll_frame, borderwidth=0)
        self.plt_canvas_scroll = ScrollWindow(self.scroll_frame, self.plt_canvas)
        self.plt_frame = ttk.Frame(self.plt_canvas_scroll.frame, borderwidth=2)
        self.plt_frame.pack(expand=True, fill=tk.BOTH, anchor='nw')

        # DOAS frame and spectrum frame
        self.doas_frame = DOASPlot(self.gui, self.root, self.plt_frame, species=doas_worker.ref_spec_used)
        self.spec_frame = SpectraPlot(self.gui, self.root, self.plt_frame, self.doas_frame)
        self.spec_frame.frame.pack(side='top', expand=1, anchor='n', fill=tk.X)
        self.doas_frame.frame.pack(side='top', expand=1, anchor='n', fill=tk.X)

        # CD time series frame
        self.series = CDSeries(self.plt_frame, doas_work=doas_worker)
        self.series.frame.pack(side='top', expand=1, anchor='n', fill=tk.X)


class AnalysisWind:
    """Window for viewing analysis of data - absorbance images and emission rates etc"""
    def __init__(self, main_gui, parent, name='Analysis'):
        self.main_gui = main_gui
        self.parent = parent
        self.name = name
        self.frame = ttk.Frame(self.parent)

        # ========================================================
        # Scroll window
        self.plt_canvas = tk.Canvas(self.frame, borderwidth=0)
        self.plt_canvas_scroll = ScrollWindow(self.frame, self.plt_canvas)
        self.scroll_frame = ttk.Frame(self.plt_canvas_scroll.frame, borderwidth=2)
        self.scroll_frame.pack(expand=True, fill=tk.BOTH, anchor='nw')
        # ==========================================================

        col_1 = ttk.Frame(self.scroll_frame)
        col_1.pack(side=tk.LEFT, anchor='nw')
        col_2 = ttk.Frame(self.scroll_frame)
        col_2.pack(side=tk.LEFT, anchor='nw', expand=True, fill=tk.Y)

        # Sequence info
        self.seq_info = SequenceInfo(main_gui, col_1)
        self.seq_info.frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # SO2 image frame
        self.so2_img = ImageSO2(main_gui, col_1)
        self.so2_img.frame.grid(row=1, column=0, padx=5, pady=5)

        # SO2 time series
        self.time_series = TimeSeriesFigure(main_gui, col_2)
        self.time_series.frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
