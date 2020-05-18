# -*- coding: utf-8 -*-

"""Module containing the core classes to create each window frame of the GUI
Each window forms a tab which can be accessed through the 'View' menu."""

import tkinter as tk
import tkinter.ttk as ttk
from .acquisition import CameraSettingsWidget, SpectrometerSettingsWidget


class CameraWind:
    """Window for viewing camera images and associated data

    Parameters
    ----------
    parent: ttk.Frame
        Frame to place CameraWind frame within
    name: str
        Name of widget (mainly for ttk.Notebook use)
    """
    def __init__(self, parent, name='Camera'):
        self.parent = parent
        self.name = name
        self.padx = 5
        self.pady = 5

        self.frame = ttk.Frame(self.parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.acq_settings = CameraSettingsWidget(self.frame)
        self.acq_settings.frame.grid(row=0, column=0, sticky='nw', padx=self.padx, pady=self.pady)


class SpecWind:
    """Window for viewing spectrometer measurements and assciated DOAS retrievals"""
    def __init__(self, parent, name='Spectrometer'):
        self.parent = parent
        self.name = name
        self.padx = 5
        self.pady = 5
        self.frame = ttk.Frame(self.parent)

        self.acq_settings = SpectrometerSettingsWidget(self.frame)
        self.acq_settings.frame.grid(row=0, column=0, sticky='nw', padx=self.padx, pady=self.pady)


class AnalysisWind:
    """Window for viewing analysis of data - absorbance images and emission rates etc"""
    def __init__(self, parent, name='Analysis'):
        self.parent = parent
        self.name = name
        self.frame = ttk.Frame(self.parent)