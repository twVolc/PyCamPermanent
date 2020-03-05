# -*- coding: utf-8 -*-

"""Module containing the core classes to create each window frame of the GUI
Each window forms a tab which can be accessed through the 'View' menu."""

import tkinter.ttk as ttk


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
        self.frame = ttk.Frame(self.parent)


class SpecWind:
    """Window for viewing spectrometer measurements and assciated DOAS retrievals"""
    def __init__(self, parent, name='Spectrometer'):
        self.parent = parent
        self.name = name
        self.frame = ttk.Frame(self.parent)


class AnalysisWind:
    """Window for viewing analysis of data - absorbance images and emission rates etc"""
    def __init__(self, parent, name='Analysis'):
        self.parent = parent
        self.name = name
        self.frame = ttk.Frame(self.parent)