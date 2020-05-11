# -*- coding: utf-8 -*-

"""
Module containing widgets for camera and spectrometer control, by connecting to the instrument via a socket and sneding
messages other comms
"""

import tkinter as tk
import tkinter.ttk as ttk

class CameraSettingsWidget:
    """
    Hold all widgets required for communicating with the camera

    Parameters
    ----------
    parent: tk.Frame, ttk.Frame
        Parent frame that widget will be placed into
    """
    def __init__(self, parent):
        self.parent = parent


class SpectrometerSettingsWidget:
    """
    Hold all widgets required for spectrometer settings

    Parameters
    ----------
    parent: tk.Frame, ttk.Frame
        Parent frame that widget will be placed into
    """
    def __init__(self, parent, sock=None):
        pass


