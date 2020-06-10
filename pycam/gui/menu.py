# -*- coding: utf-8 -*-

"""Holds classes for building a menu bar in tkinter"""

from pycam.setupclasses import pycam_details
from pycam.gui.network import ConnectionGUI, instrument_cmd, run_pycam
import pycam.gui.cfg as cfg
from pycam.gui.misc import About
import pycam.gui.settings as settings
from pycam.gui.figures_doas import CalibrationWindow

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import time


class PyMenu:
    """tk menu bar for placing along the top o the GUI

    Parameters
    ----------
    parent: PyCam object
        Object must have an exit_app method which is called
    frame: ttk.Frame
        tkinter Frame object which PyMenu will take as its parent
    sock: SocketClient
        Socket used for connection to external instrument"""
    def __init__(self, parent, frame):
        self.parent = parent
        self.frame = tk.Menu(frame)

        # Setup dictionary to hold menu tabs and list to hold the menu tab names. Each menu tab is held in the
        # dictionary with the assocaited key stored in keys. This makes cascade setup easy by looping through keys
        self.menus = dict()
        keys = list()

        # File tab
        tab = 'File'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_command(label='Settings', command=Settings)
        self.menus[tab].add_separator()
        self.menus[tab].add_command(label='Exit', command=self.parent.exit_app)

        # ---------------------------------------------------------------------------------------------
        # Instrument tab for interfacing with instrument
        tab = 'Instrument'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        self.menus[tab].add_command(label='Connect', command=cfg.indicator.connect_sock)
        self.menus[tab].add_command(label='Disconnect', command=cfg.indicator.disconnect_sock)

        self.submenu_cmd = tk.Menu(self.frame, tearoff=0)
        self.submenu_cmd.add_command(label='Shutdown', command=lambda: instrument_cmd('EXT'))
        self.submenu_cmd.add_separator()
        self.submenu_cmd.add_command(label='Restart', command=lambda: instrument_cmd('RST'))
        self.submenu_cmd.add_command(label='Restart cameras', command=lambda: instrument_cmd('RSC'))
        self.submenu_cmd.add_command(label='Restart spectrometer', command=lambda: instrument_cmd('RSS'))
        self.submenu_cmd.add_separator()
        self.submenu_cmd.add_command(label='Run pycam', command=lambda: run_pycam(cfg.sock.host_ip))

        self.menus[tab].add_cascade(label='Commands', menu=self.submenu_cmd)
        # -------------------------------------------------------------------------------------------------

        # View tab - can be used for toggling between views (e.g., camera frame, DOAS frame, processing frame)
        tab = 'View'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        # More windows cascade
        self.submenu_windows = tk.Menu(self.frame, tearoff=0)
        self.submenu_windows.add_command(label="DOAS calibration", command=CalibrationWindow)
        self.menus[tab].add_cascade(label="More windows", menu=self.submenu_windows)

        self.menus[tab].add_separator()
        self.menus[tab].add_command(label='Camera window')
        self.menus[tab].add_command(label='DOAS window')
        self.menus[tab].add_command(label='Analysis window')
        self.menus[tab].add_separator()



        # -------------------------------------------------------------------------------------------------------
        # Help tab
        tab = 'Help'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_command(label='About', command=About)

        # Loop through keys to setup cascading menus
        for key in keys:
            self.frame.add_cascade(label=key, menu=self.menus[key])


class Settings:
    """Class to control the settings from the GUI toolbar"""
    def __init__(self):
        self.frame = tk.Toplevel()
        self.frame.title('PyCam Settings')
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth()/2),
                                                 int(self.frame.winfo_screenheight()/2),
                                                 int(self.frame.winfo_screenwidth()/4),
                                                 int(self.frame.winfo_screenheight()/4)))

        # Setup notebook tabs
        self.windows = ttk.Notebook(self.frame)
        self.windows.pack(fill='both', expand=1, padx=5, pady=5)

        # Generate the frames for each tab
        self.connection_gui = ConnectionGUI(self.windows)
        self.gui_settings = settings.SettingsFrame(self.windows, settings=cfg.gui_setts)

        # Add the frames for each tab to the notebook
        self.windows.add(self.connection_gui.frame, text=self.connection_gui.name)
        self.windows.add(self.gui_settings.frame, text=self.gui_settings.name)



