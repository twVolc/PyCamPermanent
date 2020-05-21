# -*- coding: utf-8 -*-

"""Holds classes for building a menu bar in tkinter"""

from pycam.setupclasses import pycam_details
from pycam.gui.network import ConnectionGUI
from pycam.networking.sockets import SocketClient

import tkinter as tk
import tkinter.ttk as ttk


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
    def __init__(self, parent, frame, sock=SocketClient(host_ip=None, port=None)):
        self.parent = parent
        self.frame = tk.Menu(frame)
        self.sock = sock

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

        # View tab - can be used for toggling between views (e.g., camera frame, DOAS frame, processing frame)
        tab = 'View'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_separator()
        self.menus[tab].add_command(label='Camera window')
        self.menus[tab].add_command(label='DOAS window')
        self.menus[tab].add_command(label='Analysis window')
        self.menus[tab].add_separator()

        # Help tab
        tab = 'Help'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_command(label='About', command=About)

        # Loop through keys to setup cascading menus
        for key in keys:
            self.frame.add_cascade(label=key, menu=self.menus[key])


class About:
    """Class to create a tkinter frame containing the details of PyCam"""
    def __init__(self):
        self.frame = tk.Toplevel()
        self.frame.title('About PyCam')

        label = ttk.Label(self.frame, text='PyCam v{}'.format(pycam_details['version'])).pack(anchor='w')
        label = ttk.Label(self.frame, text='Built on {}'.format(pycam_details['date'])).pack(anchor='w')


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

        # Add the frames for each tab to the notebook
        self.windows.add(self.connection_gui.frame, text=self.connection_gui.name)


