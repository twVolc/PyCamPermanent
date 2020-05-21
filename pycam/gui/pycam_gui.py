# -*- coding: utf-8 -*-

"""Main GUI script to be run as main executable"""

from pycam.gui.menu import PyMenu
from pycam.gui.windows import CameraWind, SpecWind, AnalysisWind
from pycam.networking.sockets import SocketClient
from pycam.setupclasses import ConfigInfo, FileLocator
from pycam.utils import read_file
import pycam.gui.network_cfg as cfg

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from ttkthemes import ThemedStyle

import sys


class PyCam(ttk.Frame):
    def __init__(self, parent, x_size, y_size):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        parent.title('PyCam')
        self.parent.protocol('WM_DELETE_WINDOW', self.exit_app)

        # Initiate indicator widget
        cfg.indicator.initiate_indicator()

        # Load in configuration file(s)
        self.config = read_file(FileLocator.CONFIG_WINDOWS)

        # Setup socket
        self.sock = SocketClient(host_ip=self.config[ConfigInfo.host_ip], port=int(self.config[ConfigInfo.port_ext]))


        # Setup style
        self.style = ThemedStyle(self.parent)
        # self.style.set_theme('equilux')
        self.style.set_theme('breeze')

        # Menu bar setup
        self.menu = PyMenu(self, self.parent, sock=self.sock)
        self.parent.config(menu=self.menu.frame)

        # -----------------------------------------------
        # Windows setup
        self.windows = ttk.Notebook(self.parent)
        self.windows.pack(fill='both', expand=1)

        # Create object of each window
        self.cam_wind = CameraWind(self.windows)
        self.spec_wind = SpecWind(self.windows)
        self.anal_wind = AnalysisWind(self.windows)

        # Add each window to Notebook
        self.windows.add(self.cam_wind.frame, text=self.cam_wind.name)
        self.windows.add(self.spec_wind.frame, text=self.spec_wind.name)
        self.windows.add(self.anal_wind.frame, text=self.anal_wind.name)
        # -----------------------------------------------



    def exit_app(self):
        """Closes application"""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):

            # Close main window and stop program
            self.parent.destroy()
            sys.exit()


def run_GUI():
    padx = 0
    pady = 0
    root = tk.Tk()
    root.geometry('{}x{}+0+0'.format(root.winfo_screenwidth() - padx, root.winfo_screenheight() - pady))
    x_size = root.winfo_screenwidth()  # Get screen width
    y_size = root.winfo_screenheight()  # Get screen height
    myGUI = PyCam(root, x_size, y_size)
    root.mainloop()


if __name__ == '__main__':
    run_GUI()


