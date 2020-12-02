# -*- coding: utf-8 -*-

"""Main GUI script to be run as main executable"""
# Enables Basemap import by pointing to
import os
os.environ["PROJ_LIB"] = 'C:\\Users\\tw9616\Anaconda3\\envs\\py38\\Lib\\site-packages\\pyproj'

import sys
sys.path.append("C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\")

from pycam.gui.menu import PyMenu
from pycam.gui.windows import CameraWind, SpecWind, AnalysisWind
from pycam.networking.sockets import SocketClient
from pycam.setupclasses import ConfigInfo, FileLocator
from pycam.utils import read_file
from pycam.gui.cfg_menu_frames import geom_settings, process_settings, plume_bg, doas_fov, opti_flow, \
    light_dilution, cross_correlation
import pycam.gui.cfg as cfg
from pycam.cfg import pyplis_worker
from pycam.doas.cfg import doas_worker

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from ttkthemes import ThemedStyle

import sys
import warnings
# warnings.simplefilter("ignore", UserWarning)    # Ignore UserWarnings, in particular tight_layout which is annoying


class PyCam(ttk.Frame):
    def __init__(self, root, x_size, y_size):
        ttk.Frame.__init__(self, root)
        self.root = root
        self.root.title('PyCam')
        self.root.protocol('WM_DELETE_WINDOW', self.exit_app)

        # Initiate indicator widget
        cfg.indicator.initiate_indicator()

        # Load in configuration file(s)
        self.config = read_file(FileLocator.CONFIG_WINDOWS)

        # Setup socket
        self.sock = SocketClient(host_ip=self.config[ConfigInfo.host_ip], port=int(self.config[ConfigInfo.port_ext]))

        # Setup style
        self.style = ThemedStyle(self.root)
        # self.style.set_theme('equilux')
        self.style.set_theme('breeze')
        self.layout_old = self.style.layout('TNotebook.Tab')
        self.style.layout('TNotebook.Tab', [])          # Turns off notepad bar

        # Menu bar setup
        self.menu = PyMenu(self, self.root)
        self.root.config(menu=self.menu.frame)

        # -----------------------------------------------
        # Windows setup
        self.windows = ttk.Notebook(self.root)
        self.windows.pack(fill='both', expand=1)

        # Create object of each window
        self.cam_wind = CameraWind(self.windows)
        self.spec_wind = SpecWind(self.root, self.windows)
        self.anal_wind = AnalysisWind(self.windows)

        # Add each window to Notebook
        self.windows.add(self.cam_wind.frame, text=self.cam_wind.name)
        self.windows.add(self.spec_wind.frame, text=self.spec_wind.name)
        self.windows.add(self.anal_wind.frame, text=self.anal_wind.name)
        # -----------------------------------------------

        # LOAD ALL DEFAULT INFO FROM OBJECTS WHICH REQUIRE THIS TO BE DONE AFTER INTIAL TK BUILD
        self.info_load()

    def info_load(self):
        """Instantiates all frames which require some kind of start-up instantiation"""
        geom_settings.initiate_variables()
        process_settings.initiate_variables()
        plume_bg.initiate_variables()
        doas_fov.start_draw(self.root)      # start drawing of frame
        doas_fov.initiate_variables()
        cross_correlation.start_draw(self.root)
        cross_correlation.initiate_variables()
        opti_flow.initiate_variables()
        light_dilution.initiate_variables()
        self.menu.load_frame.img_reg_frame = self.cam_wind.img_reg_frame
        self.menu.load_frame.load_all()

        # Load in initial sequence directory
        pyplis_worker.doas_worker = doas_worker     # Set DOAS worker to pyplis attribute
        pyplis_worker.load_sequence(pyplis_worker.img_dir, plot_bg=False)
        doas_worker.load_dir(prompt=False, plot=True)

    def exit_app(self):
        """Closes application"""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):

            # Close main window and stop program
            self.root.destroy()
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


