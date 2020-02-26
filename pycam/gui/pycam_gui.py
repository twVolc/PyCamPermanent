# -*- coding: utf-8 -*-

"""Main GUI script to be run as main executable"""

from pycam.gui.menu import PyMenu

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import sys


class PyCam(ttk.Frame):
    def __init__(self, parent, x_size, y_size):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        parent.title('PySpec')
        self.parent.protocol('WM_DELETE_WINDOW', self.exit_app)

        self.menu = PyMenu(self, self.parent)
        self.parent.config(menu=self.menu.frame)

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
    root.geometry('{0}x{1}+0+0'.format(root.winfo_screenwidth() - padx, root.winfo_screenheight() - pady))
    x_size = root.winfo_screenwidth()  # Get screen width
    y_size = root.winfo_screenheight()  # Get screen height
    myGUI = PyCam(root, x_size, y_size)
    root.mainloop()


if __name__ == '__main__':
    run_GUI()


