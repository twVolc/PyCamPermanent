# -*- coding:utf-8 -*-

"""Contains a number of miscellaneous GUI classes for use"""
import pycam.gui.cfg as cfg
from pycam.setupclasses import pycam_details

import tkinter as tk
from tkinter import messagebox
from tkinter import font
import tkinter.ttk as ttk

from PIL import ImageTk, Image
import time
import threading


class About:
    """Class to create a tkinter frame containing the details of PyCam"""
    def __init__(self):
        info = ['PyCam v{}'.format(pycam_details['version']), 'Built on {}'.format(pycam_details['date'])]

        messagebox.showinfo('About Pycam', "\n".join(info))


class Indicator:
    """
    An indicator widget which turns off and on. Has the functionality to generate multiple instances of the
    indicator, so that it can be placed in numerous places in the GUI
    """
    def __init__(self, initiate=False):
        self.indicators = []
        self.frames = []
        self.labels = []
        self.buttons = []
        self.size = (40, 40)

        self.connected = False

        self.sock = cfg.sock

        if initiate:
            self.initiate_indicator()

    def initiate_indicator(self):
        """Initates class - only to be done after tk is running, otherwise an error is thrown"""
        self.font = font.Font(family='Helvetica', size=10, weight='bold')

        self.img_on = ImageTk.PhotoImage(Image.open('./icons/green-led.png').resize(self.size, Image.ANTIALIAS))
        self.img_off = ImageTk.PhotoImage(Image.open('./icons/red-led.png').resize(self.size, Image.ANTIALIAS))

    def generate_indicator(self, frame):
        """Generates a new widget"""
        self.frames.append(ttk.Frame(frame))
        self.indicators.append(tk.Canvas(self.frames[-1], width=self.size[0], height=self.size[1], bd=0, highlightthickness=0))
        self.indicators[-1].create_image(0, 0, image=self.img_off, anchor='nw', tags='IMG')
        self.indicators[-1].grid(row=0, column=0, rowspan=2, sticky='nsew')
        self.labels.append(ttk.Label(self.frames[-1], text='No Instrument Connected', font=self.font))
        self.labels[-1].grid(row=0, column=1, padx=5)
        self.buttons.append(ttk.Button(self.frames[-1], text='Connect', command=self.connect_sock))
        self.buttons[-1].grid(row=1, column=1, padx=5, sticky='ew')

    def indicator_on(self):
        """Changes all widgets to on state"""
        # Bool flag for connection
        self.connected = True

        x = 0
        for widget in self.indicators:
            widget.delete('IMG')
            widget.create_image(0, 0, image=self.img_on, anchor='nw', tags='IMG')
            self.labels[x].configure(text='Instrument Connected')
            self.buttons[x].configure(text='Disconnect', command=self.disconnect_sock)
            x += 1

    def indicator_off(self):
        """Changes all widgets to off state"""
        # Bool flag for connection
        self.connected = False

        x = 0
        for widget in self.indicators:
            widget.delete('IMG')
            widget.create_image(0, 0, image=self.img_off, anchor='nw', tags='IMG')
            self.labels[x].configure(text='No Instrument Connected')
            self.buttons[x].configure(text='Connect', command=self.connect_sock)
            x += 1

    # def connecting(self):
    #     """Changes label to connecting status"""
    #     for widget in self.labels:
    #         widget.configure(text='Connecting...')

    def connect_sock(self):
        """Attempts to connect to socket - threads connection attempt and will timeout after given time"""
        if self.connected:
            MessageWindow('Connection Error', ['Instrument already connected'])
            return

        try:
            self.sock.connect_socket_timeout(timeout=5)
        except ConnectionError:
            self.indicator_off()
            return

        # If the connection was made successfully we turn the indicator on
        self.indicator_on()

        # Setup function to receive from the socket
        cfg.recv_comms.thread_func()

        # Setup function to send comms to socket
        cfg.send_comms.thread_func()

    def disconnect_sock(self):
        """Closes socket if we are connected"""
        if not self.connected:
            MessageWindow('Connection Error', ['No connection was found to disconnect from'])

        self.sock.close_socket()

        # Set indicator to off
        self.indicator_off()


class NoInstrumentConnected:
    """Class to create a tkinter frame containing the details of PyCam"""
    def __init__(self):
        self.frame = tk.Toplevel()
        self.frame.title('Connect to instrument')

        label = ttk.Label(self.frame, text='No instrument connected.')
        label.pack(padx=5, pady=5)
        label = ttk.Label(self.frame, text='Please first connect to instrument before attempting shutdown')
        label.pack(padx=5, pady=5)

        button = ttk.Button(self.frame, text='Ok', command=self.close)
        button.pack(padx=5, pady=5)

    def close(self):
        """Closes widget"""
        self.frame.destroy()


class MessageWindow:
    """Class to create a tkinter frame containing any message

    Parameters
    ----------
    title: str
        Title to be placed at the top of the frame
    message: list
        List of messages to be placed as labels within the frame
    """

    def __init__(self, title, message):
        self.frame = tk.Toplevel()
        self.frame.title(title)

        for mess in message:
            label = ttk.Label(self.frame, text=mess)
            label.pack(padx=5, pady=5)

        button = ttk.Button(self.frame, text='Ok', command=self.close)
        button.pack(padx=5, pady=5)

    def close(self):
        """Closes widget"""
        self.frame.destroy()


class ScrollWindow:
    """Class to a frame within a window which can be scrolled if it doesn't fit on the screen
    A canvas for the window must already be created as well as a frame - the frame is the master of the canvas"""
    def __init__(self, frame, canvas, vertical=True, horizontal=True):

        self.canvas = canvas
        self.frame = tk.Frame(self.canvas)

        # If both paremeters are set to false we raise warning and return
        if not vertical and not horizontal:
            print('Scroll bar not used as both vertical and horizontal variables are set to False')
            return

        # Set up y direction scroll bar
        if vertical:
            self.scrollbar_y = tk.Scrollbar(frame, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=self.scrollbar_y.set)
            self.scrollbar_y.pack(side='right', fill='y')

        # Set up x direction scroll bar
        if horizontal:
            self.scrollbar_x = tk.Scrollbar(frame, orient='horizontal', command=canvas.xview)
            canvas.configure(xscrollcommand=self.scrollbar_x.set)
            self.scrollbar_x.pack(side='bottom', fill='x')

        # Finalise setup by packing canvas
        self.canvas.pack(fill="both", expand=True, anchor='nw')
        self.canvas.create_window((1,5), window=self.frame, anchor='nw', tags=self.frame)

        self.frame.bind('<Configure>', self.__on_frame_configure__)

    def __on_frame_configure__(self, event):
        """Controls movement of window on click event"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))