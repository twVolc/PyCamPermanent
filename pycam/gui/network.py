# -*- coding: utf-8 -*-

"""Code to build interface between the GUI and the instrument"""

import pycam.gui.cfg as cfg
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd
from pycam.setupclasses import FileLocator

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import subprocess
import platform
import time


def run_pycam(ip):
    """Runs main pycam script on remote machine"""
    if messagebox.askyesno("Please confirm", "Are you sure you want to run pycam_masterpi.py?\n"
                                             "Running this on a machine which already has the script running could cause issues"):
        print('Running pycam_masterpi on {}'.format(ip))

        # Path to executable
        pycam_path = FileLocator.SCRIPTS + 'pycam_masterpi.py'

        # Open ssh connection
        connection = open_ssh(ip)

        # Run ssh command
        stderr, stdout = ssh_cmd(connection, 'python3 {}'.format(pycam_path), background=False)

        # Close ssh connection
        close_ssh(connection)


def instrument_cmd(cmd):
    """Checks if you wanted to shutdown the camera and then sends EXT command to instrument"""
    timeout = 5  # Timeout for instrument shutting down on 'EXT' request

    # Generate message depending on command
    if cmd == 'EXT':
        mess = "Are you sure you want to shutdown the instrument?"
    elif cmd == 'RST':
        mess = "Are you sure you want to restart the instrument?"
    elif cmd == 'RSC':
        mess = "Are you sure you want to restart the cameras?"
    elif cmd == 'RSS':
        mess = "Are you sure you want to restart the spectrometer?"

    # Check if we have a connection to the instrument
    if cfg.indicator.connected:

        if messagebox.askyesno("Please confirm", mess):

            # Add command to queue to shutdown instrument
            cfg.send_comms.q.put({cmd: 1})

            # If EXT: Wait for system to shutdown then indicate that we no longer are connected to it
            if cmd == 'EXT':
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if not cfg.recv_comms.working:
                        cfg.indicator.indicator_off()
                        break
                else:
                    messagebox.showerror('Command Error', 'Instrument appears to still be running')

    # Raise instrument connection error
    else:
        messagebox.showerror('Connection error', 'No instrument connected')
        # NoInstrumentConnected()


class ConnectionGUI:
    """Frame containing code to generate a GUI allowing definition of connection parameters to the instrument"""
    def __init__(self, parent, name='Connection'):
        self.parent = parent
        self.name = name

        self.frame = ttk.Frame(self.parent)
        self.pdx = 5
        self.pdy = 5

        self._host_ip = tk.StringVar()
        self._port = tk.IntVar()
        self.host_ip = cfg.sock.host_ip
        self.port = cfg.sock.port

        ttk.Label(self.frame, text='IP address:').grid(row=0, column=0, padx=self.pdx, pady=self.pdy, sticky='e')
        ttk.Label(self.frame, text='Port:').grid(row=1, column=0, padx=self.pdx, pady=self.pdy, sticky='e')
        ttk.Entry(self.frame, width=15, textvariable=self._host_ip).grid(row=0, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')
        ttk.Entry(self.frame, width=6, textvariable=self._port).grid(row=1, column=1, padx=self.pdx, pady=self.pdy, sticky='ew')

        self.test_butt = ttk.Button(self.frame, text='Test Connection', command=self.test_connection)
        self.test_butt.grid(row=0, column=2, padx=self.pdx, pady=self.pdy)

        self.connection_label = ttk.Label(self.frame, text='')
        self.connection_label.grid(row=0, column=3, padx=self.pdx, pady=self.pdy)

        self.update_butt = ttk.Button(self.frame, text='Update connection', command=self.update_connection)
        self.update_butt.grid(row=2, column=1, padx=self.pdx, pady=self.pdy)

    @property
    def host_ip(self):
        """Public access to tk variable _host_ip"""
        return self._host_ip.get()

    @host_ip.setter
    def host_ip(self, value):
        """Public access setter of tk variable _host_ip"""
        self._host_ip.set(value)

    @property
    def port(self):
        """Public access to tk variable _port"""
        return self._port.get()

    @port.setter
    def port(self, value):
        """Public access setter of tk variable _port"""
        self._port.set(value)

    def update_connection(self):
        """Updates socket address information"""
        cfg.sock.update_address(self.host_ip, self.port)

    def test_connection(self):
        """Tests that IP address is available"""
        # Attempt ping
        try:
            output = subprocess.check_output(
                "ping -{} 1 {}".format('n' if platform.system().lower() == "windows" else 'c', self.host_ip),
                shell=True)

            # If output says host unreachable we flag that there is no connection
            if b'Destination host unreachable' in output:
                raise Exception

            # Otherwise there must be a connection, so we update our label to say this
            else:
                self.connection_label.configure(text='Connection found')

                # Update the connection if we have a good connection
                self.update_connection()

        except Exception as e:
            print(e)
            self.connection_label.configure(text='No connection found at this address')





