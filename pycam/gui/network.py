# -*- coding: utf-8 -*-

"""Code to build interface between the GUI and the instrument"""

import pycam.gui.cfg as cfg
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd
from pycam.setupclasses import FileLocator
from pycam.io import write_witty_schedule_file, read_witty_schedule_file

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import subprocess
import platform
import time
import datetime


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


class InstrumentConfiguration:
    """
    Class creating a widget for configuring the instrument, e.g. adjusting its off/on time through Witty Pi
    """
    def __init__(self, ftp):
        self.ftp = ftp
        self.time_fmt = '{}:{}'
        self.frame = None
        self.in_frame = False

    def initiate_variable(self):
        """Initiate tkinter variables"""
        self._on_hour = tk.IntVar()  # Hour to turn on pi
        self._on_min = tk.IntVar()

        self._off_hour = tk.IntVar()        # Hour to shutdown pi
        self._off_min = tk.IntVar()

        self._capt_start_hour = tk.IntVar()     # Hour to start capture
        self._capt_start_min = tk.IntVar()

        self._capt_stop_hour = tk.IntVar()      # Hour to stop capture
        self._capt_stop_min = tk.IntVar()

        on_time, off_time = read_witty_schedule_file(FileLocator.SCHEDULE_FILE)
        self.on_hour, self.on_min = on_time
        self.off_hour, self.off_min = off_time

    def generate_frame(self):
        """Generates frame containing GUI widgets"""
        if self.in_frame:
            self.frame.attributes('-topmost', 1)
            self.frame.attributes('-topmost', 0)
            return

        self.frame = tk.Toplevel()
        self.frame.title('Instrument configuration')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)
        self.in_frame = True

        frame_on = tk.LabelFrame(self.frame, text='Start-up/Shut-down times', relief=tk.RAISED, borderwidth=2)
        frame_on.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        ttk.Label(frame_on, text='Start-up (hour:minutes):').grid(row=0, column=0, sticky='w', padx=2, pady=2)
        ttk.Label(frame_on, text='Shut-down (hour:minutes):').grid(row=1, column=0, sticky='w', padx=2, pady=2)

        hour_start = ttk.Spinbox(frame_on, textvariable=self._on_hour, from_=00, to=23, increment=1, width=2,
                                 format="%02.0f")
        hour_start.set("{:02d}".format(self.on_hour))
        hour_start.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(frame_on, text=':').grid(row=0, column=2, padx=2, pady=2)
        min_start = ttk.Spinbox(frame_on, textvariable=self._on_min, from_=00, to=59, increment=1, width=2,
                                format="%02.0f")
        min_start.set("{:02d}".format(self.on_min))
        min_start.grid(row=0, column=3, padx=2, pady=2)

        hour_stop = ttk.Spinbox(frame_on, textvariable=self._off_hour, from_=00, to=23, increment=1, width=2,
                                format="%02.0f")
        hour_stop.set("{:02d}".format(self.off_hour))
        hour_stop.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(frame_on, text=':').grid(row=1, column=2, padx=2, pady=2)
        min_stop = ttk.Spinbox(frame_on, textvariable=self._off_min, from_=00, to=59, increment=1, width=2,
                               format="%02.0f")
        min_stop.set("{:02d}".format(self.off_min))
        min_stop.grid(row=1, column=3, padx=2, pady=2)

        # Update button
        butt = ttk.Button(frame_on, text='Update', command=self.update_on_off)
        butt.grid(row=2, column=0, columnspan=4, sticky='e', padx=2, pady=2)

        # Start/stop control of acquisition times
        frame_acq = tk.LabelFrame(self.frame, text='Start/stop acquisition', relief=tk.RAISED, borderwidth=2)
        frame_acq.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)

        ttk.Label(frame_acq, text='Start (hour:minutes):').grid(row=0, column=0, sticky='w', padx=2, pady=2)
        ttk.Label(frame_acq, text='Stop (hour:minutes):').grid(row=1, column=0, sticky='w', padx=2, pady=2)

        hour_start = ttk.Spinbox(frame_acq, textvariable=self._capt_start_hour, from_=00, to=23, increment=1, width=2,
                                 format="%02.0f")
        hour_start.set("{:02d}".format(self.capt_start_hour))
        hour_start.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(frame_acq, text=':').grid(row=0, column=2, padx=2, pady=2)
        min_start = ttk.Spinbox(frame_acq, textvariable=self._capt_start_min, from_=00, to=59, increment=1, width=2,
                                format="%02.0f")
        min_start.set("{:02d}".format(self.capt_start_min))
        min_start.grid(row=0, column=3, padx=2, pady=2)

        hour_stop = ttk.Spinbox(frame_acq, textvariable=self._capt_stop_hour, from_=00, to=23, increment=1, width=2,
                                format="%02.0f")
        hour_stop.set("{:02d}".format(self.capt_stop_hour))
        hour_stop.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(frame_acq, text=':').grid(row=1, column=2, padx=2, pady=2)
        min_stop = ttk.Spinbox(frame_acq, textvariable=self._capt_stop_min, from_=00, to=59, increment=1, width=2,
                               format="%02.0f")
        min_stop.set("{:02d}".format(self.capt_stop_min))
        min_stop.grid(row=1, column=3, padx=2, pady=2)

        # Update button
        butt = ttk.Button(frame_acq, text='Update', command=self.update_acq_time)
        butt.grid(row=2, column=0, columnspan=4, sticky='e', padx=2, pady=2)

        # TODO Have option for defining time for dark acquisitions

    def update_on_off(self):
        """Controls updating start/stop time of pi"""
        # Write wittypi schedule file locally
        write_witty_schedule_file(FileLocator.SCHEDULE_FILE, self.on_time, self.off_time)

        # Transfer file to instrument
        self.ftp.move_file_to_instrument(FileLocator.SCHEDULE_FILE, FileLocator.SCHEDULE_FILE_PI)

        # Open ssh and run wittypi update script
        ssh_cli = open_ssh(self.ftp.host_ip)

        std_in, std_out, std_err = ssh_cmd(ssh_cli, '(cd /home/pi/wittypi; sudo ./runScript.sh)', background=False)
        # std_in, std_out, std_err = ssh_cmd(ssh_cli, 'sudo python3 /home/pi/pycam/scripts/update_witty.py')
        print(std_out.readlines())
        print(std_err.readlines())

        a = tk.messagebox.showinfo('Instrument update',
                                   'Updated instrument start-up/shut-down schedule\n'
                                   'Start-up: {} UTC\n''Shut-down: {} UTC'.format(self.on_time.strftime('%H:%M'),
                                                                                  self.off_time.strftime('%H:%M')))

        self.frame.attributes('-topmost', 1)
        self.frame.attributes('-topmost', 0)

    def update_acq_time(self):
        """Updates acquisition period of instrument"""
        pass

    def close_frame(self):
        self.in_frame = False
        self.frame.destroy()

    @property
    def on_time(self):
        """Return datetime object of time to turn pi on. Date is not important, only time, so use arbitrary date"""
        return datetime.datetime(year=2020, month=1, day=1, hour=self.on_hour, minute=self.on_min)

    @property
    def off_time(self):
        """Return datetime object of time to turn pi off. Date is not important, only time, so use arbitrary date"""
        return datetime.datetime(year=2020, month=1, day=1, hour=self.off_hour, minute=self.off_min)

    @property
    def start_capt_time(self):
        """Return datetime object of time to turn start acq. Date is not important, only time, so use arbitrary date"""
        return datetime.datetime(year=2020, month=1, day=1, hour=self.capt_start_hour, minute=self.capt_start_min)

    @property
    def stop_capt_time(self):
        """Return datetime object of time to turn stop acq. Date is not important, only time, so use arbitrary date"""
        return datetime.datetime(year=2020, month=1, day=1, hour=self.capt_stop_hour, minute=self.capt_stop_min)

    @property
    def on_hour(self):
        return self._on_hour.get()

    @on_hour.setter
    def on_hour(self, value):
        self._on_hour.set(value)

    @property
    def on_min(self):
        return self._on_min.get()

    @on_min.setter
    def on_min(self, value):
        self._on_min.set(value)

    @property
    def off_hour(self):
        return self._off_hour.get()

    @off_hour.setter
    def off_hour(self, value):
        self._off_hour.set(value)

    @property
    def off_min(self):
        return self._off_min.get()

    @off_min.setter
    def off_min(self, value):
        self._off_min.set(value)

    @property
    def capt_start_hour(self):
        return self._capt_start_hour.get()

    @capt_start_hour.setter
    def capt_start_hour(self, value):
        self._capt_start_hour.set(value)

    @property
    def capt_start_min(self):
        return self._capt_start_min.get()

    @capt_start_min.setter
    def capt_start_min(self, value):
        self._capt_start_min.set(value)

    @property
    def capt_stop_hour(self):
        return self._capt_stop_hour.get()

    @capt_stop_hour.setter
    def capt_stop_hour(self, value):
        self._capt_stop_hour.set(value)

    @property
    def capt_stop_min(self):
        return self._capt_stop_min.get()

    @capt_stop_min.setter
    def capt_stop_min(self, value):
        self._capt_stop_min.set(value)