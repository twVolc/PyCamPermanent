# -*- coding: utf-8 -*-

"""Holds classes for building a menu bar in tkinter"""

from pycam.setupclasses import pycam_details
from pycam.gui.network import ConnectionGUI, instrument_cmd, run_pycam
import pycam.gui.cfg as cfg
from pycam.gui.cfg_menu_frames import geom_settings, process_settings, plume_bg, cell_calib, \
    opti_flow, light_dilution, cross_correlation, doas_fov, basic_acq_handler, automated_acq_handler,\
    calibration_wind, instrument_cfg, temp_log
from pycam.gui.misc import About, LoadSaveProcessingSettings
from pycam.io_py import save_pcs_line, load_pcs_line, save_light_dil_line, load_light_dil_line, create_video
import pycam.gui.settings as settings
from pycam.networking.FTP import FileTransferGUI
from pycam.cfg import pyplis_worker
from pycam.doas.cfg import doas_worker
from pycam.setupclasses import FileLocator
from pycam.networking.ssh import open_ssh, ssh_cmd, close_ssh

from pyplis import LineOnImage

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import time
import os
import threading
import multiprocessing


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
    def __init__(self, parent, frame, pyplis_work=pyplis_worker, doas_work=doas_worker):
        self.parent = parent
        self.frame = tk.Menu(frame)
        self.pyplis_worker = pyplis_work
        self.doas_worker = doas_work
        self.ftp_transfer = FileTransferGUI(cfg.ftp_client, self.pyplis_worker, self.doas_worker,
                                            cfg.current_dir_img, cfg.current_dir_spec, self)

        # Setup dictionary to hold menu tabs and list to hold the menu tab names. Each menu tab is held in the
        # dictionary with the assocaited key stored in keys. This makes cascade setup easy by looping through keys
        self.menus = dict()
        keys = list()

        # -----------------------------------------------------------------------------------------------
        # File tab
        tab = 'File'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        # Load options
        self.load_frame = LoadFrame(self.parent, pyplis_work=pyplis_worker, doas_work=doas_worker)
        self.submenu_load = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Load', menu=self.submenu_load)
        self.submenu_load.add_command(label='Load config file', command=self.load_frame.load_config_file)
        self.submenu_load.add_command(label='Load PCS line', command=self.load_frame.load_pcs)
        self.submenu_load.add_command(label='Load light dilution line', command=self.load_frame.load_dil)
        self.submenu_load.add_command(label='Load image registration', command=self.load_frame.load_img_reg)
        self.submenu_load.add_separator()
        self.submenu_load.add_command(label='Configure start-up', command=self.load_frame.generate_frame)

        # Save
        self.save_frame = SaveFrame(self.parent, pyplis_work=pyplis_worker)
        self.submenu_save = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Save', menu=self.submenu_save)
        self.submenu_save.add_command(label='Options', command=self.save_frame.generate_frame)

        # Export
        self.submenu_export = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Export', menu=self.submenu_export)
        self.submenu_export.add_command(label='Export image sequence to video', command=self.export_video)
        self.menus[tab].add_separator()

        self.menus[tab].add_command(label='Settings', command=lambda: Settings(self.parent))
        self.menus[tab].add_separator()
        self.menus[tab].add_command(label='Exit', command=self.parent.exit_app)

        # ---------------------------------------------------------------------------------------------
        # Instrument tab for interfacing with instrument
        tab = 'Instrument'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        # Configure instrument
        self.menus[tab].add_command(label='Configure', command=instrument_cfg.generate_frame)
        self.menus[tab].add_separator()

        # General networking commands
        self.menus[tab].add_command(label='Connect', command=cfg.indicator.connect_sock)
        self.menus[tab].add_command(label='Disconnect', command=cfg.indicator.disconnect_sock)

        self.submenu_cmd = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Commands', menu=self.submenu_cmd)
        self.submenu_cmd.add_command(label='Restart', command=lambda: instrument_cmd('RST'))
        self.submenu_cmd.add_command(label='Restart cameras', command=lambda: instrument_cmd('RSC'))
        self.submenu_cmd.add_command(label='Restart spectrometer', command=lambda: instrument_cmd('RSS'))
        self.submenu_cmd.add_separator()
        self.submenu_cmd.add_command(label='Run pycam (with automated capture)',
                                     command=lambda: run_pycam(cfg.sock.host_ip, auto_capt=1))
        self.submenu_cmd.add_command(label='Run pycam (without automated capture)',
                                     command=lambda: run_pycam(cfg.sock.host_ip, auto_capt=0))
        self.submenu_cmd.add_command(label='Stop pycam', command=lambda: instrument_cmd('EXT'))
        self.menus[tab].add_separator()

        # Data transfer
        self.submenu_data = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Data Transfer', menu=self.submenu_data)
        self.submenu_data.add_command(label='Start transfer', command=self.ftp_transfer.start_transfer)
        self.submenu_data.add_command(label='Start transfer (new images only)',
                                      command=lambda: self.ftp_transfer.start_transfer(new_only=True))
        self.submenu_data.add_command(label='Stop transfer', command=self.ftp_transfer.stop_transfer)
        self.submenu_data.add_command(label='Options', command=self.ftp_transfer.generate_frame)  # Add options such as directory to transfer to/from?? Maybe only transfer certain data - certain times etc
        self.submenu_data.add_separator()
        self.submenu_data.add_command(label='Get temperature log', command=temp_log.generate_frame)
        self.submenu_data.add_separator()
        self.submenu_data.add_command(label='Mount SSD', command=lambda: self.mount_ssd(cfg.ftp_client.host_ip))
        self.submenu_data.add_command(label='Unmount SSD', command=lambda: self.unmount_ssd(cfg.ftp_client.host_ip))
        self.submenu_data.add_command(label='SSD full download', command=cfg.ftp_client.full_ssd_download)
        self.submenu_data.add_command(label='Clear SSD data', command=lambda: self.clear_ssd(cfg.ftp_client.host_ip))
        self.submenu_data.add_command(label='Free space on SSD', command=lambda: self.free_ssd(cfg.ftp_client.host_ip))
        self.menus[tab].add_separator()

        # Manual acquisition
        self.submenu_acq = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Acquisition settings', menu=self.submenu_acq)
        self.submenu_acq.add_command(label='Manual acquisition', command=basic_acq_handler.build_manual_capture_frame)
        self.submenu_acq.add_separator()
        self.submenu_acq.add_command(label='Start automated acquisition',
                                     command=lambda: automated_acq_handler.acq_comm(start_cont=True))
        self.submenu_acq.add_command(label='Stop automated acquisition',
                                     command=automated_acq_handler.stop_cont)
        self.submenu_acq.add_separator()
        self.submenu_acq.add_command(label='Update all instrument settings', command=automated_acq_handler.acq_comm)
        self.submenu_acq.add_command(label='Update spectrometer settings', command=automated_acq_handler.acq_spec_full)
        self.submenu_acq.add_command(label='Update camera settings', command=automated_acq_handler.acq_cam_full)
        self.submenu_acq.add_command(label='Retrieve current settings',
                                     command=automated_acq_handler.get_instrument_settings)
        self.submenu_acq.add_separator()
        self.submenu_acq.add_command(label='Acquire dark set', command=automated_acq_handler.acq_darks)
        self.menus[tab].add_separator()

        # Geometry setup
        self.menus[tab].add_command(label='Geometry configuration', command=geom_settings.generate_frame)
        # -------------------------------------------------------------------------------------------------

        # File tab
        tab = 'Processing'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        # Processing submenu
        self.submenu_proc = tk.Menu(self.frame, tearoff=0)
        self.menus[tab].add_cascade(label='Post-Processing', menu=self.submenu_proc)
        self.submenu_proc.add_command(label='Load sequence', command=lambda: pyplis_worker.load_sequence(plot_bg=False))
        self.submenu_proc.add_separator()
        self.submenu_proc.add_command(label='Load DOAS results', command=doas_worker.load_results)
        self.submenu_proc.add_command(label='Load DOAS directory', command=doas_worker.load_dir)
        self.submenu_proc.add_command(label='Process DOAS', command=self.thread_doas_processing)
        self.submenu_proc.add_separator()
        self.submenu_proc.add_command(label='Run', command=pyplis_worker.process_sequence)
        self.submenu_proc.add_command(label='Stop processing', command=self.stop_sequence_processing)

        self.menus[tab].add_command(label='Background model', command=plume_bg.generate_frame)
        self.menus[tab].add_command(label='Settings', command=process_settings.generate_frame)
        self.menus[tab].add_separator()
        self.menus[tab].add_command(label='Unpack data', command=self.unpack_data)
        self.menus[tab].add_separator()
        self.disp_var = tk.IntVar()
        self.disp_var.set(0)
        self.menus[tab].add_checkbutton(label='Display only mode', var=self.disp_var, command=self.set_display_mode)

        # ---------------------------------------------------------------------------------------------------------

        # View tab - can be used for toggling between views (e.g., camera frame, DOAS frame, processing frame)
        tab = 'View'
        keys.append(tab)
        self.menus[tab] = tk.Menu(self.frame, tearoff=0)

        # More windows cascade
        self.submenu_windows = tk.Menu(self.frame, tearoff=0)
        self.submenu_windows.add_command(label="DOAS calibration", command=calibration_wind.generate_frame)
        self.submenu_windows.add_command(label='Cell calibration',
                                         command=lambda: cell_calib.update_plot(generate_frame=True))
        self.submenu_windows.add_command(label="Camera-DOAS calibration", command=doas_fov.generate_frame)
        self.submenu_windows.add_separator()
        self.submenu_windows.add_command(label='Plume velocity settings', command=opti_flow.generate_frame)
        self.submenu_windows.add_command(label='Cross-correlation plot', command=cross_correlation.generate_frame)
        self.submenu_windows.add_command(label='Light dilution settings', command=light_dilution.generate_frame)
        self.menus[tab].add_cascade(label="More windows", menu=self.submenu_windows)

        self.menus[tab].add_separator()
        self.init_var = tk.IntVar()
        self.init_var.set(1)
        self.menus[tab].add_radiobutton(label='Camera window', value=1, var=self.init_var,
                                        command=lambda: self.parent.windows.select(0))
        self.menus[tab].add_radiobutton(label='DOAS window',  value=0, var=self.init_var,
                                        command=lambda: self.parent.windows.select(1))
        self.menus[tab].add_radiobutton(label='Analysis window',  value=2, var=self.init_var,
                                        command=lambda: self.parent.windows.select(2))
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

    def not_implemented(self):
        """Function to call if the desired command has not been implemented yet"""
        messagebox.showinfo('Not implemented', 'This command is currently under development '
                                               'and therefore not available.')

    def thread_doas_processing(self):
        """Threads DOAS processsing so gui frees up"""
        thread = threading.Thread(target=doas_worker.start_processing_threadless, args=())
        thread.daemon = True
        thread.start()

    def export_video(self):
        """Controls generation of video from image sequence"""
        frame = tk.Toplevel()
        frame.title('Select band')

        bands = ['on', 'off']
        _bands = tk.StringVar()
        _bands.set(bands[0])

        ttk.Label(frame, text='Band:').grid(row=0, column=0, padx=5, pady=5)
        band_opts = ttk.OptionMenu(frame, _bands, 'on', *bands)
        band_opts.grid(row=0, column=1, padx=5, pady=5)

        button = ttk.Button(frame, text='Select', command=lambda: create_video(band=_bands.get()))
        button.grid(row=1, column=1, sticky='e', padx=5, pady=5)

    def set_display_mode(self):
        """Sets the display mode on click of checkbutton"""
        pyplis_worker.display_only = bool(self.disp_var.get())

    def mount_ssd(self, ip, username='pi', password='raspberry'):
        """
        Attempts to mount SSD on raspberry pi
        :param ip:  str     IP address of pi
        """
        try:
            client = open_ssh(ip, uname=username, pwd=password)
            ssh_cmd(client, 'python3 {}'.format(FileLocator.MOUNT_SSD_SCRIPT))
            close_ssh(client)
            messagebox.showinfo('SSD mounted', 'SSD should now be successfully mounted to the Raspberry Pi')
        except BaseException as e:
            messagebox.showerror('Error mounting SSD',
                                 'An error occurred when attempting to mount SSD vis SSH.\n'
                                 '{}'.format(e))

    def unmount_ssd(self, ip, username='pi', password='raspberry'):
        """
        Attempts to mount SSD on raspberry pi
        :param ip:  str     IP address of pi
        """
        try:
            client = open_ssh(ip, uname=username, pwd=password)
            ssh_cmd(client, 'python3 {}'.format(FileLocator.UNMOUNT_SSD_SCRIPT))
            close_ssh(client)
            messagebox.showinfo('SSD unmounted', 'SSD should now be successfully unmounted to the Raspberry Pi')
        except BaseException as e:
            messagebox.showerror('Error unmounting SSD',
                                 'An error occurred when attempting to mount SSD vis SSH.\n'
                                 '{}'.format(e))

    def clear_ssd(self, ip, username='pi', password='raspberry'):
        """
        Clears all SSD data by SSHing into pi then running the clear_ssd.py script
        :param ip:  str     IP address of pi
        """
        a = messagebox.askyesno('Clear all data?',
                                'The action will irreversibly clear all of the data currently saved on the SSD drive. '
                                'Are you sure you want to proceed?')
        if a:
            try:
                client = open_ssh(ip, uname=username, pwd=password)
                ssh_cmd(client, 'python3 {}'.format(FileLocator.CLEAR_SSD_SCRIPT))
                close_ssh(client)
                messagebox.showinfo('SSD cleared', 'SSD data has been cleared.')
            except BaseException:
                messagebox.showerror('Error deleting data',
                                     'An error occurred when attempting to clear SSD data. Please check connection to'
                                     ' instrument and try again')
        else:
            messagebox.showinfo('Aborted', 'Data clear was aborted. No data has been deleted.')

    def free_ssd(self, ip):
        """
        Frees space on SSD data by SSHing into pi then running the free_space_ssd.py script
        :param ip:  str     IP address of pi
        """
        self.space_frame = tk.Toplevel()
        lab = ttk.Label(self.space_frame, text='Define how much space to make available on SSD')
        lab.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        lab = ttk.Label(self.space_frame, text='Create space (GB):')
        lab.grid(row=1, column=0)

        # Free space entry
        free_space = tk.IntVar()
        free_space.set(50)
        entry = ttk.Spinbox(self.space_frame, from_=0, to=1000, increment=5, textvariable=free_space)
        entry.grid(row=1, column=1, sticky='nsew')

        butt = ttk.Button(self.space_frame,text='Create space', command=lambda: self._free_ssd(ip, free_space.get()))
        butt.grid(row=2, column=0, sticky='nsew')
        butt = ttk.Button(self.space_frame, text='Cancel', command=self.space_frame.destroy)
        butt.grid(row=2, column=1, sticky='nsew')

    def _free_ssd(self, ip, space, username='pi', password='raspberry'):
        """
        Frees space on SSD data by SSHing into pi then running the free_space_ssd.py script
        """
        a = messagebox.askyesno('Free up space?',
                                'The action will free up {}GB of space on the SSD. This may lead to '
                                'irreversible deletion of data.\n'
                                'Are you sure you want to proceed?'.format(space))
        if a:
            try:
                client = open_ssh(ip, uname=username, pwd=password)
                ssh_cmd(client, 'python3 {} {}'.format(FileLocator.FREE_SPACE_SSD_SCRIPT, space))
                close_ssh(client)
                messagebox.showinfo('SSD cleared', 'SSD data has been cleared.')
            except BaseException:
                messagebox.showerror('Error deleting data',
                                     'An error occurred when attempting to free space on SSD. Please check connection to'
                                     ' instrument and try again')
        else:
            messagebox.showinfo('Aborted', 'Data clear was aborted. No data has been deleted.')

        # Close frame
        self.space_frame.destroy()

    def unpack_data(self):
        """Controls unpacking of data"""
        directory = filedialog.askdirectory(title='Select directory containing subdirectories of data',
                                            initialdir=FileLocator.IMG_SPEC_PATH_WINDOWS)
        if not directory:
            return
        print('Unpacking camera data...')
        cfg.ftp_client.img_dir.unpack_data(directory)
        print('Unpacking spectrometer data...')
        cfg.ftp_client.spec_dir.unpack_data(directory)

    def stop_sequence_processing(self):
        """Stops sequence processing of SO2 camera and DOAS"""
        pyplis_worker.stop_sequence_processing()
        doas_worker.stop_sequence_processing()


class Settings:
    """Class to control the settings from the GUI toolbar"""
    def __init__(self, parent):
        self.frame = tk.Toplevel()
        self.frame.title('PyCam Settings')
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth()/1.2),
                                                 int(self.frame.winfo_screenheight()/1.2),
                                                 int(self.frame.winfo_screenwidth()/10),
                                                 int(self.frame.winfo_screenheight()/10)))

        # Setup notebook tabs
        style = parent.style
        style.configure('One.TNotebook.Tab', **parent.layout_old[0][1])
        self.windows = ttk.Notebook(self.frame, style='One.TNotebook.Tab')
        self.windows.pack(fill='both', expand=1, padx=5, pady=5)

        # Generate the frames for each tab
        self.connection_gui = ConnectionGUI(parent, self.windows)
        self.gui_settings = settings.SettingsFrame(parent, self.windows, settings=cfg.gui_setts)

        # Add the frames for each tab to the notebook
        self.windows.add(self.connection_gui.frame, text=self.connection_gui.name)
        self.windows.add(self.gui_settings.frame, text=self.gui_settings.name)


class LoadFrame(LoadSaveProcessingSettings):
    """
    Class giving options to load a range of variables, either during immediately or on startup
    """
    def __init__(self, main_gui, pyplis_work=pyplis_worker, doas_work=doas_worker, generate_frame=False,
                 init_dir=FileLocator.SAVED_OBJECTS):
        super().__init__()
        self.main_gui = main_gui
        self.pyplis_worker = pyplis_work
        self.doas_worker = doas_work
        self.init_dir = init_dir
        self.pdx, self.pdy = 2, 5
        self.sep = ','
        self.max_len_str = 50
        self.no_line = 'No line'
        self.img_reg_frame = None
        self.in_frame = False

        self.initiate_variables()
        self.load_defaults()

        if generate_frame:
            self.generate_frame()

    def initiate_variables(self):
        """Setup tk variables for save options"""
        self.vars = {'pcs_lines': str,
                     'img_registration': str,
                     'dil_lines': str,
                     'ld_lookup_1': str,
                     'ld_lookup_2': str
                     }
        self.num_pcs_lines = 5
        self._pcs_lines = [self.no_line] * self.num_pcs_lines
        self.img_registration = ''
        self.num_dil_lines = 5
        self._dil_lines = [self.no_line] * self.num_dil_lines
        self.ld_lookup_1 = 'None'
        self.ld_lookup_2 = 'None'

    def gather_vars(self):
        """Required for LoadSaveProcessSettings as it is called after loading. Do some general house keeping"""
        pass

    def generate_frame(self):
        """Build frame"""
        # Load the settings each time the frame is opened, so that we ensure we are working with current saved settings
        # rather than
        self.load_defaults()

        self.in_frame = True
        self.frame = tk.Toplevel()
        self.frame.attributes('-topmost', 1)    # Fix to top
        self.frame.title('Save options')
        self.frame.protocol('WM_DELETE_WINDOW', self.close_frame)
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth() / 1.2),
                                                 int(self.frame.winfo_screenheight() / 1.2),
                                                 int(self.frame.winfo_screenwidth() / 10),
                                                 int(self.frame.winfo_screenheight() / 10)))

        self.load_frame = ttk.LabelFrame(self.frame, text='Load objects on start-up', relief=tk.RAISED, borderwidth=3)
        self.load_frame.grid(row=0, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        row = 0
        pcs_frame = tk.LabelFrame(self.load_frame, text='ICA lines', relief=tk.RAISED, borderwidth=2,
                                  font=self.main_gui.main_font)
        pcs_frame.grid(row=row, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)
        self.pcs_lines_labs = [None] * self.num_pcs_lines
        self.pcs_add_butt = [None] * self.num_pcs_lines
        self.pcs_rem_butt = [None] * self.num_pcs_lines
        row_line = 0
        for i in range(self.num_pcs_lines):
            lab = ttk.Label(pcs_frame, text='File:', font=self.main_gui.main_font)
            lab.grid(row=row_line, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
            self.pcs_lines_labs[i] = ttk.Label(pcs_frame, text=self.pcs_lines_short[i], width=self.max_len_str,
                                               font=self.main_gui.main_font)
            self.pcs_lines_labs[i].grid(row=row_line, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
            self.pcs_add_butt[i] = ttk.Button(pcs_frame, text='Change line',
                                              command=lambda i=i: self.add_pcs_startup(i))
            self.pcs_add_butt[i].grid(row=row_line, column=2, sticky='ew', padx=self.pdx, pady=self.pdy)
            self.pcs_rem_butt[i] = ttk.Button(pcs_frame, text='Remove line',
                                              command=lambda i=i: self.remove_pcs_startup(i))
            self.pcs_rem_butt[i].grid(row=row_line, column=3, sticky='ew', padx=self.pdx, pady=self.pdy)
            row_line += 1

        # Image registration load
        row += 1
        img_reg_frame = tk.LabelFrame(self.load_frame, text='Image registration', relief=tk.RAISED, borderwidth=2,
                                      font=self.main_gui.main_font)
        img_reg_frame.grid(row=row, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)
        lab = ttk.Label(img_reg_frame, text='File:', font=self.main_gui.main_font)
        lab.grid(row=0, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        self.img_reg_lab = ttk.Label(img_reg_frame, text=self.img_reg_short, width=self.max_len_str,
                                     font=self.main_gui.main_font)
        self.img_reg_lab.grid(row=0, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        self.reg_add_butt = ttk.Button(img_reg_frame, text='Change', command=self.add_reg_startup)
        self.reg_add_butt.grid(row=0, column=2, sticky='ew', padx=self.pdx, pady=self.pdy)
        self.reg_rem_butt = ttk.Button(img_reg_frame, text='Remove', command=self.remove_reg_startup)
        self.reg_rem_butt.grid(row=0, column=3, sticky='ew', padx=self.pdx, pady=self.pdy)

        # Light dilution lines
        row += 1
        dil_frame = tk.LabelFrame(self.load_frame, text='Light dilution lines', relief=tk.RAISED, borderwidth=2,
                                  font=self.main_gui.main_font)
        dil_frame.grid(row=row, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)
        self.dil_lines_labs = [None] * self.num_dil_lines
        self.dil_add_butt = [None] * self.num_dil_lines
        self.dil_rem_butt = [None] * self.num_dil_lines
        for i in range(self.num_dil_lines):
            lab = ttk.Label(dil_frame, text='File:', font=self.main_gui.main_font)
            lab.grid(row=i, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
            self.dil_lines_labs[i] = ttk.Label(dil_frame, text=self.dil_lines_short[i], width=self.max_len_str,
                                               font=self.main_gui.main_font)
            self.dil_lines_labs[i].grid(row=i, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
            self.dil_add_butt[i] = ttk.Button(dil_frame, text='Change line',
                                              command=lambda i=i: self.add_dil_startup(i))
            self.dil_add_butt[i].grid(row=i, column=2, sticky='ew', padx=self.pdx, pady=self.pdy)
            self.dil_rem_butt[i] = ttk.Button(dil_frame, text='Remove line',
                                              command=lambda i=i: self.remove_dil_startup(i))
            self.dil_rem_butt[i].grid(row=i, column=3, sticky='ew', padx=self.pdx, pady=self.pdy)

        # Light dilution spectrometer lookup
        row += 1
        spec_dil_frame = tk.LabelFrame(self.load_frame, text='Spectrometer light dilution lookup',
                                       relief=tk.RAISED, borderwidth=2, font=self.main_gui.main_font)
        spec_dil_frame.grid(row=row, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)
        self.spec_dil_labs = [None, None]
        self.spec_dil_butt = [None, None]
        for i in range(2):
            lab = ttk.Label(spec_dil_frame, text='Fit window {}:'.format(i+1), font=self.main_gui.main_font)
            lab.grid(row=i, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
            self.spec_dil_labs[i] = ttk.Label(spec_dil_frame, text=self.ld_lookup_short[i], width=self.max_len_str,
                                              font=self.main_gui.main_font)
            self.spec_dil_labs[i].grid(row=i, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
            self.spec_dil_butt[i] = ttk.Button(spec_dil_frame, text='Change grid',
                                               command=lambda i=i: self.add_lookup_startup(i))
            self.spec_dil_butt[i].grid(row=i, column=2, sticky='e', padx=self.pdx, pady=self.pdy)

        row += 1
        update_butt = ttk.Button(self.load_frame, text='Save changes', command=lambda: self.set_defaults(self.frame))
        update_butt.grid(row=row, column=0, sticky='e', padx=self.pdx, pady=self.pdy)

    @property
    def pcs_lines(self):
        return [line for line in self._pcs_lines if line != self.no_line]

    @pcs_lines.setter
    def pcs_lines(self, lines):
        """Given a string which may contain multiple file paths we split it and individually set them to tk variables"""
        #lines = value.split(self.sep)
        for i, line in enumerate(lines):
            if i < self.num_pcs_lines:
                if line == '':
                    line = self.no_line
                self._pcs_lines[i] = line

    @property
    def pcs_lines_short(self):
        short_list = [''] * self.num_pcs_lines
        for i, line in enumerate(self._pcs_lines):
            if len(line) > self.max_len_str:
                short_list[i] = '...' + line[-self.max_len_str+3:]
            else:
                short_list[i] = line

        return short_list

    @property
    def dil_lines(self):
        return self.sep.join([line for line in self._dil_lines if line != self.no_line])

    @dil_lines.setter
    def dil_lines(self, value):
        """Given a string which may contain multiple file paths we split it and individually set them to tk variables"""
        lines = value.split(self.sep)
        for i, line in enumerate(lines):
            if i < self.num_dil_lines:
                if line == '':
                    line = self.no_line
                self._dil_lines[i] = line

    @property
    def dil_lines_short(self):
        short_list = [''] * self.num_dil_lines
        for i, line in enumerate(self._dil_lines):
            if len(line) > self.max_len_str:
                short_list[i] = '...' + line[-self.max_len_str+3:]
            else:
                short_list[i] = line

        return short_list

    @property
    def ld_lookup_short(self):
        short_list = ['None'] * 2
        for i, line in enumerate([self.ld_lookup_1, self.ld_lookup_2]):
            if len(line) > self.max_len_str:
                short_list[i] = '...' + line[-self.max_len_str+3:]
            else:
                short_list[i] = line
        return short_list

    @property
    def img_reg_short(self):
        if len(self.img_registration) > self.max_len_str:
            return '...' + self.img_registration[-self.max_len_str+3:]
        else:
            return self.img_registration

    def load_pcs(self, filename=None, new_line=False):
        """Loads PCS into GUI"""
        if filename is None:
            kwargs = {}
            if self.in_frame:
                kwargs['parent'] = self.frame
            filename = filedialog.askopenfilename(initialdir=self.init_dir, **kwargs)

        if len(filename) > 0:
            line = load_pcs_line(filename)
            if new_line:
                line_num = None
            else:
                line_num = self.pyplis_worker.fig_tau.current_ica
            self.pyplis_worker.fig_tau.add_pcs_line(line, line_num=line_num, force_add=True)

    def add_pcs_startup(self, num):
        """
        Edits pcs lines used at startup
        :param num:     int     Line index to add
        """
        filename = filedialog.askopenfilename(parent=self.frame, initialdir=os.path.join(self.init_dir, 'pcs_lines'),
                                              filetypes=[('Text file', '*.txt')])

        if len(filename) > 0:
            # Update line list
            self._pcs_lines[num] = filename
            self.pcs_lines_labs[num].configure(text=self.pcs_lines_short[num])

    def remove_pcs_startup(self, num):
        """
        Remove pcs lines used at startup
        :param num:     int     Line index to add
        """
        self._pcs_lines[num] = self.no_line
        self.pcs_lines_labs[num].configure(text=self.pcs_lines_short[num])

    def set_all_pcs_lines(self):
        """Loads all lines held in this object and updates pyplis_worker.fig tau to contain these lines"""
        for line in self._pcs_lines:
            if line is not self.no_line:
                self.load_pcs(filename=line, new_line=True)

    def load_dil(self, filename=None):
        """Loads light dilution line into GUI"""
        if filename is None:
            kwargs = {}
            if self.in_frame:
                kwargs['parent'] = self.frame
            filename = filedialog.askopenfilename(initialdir=os.path.join(self.init_dir, 'dil_lines'), **kwargs)

        if len(filename) > 0:
            line = load_light_dil_line(filename)
            self.pyplis_worker.fig_dilution.add_dil_line(line, force_add=True)

    def add_dil_startup(self, num):
        """
        Edits light dilution lines used at startup
        :param num:     int     Line index to add
        """
        filename = filedialog.askopenfilename(parent=self.frame, initialdir=os.path.join(self.init_dir, 'dil_lines'),
                                              filetypes=[('Text file', '*.txt')])

        if len(filename) > 0:
            # Update line list
            self._dil_lines[num] = filename
            self.dil_lines_labs[num].configure(text=self.dil_lines_short[num])

    def remove_dil_startup(self, num):
        """
        Remove light dilution lines used at startup
        :param num:     int     Line index to add
        """
        self._dil_lines[num] = self.no_line
        self.dil_lines_labs[num].configure(text=self.dil_lines_short[num])

    def set_all_dil_lines(self):
        """Loads all lines held in this object and updates pyplis_worker.fig_dil to contain these lines"""
        for line in self._dil_lines:
            if line is not self.no_line:
                self.load_dil(filename=line)

    def add_lookup_startup(self, num):
        """
        Edits light dilution lookup table used at startup
        :param num:     int     Lookup table index (fit window. 1 or 2)
        """
        filename = filedialog.askopenfilename(parent=self.frame, initialdir=FileLocator.LD_LOOKUP,
                                              filetypes=[('Numpy array', '*.npy')])

        if len(filename) > 0:
            # Update lookup
            setattr(self, 'ld_lookup_{}'.format(num + 1), filename)
            self.spec_dil_labs[num].configure(text=self.ld_lookup_short[num])

    def set_ld_lookups(self):
        """Loads all lookup tables"""
        for i, filepath in enumerate([self.ld_lookup_1, self.ld_lookup_2]):
            if filepath != 'None':
                self.load_lookup(filename=filepath, num=i)

    def load_lookup(self, filename=None, num=0):
        """Loads lookup table"""
        if filename is None:
            kwargs = {}
            if self.in_frame:
                kwargs['parent'] = self.frame
            filename = filedialog.askopenfilename(initialdir=FileLocator.LD_LOOKUP, **kwargs)

        # Load lookup table through the light dilution gui
        if len(filename) > 0:
            light_dilution.choose_grid(num, grid_path=filename)

    def add_reg_startup(self):
        """
        Edits image registration object used at startup
        """
        filename = filedialog.askopenfilename(parent=self.frame,
                                              initialdir=os.path.join(self.init_dir, 'image_registration'),
                                              filetypes=[('numpy file', '*.npy'), ('pickled file', '*.pkl')])

        if len(filename) > 0:
            # Update line list
            self.img_registration = filename
            self.img_reg_lab.configure(text=self.img_reg_short)

    def remove_reg_startup(self):
        """
        Remove image registration object used at startup
        """
        self.img_registration = ''
        self.img_reg_lab.configure(text=self.img_registration)

    def load_img_reg(self, filename=None, rerun=True):
        """Loads in image registration to the pyplis object
        :param bool rerun:  If true, the load directory is rerun, with the new image registration in place"""
        if filename is None:
            kwargs = {}
            if self.in_frame:
                kwargs['frame'] = self.frame
            init_dir = os.path.join(self.init_dir, 'image_registration')
            filename = filedialog.askopenfilename(initialdir=init_dir,
                                                  **kwargs)
            if len(filename) > 0:
                filename = os.path.join(init_dir, filename)
            else:
                return

        if os.path.exists(filename):
            self.pyplis_worker.img_reg.load_registration(filename, img_reg_frame=self.img_reg_frame, rerun=rerun)

    def load_all(self):
        """Runs all load functions to prepare pyplis worker"""
        self.set_all_pcs_lines()
        self.set_all_dil_lines()
        self.set_ld_lookups()
        # Don't need to run pyplis_worker.load_sequence on startup as it is run later elsewhere
        self.load_img_reg(filename=self.img_registration, rerun=False)

    def close_frame(self):
        """CLose window"""
        self.in_frame = False
        self.frame.destroy()

    def load_config_file(self):
        """Load in a config file selected by the user"""
        filename = filedialog.askopenfilename(
            title='Select config file',
            initialdir=self.init_dir)
        
        if len(filename) > 0:
            self.pyplis_worker.load_config(filename, "user")

            self.reload_config()

    def reload_config(self):
        self.load_defaults()
        self.main_gui.anal_wind.so2_img.load_defaults()
        process_settings.load_defaults()
        plume_bg.load_defaults()
        opti_flow.load_defaults()
        light_dilution.load_defaults()
        cross_correlation.load_defaults()
        doas_fov.load_defaults()

        calibration_wind.ils_frame.ILS_path = self.pyplis_worker.config["ILS_path"]

    def reset_pcs_lines(self):

        current_lines = [i for i, v in enumerate(self.pyplis_worker.fig_tau.PCS_lines_list) if v is not None]

        [self.pyplis_worker.fig_tau.del_ica(line_n) for line_n in current_lines]


class SaveFrame(LoadSaveProcessingSettings):
    """
    Class giving options to save a range of variables, either during processing or upon click
    """
    def __init__(self, main_gui, pyplis_work=pyplis_worker, generate_frame=False, init_dir=FileLocator.SAVED_OBJECTS):
        self.main_gui = main_gui
        self.pyplis_worker = pyplis_work
        self.init_dir = init_dir
        self.pdx, self.pdy = 2, 2

        self.initiate_variables()
        self.load_defaults()

        if generate_frame:
            self.generate_frame()

    def initiate_variables(self):
        """Setup tk variables for save options"""
        # Objects to save with a click
        self._pcs_line = tk.IntVar()
        self.pcs_ext = '.txt'

        self._dil_line = tk.IntVar()
        self.dil_ext = '.txt'

        self._img_reg = tk.StringVar()
        self.reg_ext = {'cp': '.pkl',
                        'cv': '.npy'}

        # Objects to save in processing
        self.vars = {'save_img_aa': int,
                     'save_img_cal': int,
                     'save_img_so2': int,
                     'save_doas_cal': int}

        self.img_types = ['.npy', '.mat']
        self._save_img_aa = tk.BooleanVar()
        self._type_img_aa = tk.StringVar()
        self.type_img_aa = self.img_types[0]
        self._save_img_cal = tk.BooleanVar()
        self._type_img_cal = tk.StringVar()
        self.type_img_cal = self.img_types[0]
        self._save_img_so2 = tk.BooleanVar()
        self._png_compression = tk.IntVar()
        self.png_compression = 0

        self._save_doas_cal = tk.BooleanVar()

    def gather_vars(self):
        self.pyplis_worker.save_img_aa = self.save_img_aa
        self.pyplis_worker.type_img_aa = self.type_img_aa

        self.pyplis_worker.save_img_cal = self.save_img_cal
        self.pyplis_worker.type_img_cal = self.type_img_cal

        self.pyplis_worker.save_img_so2 = self.save_img_so2
        self.pyplis_worker.png_compression = self.png_compression

        self.pyplis_worker.save_doas_cal = self.save_doas_cal

        if hasattr(self, 'frame'):
            tk.messagebox.showinfo('Save settings updated',
                                   'Save settings have been updated and will apply to next processing run',
                                   parent=self.frame)

    @property
    def img_reg(self):
        return self._img_reg.get()

    @property
    def save_img_aa(self):
        return self._save_img_aa.get()

    @save_img_aa.setter
    def save_img_aa(self, value):
        self._save_img_aa.set(value)

    @property
    def type_img_aa(self):
        return self._type_img_aa.get()

    @type_img_aa.setter
    def type_img_aa(self, value):
        self._type_img_aa.set(value)

    @property
    def save_img_cal(self):
        return self._save_img_cal.get()

    @save_img_cal.setter
    def save_img_cal(self, value):
        self._save_img_cal.set(value)

    @property
    def type_img_cal(self):
        return self._type_img_cal.get()

    @type_img_cal.setter
    def type_img_cal(self, value):
        self._type_img_cal.set(value)

    @property
    def save_img_so2(self):
        return self._save_img_so2.get()

    @save_img_so2.setter
    def save_img_so2(self, value):
        self._save_img_so2.set(value)

    @property
    def png_compression(self):
        return self._png_compression.get()

    @png_compression.setter
    def png_compression(self, value):
        self._png_compression.set(value)

    @property
    def save_doas_cal(self):
        return self._save_doas_cal.get()

    @save_doas_cal.setter
    def save_doas_cal(self, value):
        self._save_doas_cal.set(value)

    def generate_frame(self):
        """Build frame"""
        self.frame = tk.Toplevel()
        self.frame.attributes('-topmost', 1)    # Fix to top
        self.frame.title('Save options')
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth() / 1.2),
                                                 int(self.frame.winfo_screenheight() / 1.2),
                                                 int(self.frame.winfo_screenwidth() / 10),
                                                 int(self.frame.winfo_screenheight() / 10)))

        # SAVE NOW FRAME
        self.save_now_frame = ttk.LabelFrame(self.frame, text='Save objects now', relief=tk.RAISED, borderwidth=3)
        self.save_now_frame.grid(row=0, column=0, sticky='nsew', padx=self.pdx, pady=self.pdy)

        row = 0
        # pcs line saving
        self.pcs_lines = [int(x) for x in self.pyplis_worker.fig_series.lines]
        label = ttk.Label(self.save_now_frame, text='Save ICA line:', font=self.main_gui.main_font)
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        spin_pcs = ttk.OptionMenu(self.save_now_frame, self._pcs_line, self.pyplis_worker.fig_tau.current_ica, *self.pcs_lines)
        spin_pcs.grid(row=row, column=1, sticky='ew', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.save_now_frame, text='Save', command=self.save_pcs)
        butt.grid(row=row, column=2, stick='w', padx=self.pdx, pady=self.pdy)

        # light dilution line saving
        row += 1
        self.dil_lines = [i+1 for i, x in enumerate(self.pyplis_worker.fig_dilution.lines_pyplis)
                          if isinstance(x, LineOnImage)]
        label = ttk.Label(self.save_now_frame, text='Save light dilution line:', font=self.main_gui.main_font)
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        spin_dil = ttk.OptionMenu(self.save_now_frame, self._dil_line, self.pyplis_worker.fig_dilution.current_line,
                                  *self.dil_lines)
        spin_dil.grid(row=row, column=1, sticky='ew', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.save_now_frame, text='Save', command=self.save_dil)
        butt.grid(row=row, column=2, stick='w', padx=self.pdx, pady=self.pdy)

        # Image registration saving
        row += 1
        self.img_reg_opts = []
        for tform in self.reg_ext:
            if getattr(self.pyplis_worker.img_reg, 'got_{}_transform'.format(tform)):
                self.img_reg_opts.append(tform)
        label = ttk.Label(self.save_now_frame, text='Save image registration:', font=self.main_gui.main_font)
        label.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        spin_reg = ttk.OptionMenu(self.save_now_frame, self._img_reg, self.img_reg_opts[0] if self.img_reg_opts else '',
                                  *self.img_reg_opts if self.img_reg_opts else '')
        spin_reg.grid(row=row, column=1, sticky='ew', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(self.save_now_frame, text='Save', command=self.save_img_reg)
        butt.grid(row=row, column=2, stick='w', padx=self.pdx, pady=self.pdy)

        # PROCESSING SAVE OPTIONS
        self.save_proc_frame = ttk.LabelFrame(self.frame, text='Processing save options', relief=tk.RAISED,
                                              borderwidth=3)
        self.save_proc_frame.grid(row=0, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        row = 0

        # Img tau save
        check = ttk.Checkbutton(self.save_proc_frame, text='Save AA images', variable=self._save_img_aa)
        check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        type_frame = ttk.Frame(self.save_proc_frame, relief=tk.RAISED, borderwidth=3)
        type_frame.grid(row=row, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        lab = ttk.Label(type_frame, text='File type:', font=self.main_gui.main_font)
        lab.grid(row=0, column=0, sticky='w')
        img_types = ttk.OptionMenu(type_frame, self._type_img_aa, self.type_img_aa, *self.img_types)
        img_types.grid(row=0, column=1, sticky='nsew', padx=2)
        row += 1

        # Img cal save
        check = ttk.Checkbutton(self.save_proc_frame, text='Save calibrated images', variable=self._save_img_cal)
        check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        type_frame = ttk.Frame(self.save_proc_frame, relief=tk.RAISED, borderwidth=3)
        type_frame.grid(row=row, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        lab = ttk.Label(type_frame, text='File type:', font=self.main_gui.main_font)
        lab.grid(row=0, column=0, sticky='w')
        img_types = ttk.OptionMenu(type_frame, self._type_img_cal, self.type_img_cal, *self.img_types)
        img_types.grid(row=0, column=1, sticky='nsew', padx=2)
        row += 1

        # Img SO2 save
        check = ttk.Checkbutton(self.save_proc_frame, text='Save SO2 PNG', variable=self._save_img_so2)
        check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        type_frame = ttk.Frame(self.save_proc_frame, relief=tk.RAISED, borderwidth=3)
        type_frame.grid(row=row, column=1, sticky='nsew', padx=self.pdx, pady=self.pdy)
        lab = ttk.Label(type_frame, text='Compression:', font=self.main_gui.main_font)
        lab.grid(row=0, column=0, sticky='w')
        img_types = ttk.Spinbox(type_frame, textvariable=self._png_compression, width=3, from_=0, to=9, increment=1,
                                font=self.main_gui.main_font)
        img_types.grid(row=0, column=1, sticky='nsew', padx=2)
        row += 1

        # DOAS fit save
        check = ttk.Checkbutton(self.save_proc_frame, text='Save DOAS-AA calibration', variable=self._save_doas_cal)
        check.grid(row=row, column=0, sticky='w', padx=self.pdx, pady=self.pdy)
        row += 1

        # BUTTONS
        butt_frame = ttk.Frame(self.save_proc_frame)
        butt_frame.grid(row=row, column=0, columnspan=2, sticky='nsew')
        butt_frame.grid_columnconfigure(0, weight=1)
        butt = ttk.Button(butt_frame, text='Apply settings', command=self.gather_vars)
        butt.grid(row=0, column=0, sticky='e', padx=self.pdx, pady=self.pdy)
        butt = ttk.Button(butt_frame, text='Set as defaults', command=self.set_defaults)
        butt.grid(row=0, column=1, sticky='e', padx=self.pdx, pady=self.pdy)

    def save_pcs(self):
        """Saves PCS line"""
        if len(self.pyplis_worker.PCS_lines_all) == 0:
            print('There are no lines to be saved. Please draw an ICA line first')
            return

        filename = filedialog.asksaveasfilename(parent=self.frame, initialdir=os.path.join(self.init_dir, 'pcs_lines'))

        if len(filename) > 0:
            if self.pcs_ext not in filename:
                filename += self.pcs_ext

            # Adjust line number by -1 to get the index in python indexing (start at 0)
            line_idx = self._pcs_line.get() - 1
            line = self.pyplis_worker.PCS_lines_all[line_idx]

            # Save line
            save_pcs_line(line, filename)

    def save_dil(self):
        """Saves PCS line"""
        if len(self.dil_lines) == 0:
            print('There are no lines to be saved. Please draw an light dilution line first')
            return

        filename = filedialog.asksaveasfilename(parent=self.frame, initialdir=os.path.join(self.init_dir, 'dil_lines'))

        if len(filename) > 0:
            if self.dil_ext not in filename:
                filename += self.dil_ext

            # Adjust line number by -1 to get the index in python indexing (start at 0)
            line_idx = self._dil_line.get() - 1
            line = self.pyplis_worker.fig_dilution.lines_pyplis[line_idx]

            # Save line
            save_light_dil_line(line, filename)

    def save_img_reg(self):
        """
        Saves image registration object
        :return:
        """
        if self.img_reg == '':
            return

        if self.img_reg == 'cv':
            filetypes = (('numpy file', '*.npy'), ("All Files", "*.*"))
        elif self.img_reg == 'cp':
            filetypes = (('pickled file', '*.pkl'), ("All Files", "*.*"))

        filename = filedialog.asksaveasfilename(parent=self.frame,
                                                initialdir=os.path.join(self.init_dir, 'image_registration'),
                                                defaultextension=self.reg_ext[self.img_reg],
                                                filetypes=filetypes)

        if len(filename) > 0:
            if self.reg_ext[self.img_reg] not in filename:
                filename += self.reg_ext[self.img_reg]
            self.pyplis_worker.img_reg.save_registration(filename, method=self.img_reg)
