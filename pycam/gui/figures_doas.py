import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import queue

from pycam.doas.doas_worker import DOASWorker
from pycam.doas.cfg import doas_worker, species
from pycam.gui.cfg import gui_setts
# from acquisition_gui import AcquisitionFrame
from pycam.cfg import pyplis_worker

plt.style.use('dark_background')

refresh_rate = 100


class SpectraPlot:
    """
    Generates a widget containing 3 subplots of spectra -> dark, clear (Fraunhofer), in-plume
    """

    def __init__(self, root, frame, doas_plot=None):
        self.root = root
        self.doas_worker = doas_worker
        self.doas_plot = doas_plot

        # Give pyplis_worker and doas_worker access to this figure
        pyplis_worker.fig_spec = self
        doas_worker.fig_spec = self

        self.figsize = gui_setts.fig_spec
        self.dpi = gui_setts.dpi

        self.max_DN = 2**16 - 1  # Maximum DN for spectrometer

        # Could use threads and queues to update plots, or just us simple functions which Acquisition frame calls
        self.q = queue.Queue()

        self.__setup_gui__(frame)

    def __setup_gui__(self, frame):
        """Organise widget"""
        self.frame = ttk.Frame(frame, relief=tk.RAISED, borderwidth=4)

        # -----------------------------------------------
        # WIDGET SETUP
        # ------------------------------------------------
        self.frame2 = ttk.Frame(self.frame)
        self.frame2.pack(side=tk.TOP, fill=tk.X, expand=1)

        # STRAY RANGE
        self.stray_start = tk.DoubleVar()
        self.stray_start.set(self.doas_worker.start_stray_wave)
        self.stray_box_start = ttk.Spinbox(self.frame2, from_=0, to=400, increment=0.1, width=5, format='%.1f',
                                             textvariable=self.stray_start, command=self.update_stray_start)
        self.stray_box_start.set('{:.1f}'.format(self.doas_worker.start_stray_wave))
        self.stray_end = tk.DoubleVar()
        self.stray_end.set(self.doas_worker.end_stray_wave)
        self.stray_box_end = ttk.Spinbox(self.frame2, from_=1, to=400, increment=0.1, width=5, format='%.1f',
                                           textvariable=self.stray_end, command=self.update_stray_end)
        self.stray_box_end.set('{:.1f}'.format(self.doas_worker.end_stray_wave))

        label = tk.Label(self.frame2, text='Stray light correction (min.):').pack(side=tk.LEFT)
        self.stray_box_start.pack(side=tk.LEFT)
        label = tk.Label(self.frame2, text='Stray light correction (max.):').pack(side=tk.LEFT)
        self.stray_box_end.pack(side=tk.LEFT)

        # FIT WINDOW
        self.fit_wind_start = tk.DoubleVar()
        self.fit_wind_start.set(self.doas_worker.start_fit_wave)
        self.fit_wind_box_start = ttk.Spinbox(self.frame2, from_=0, to=400, increment=0.1, width=5, format='%.1f',
                                             textvariable=self.fit_wind_start, command=self.update_fit_wind_start)
        self.fit_wind_box_start.set('{:.1f}'.format(self.doas_worker.start_fit_wave))
        self.fit_wind_end = tk.DoubleVar()
        self.fit_wind_end.set(self.doas_worker.end_fit_wave)
        self.fit_wind_box_end = ttk.Spinbox(self.frame2, from_=1, to=400, increment=0.1, width=5, format='%.1f',
                                           textvariable=self.fit_wind_end, command=self.update_fit_wind_end)
        self.fit_wind_box_end.set('{:.1f}'.format(self.doas_worker.end_fit_wave))

        self.fit_wind_box_end.pack(side=tk.RIGHT)
        label = tk.Label(self.frame2, text='Fit wavelength (max.):').pack(side=tk.RIGHT)
        self.fit_wind_box_start.pack(side=tk.RIGHT)
        label = tk.Label(self.frame2, text='Fit wavelength (min.):').pack(side=tk.RIGHT)

        # ------------------------------------------------
        # FIGURE SETUP
        # ------------------------------------------------
        self.fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)

        self.ax = self.fig.subplots(1, 1)
        self.ax.set_ylabel('DN')
        self.ax.set_ylim([0, self.max_DN])
        self.ax.set_xlim([250, 400])
        self.ax.set_xlabel('Wavelength [nm]')
        self.ax.grid(True)
        self.plt_colours = ['b', 'g', 'r']
        for i in range(3):
            self.ax.plot([250, 400], [0, 0], self.plt_colours[i], linewidth=1)
        self.ax.legend(('Dark', 'Clear', 'Plume'), loc=2, framealpha=1)
        self.fig.tight_layout()

        # Stray light
        self.min_stray_line = self.ax.plot([self.doas_worker.start_stray_wave,self.doas_worker.start_stray_wave],
                                     [0, self.max_DN], 'm')
        self.max_stray_line = self.ax.plot([self.doas_worker.end_stray_wave,self.doas_worker.end_stray_wave],
                                     [0, self.max_DN], 'm')

        width = self.doas_worker.end_stray_wave - self.doas_worker.start_stray_wave
        self.stray_range = Rectangle((self.doas_worker.start_stray_wave,0), width=width, height=self.max_DN,
                                  fill=True, facecolor='m', alpha=0.2)
        self.stray_range_patch = self.ax.add_patch(self.stray_range)

        # Fit window
        self.min_line = self.ax.plot([self.doas_worker.start_fit_wave, self.doas_worker.start_fit_wave],
                                     [0, self.max_DN], 'y')
        self.max_line = self.ax.plot([self.doas_worker.end_fit_wave, self.doas_worker.end_fit_wave],
                                     [0, self.max_DN], 'y')

        width = self.doas_worker.end_fit_wave - self.doas_worker.start_fit_wave
        self.fit_wind = Rectangle((self.doas_worker.start_fit_wave, 0), width=width, height=self.max_DN,
                                  fill=True, facecolor='y', alpha=0.2)
        self.fit_wind_patch = self.ax.add_patch(self.fit_wind)

        # Organise and draw canvas
        self.canv = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canv.draw()
        self.canv.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Instigate canvas drawing worker
        self.__draw_canv__()

    def update_dark(self):
        """Update dark plot with new data"""
        self.ax.lines[0].set_data(self.doas_worker.wavelengths, self.doas_worker.dark_spec)
        self.ax.set_xlim([self.doas_worker.wavelengths[0],self.doas_worker.wavelengths[-1]])
        self.q.put(1)

    def update_clear(self):
        """Update clear plot with new data"""
        self.ax.lines[1].set_data(self.doas_worker.wavelengths, self.doas_worker.clear_spec_raw)
        self.ax.set_xlim([self.doas_worker.wavelengths[0], self.doas_worker.wavelengths[-1]])
        self.q.put(1)

    def update_plume(self):
        """Update clear plot with new data"""
        self.ax.lines[2].set_data(self.doas_worker.wavelengths, self.doas_worker.plume_spec_raw)
        self.ax.set_xlim([self.doas_worker.wavelengths[0], self.doas_worker.wavelengths[-1]])
        self.q.put(1)

    def update_stray_start(self):
        """Updates stray light range on plot"""
        stray_start = self.stray_start.get()

        # Ensure the end of the stray range doesn't become less than the start
        if stray_start >= self.stray_end.get():
            stray_start = self.stray_end.get() - 0.1
            self.stray_start.set(stray_start)

        # Update DOASWorker with new stray range
        self.doas_worker.start_stray_wave = stray_start

        # Update plot
        self.min_stray_line[0].set_data([self.doas_worker.start_stray_wave, self.doas_worker.start_stray_wave], [0, self.max_DN])
        self.stray_range.set_x(self.doas_worker.start_stray_wave)
        self.stray_range.set_width(self.doas_worker.end_stray_wave - self.doas_worker.start_stray_wave)
        self.q.put(1)

        # Process doas if we can (if it has previously been processed so we know we have all the data we need)
        if self.doas_worker.processed_data:
            self.doas_worker.stray_corrected = False
            self.doas_worker.process_doas()
            self.doas_plot.update_plot()

    def update_stray_end(self):
        """Updates stray light range on plot"""
        stray_end = self.stray_end.get()

        # Ensure the end of the stray range doesn't become less than the start
        if stray_end <= self.stray_start.get():
            stray_end = self.stray_start.get() + 0.1
            self.stray_end.set(stray_end)

        # Update DOASWorker with new stray range
        self.doas_worker.end_stray_wave = stray_end

        # Update plot
        self.max_stray_line[0].set_data([self.doas_worker.end_stray_wave, self.doas_worker.end_stray_wave], [0, self.max_DN])
        self.stray_range.set_width(self.doas_worker.end_stray_wave - self.doas_worker.start_stray_wave)
        self.q.put(1)

        if self.doas_worker.processed_data:
            self.doas_worker.stray_corrected = False
            self.doas_worker.process_doas()
            self.doas_plot.update_plot()

    def update_fit_wind_start(self):
        """updates fit window on plot"""
        fit_wind_start = self.fit_wind_start.get()

        # Ensure the end of the fit window doesn't become less than the start
        if fit_wind_start >= self.fit_wind_end.get():
            fit_wind_start = self.fit_wind_end.get() - 0.1
            self.fit_wind_start.set(fit_wind_start)

        # Update DOASWorker with new fit window
        self.doas_worker.start_fit_wave = fit_wind_start

        # Update plot
        self.min_line[0].set_data([self.doas_worker.start_fit_wave, self.doas_worker.start_fit_wave], [0, self.max_DN])
        self.fit_wind.set_x(self.doas_worker.start_fit_wave)
        self.fit_wind.set_width(self.doas_worker.end_fit_wave-self.doas_worker.start_fit_wave)
        self.q.put(1)

        if self.doas_worker.processed_data:
            self.doas_worker.process_doas()
            self.doas_plot.update_plot()

    def update_fit_wind_end(self):
        """updates fit window on plot"""
        fit_wind_end = self.fit_wind_end.get()

        # Ensure the end of the fit window doesn't become less than the start
        if fit_wind_end <= self.fit_wind_start.get():
            fit_wind_end = self.fit_wind_start.get() + 0.1
            self.fit_wind_end.set(fit_wind_end)

        # Update DOASWorker with new fit window
        self.doas_worker.end_fit_wave = fit_wind_end

        # Update plot
        self.max_line[0].set_data([self.doas_worker.end_fit_wave, self.doas_worker.end_fit_wave], [0, self.max_DN])
        self.fit_wind.set_width(self.doas_worker.end_fit_wave - self.doas_worker.start_fit_wave)
        self.q.put(1)

        if self.doas_worker.processed_data:
            self.doas_worker.process_doas()
            self.doas_plot.update_plot()

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.q.get(block=False)
            # print('Got {} from q'.format(update))
            if update == 1:
                self.canv.draw()
            else:
                print('Closing canvas drawing')
                return
        except queue.Empty:
            pass
        self.root.after(refresh_rate, self.__draw_canv__)

    def close_widget(self):
        """Closes widget cleanly, by stopping __draw_canv__()"""
        self.q.put(2)
        # print('Added to q')


class DOASPlot:
    """
    Generates a widget containing the DOAS fit plot
    """
    def __init__(self, root, frame, figsize=(10, 3), species='SO2'):
        self.root = root

        self.doas_worker = doas_worker

        # Give pyplis_worker and doas_worker access to this figure
        pyplis_worker.fig_doas = self
        doas_worker.fig_doas = self

        self.species = species

        self.figsize = gui_setts.fig_doas
        self.dpi = gui_setts.dpi

        self.Q = queue.Queue()

        self.acq_obj = None

        self.__setup_gui__(frame)

    def __setup_gui__(self, frame):
        """Organise widget"""
        self.frame = ttk.Frame(frame, relief=tk.RAISED, borderwidth=4)

        # -----------------------------------------------
        # WIDGET SETUP
        # ------------------------------------------------
        self.frame2 = ttk.Frame(self.frame)
        self.frame2.pack(side=tk.TOP, fill=tk.X, expand=1)

        label = tk.Label(self.frame2, text='Shift spectrum:').pack(side=tk.LEFT)
        # label.grid(row=0, column=0)
        self.shift = tk.IntVar()
        self.shift.set(self.doas_worker.shift)
        self.shift_box = ttk.Spinbox(self.frame2, from_=-20, to=20, increment=1, width=3,
                                             textvariable=self.shift, command=self.update_shift)
        # self.fit_wind_box_start.grid(row=0, column=1)
        self.shift_box.pack(side=tk.LEFT)

        label2 = tk.Label(self.frame2, text='Stretch spectrum:').pack(side=tk.LEFT)
        # label2.grid(row=0, column=2)
        self.stretch = tk.IntVar()
        self.stretch.set(self.doas_worker.stretch)
        self.stretch_box = ttk.Spinbox(self.frame2, from_=-999, to=999, increment=1, width=4,
                                           textvariable=self.stretch, command=self.update_stretch)
        # self.fit_wind_box_end.grid(row=0, column=3)
        self.stretch_box.pack(side=tk.LEFT)

        # Save button
        self.save_butt = ttk.Button(self.frame2, text='Save spectra', command=self.save_spectra)
        self.save_butt.pack(side=tk.RIGHT, anchor='e')

        # ----------------------------------------------------------------
        # Tabbed figure setup for all species and residual
        # ----------------------------------------------------------------
        # Setup up tab wideget for each species
        self.tabs = ttk.Notebook(self.frame)
        self.tabs.bind('<Button-1>', self.__update_tab__)
        self.species_tabs = dict()
        self.species_tabs['Total'] = ttk.Frame(self.tabs, borderwidth=2)
        self.tabs.add(self.species_tabs['Total'], text='Total')

        # Loop through species if we have a list or tuple. Otherwise we make a single tab for the species
        self.species_plots = dict()

        if isinstance(self.species, list) or isinstance(self.species, tuple):
            for spec in self.species:
                self.species_tabs[spec] = ttk.Frame(self.tabs, borderwidth=2)
                self.tabs.add(self.species_tabs[spec], text=spec)

                # Generate figure
                self.species_plots[spec] = DOASFigure(self.species_tabs[spec], self.doas_worker, spec,
                                                      self.figsize, self.dpi)
        elif isinstance(self.species, str):
            self.species_tabs[self.species] = ttk.Frame(self.tabs, borderwidth=2)
            self.tabs.add(self.species_tabs[self.species], text=self.species)

            # Generate figure
            self.species_plots[self.species] = DOASFigure(self.species_tabs[self.species], self.doas_worker,
                                                          self.species, self.figsize, self.dpi)
        else:
            raise TypeError('Incompatible type {} for species object'.format(type(self.species)))

        self.species_tabs['residual'] = ttk.Frame(self.tabs, borderwidth=2)
        self.tabs.add(self.species_tabs['residual'], text='residual')
        self.tabs.pack(side=tk.TOP, fill="both", expand=1)

        # Generate DOASFigure object for full absorbance
        self.species_plots['Total'] = DOASFigure(self.species_tabs['Total'], self.doas_worker, 'Total',
                                                    self.figsize, self.dpi)

        # Generate DOASFigure object for residual
        self.species_plots['residual'] = DOASFigure(self.species_tabs['residual'], self.doas_worker, 'residual',
                                                    self.figsize, self.dpi)
        # Instigate canvas drawing worker
        self.__draw_canv__()

    def update_shift(self):
        """Updates DOASWorker shift value for aligning spectra"""
        # Set shift in DOASWorker object
        self.doas_worker.shift = self.shift.get()

        # If we have a processed spectrum we now must update it
        if self.doas_worker.processed_data:
            self.doas_worker.process_doas()
            self.update_plot()

    def update_stretch(self):
        """Updates DOASWorker stretch value for aligning spectra"""
        self.doas_worker.stretch = self.stretch.get()

        # If we have a processed spectrum we now must update it
        if self.doas_worker.processed_data:
            self.doas_worker.process_doas()
            self.update_plot()

    def update_plot(self):
        """Updates doas plot"""
        for plot in self.species_plots:
            self.species_plots[plot].update_plot()

        # Draw updates
        self.Q.put(1)

    def __update_tab__(self, event):
        """
        Controls drawing of tab canvas when tab is selected
        Drawing in this manner reduces lag when playing with a figure, as only the current figure is drawn
        :return:
        """
        self.Q.put(1)

    def __draw_canv__(self):
        """Draws canvas periodically"""
        try:
            update = self.Q.get(block=False)
            # print('Got message')
            if update == 1:
                # Only draw canvas of currently selected tab (save processing power)
                species = self.tabs.tab(self.tabs.select(), "text")
                self.species_plots[species].canv.draw()
            else:
                return
        except queue.Empty:
            pass
        self.root.after(refresh_rate, self.__draw_canv__)

    def close_widget(self):
        """Closes widget cleanly, by stopping __draw_canv__()"""
        self.Q.put(2)

    def save_spectra(self):
        """Saves processed DOAS spectra"""
        # Need to redo this as acquisition frame won't exist in this code
        pass
        #
        # if isinstance(self.acq_obj, AcquisitionFrame):
        #     print('Saving...')
        #     self.acq_obj.save_processed_spec()


class DOASFigure:
    """
    Class for generating a DOAS-style figure with absorbance and reference spectrum fitting
    """
    def __init__(self, frame, doas_worker, species, figsize, dpi):
        self.frame = frame
        self.doas_worker = doas_worker
        self.species = species
        self.figsize = figsize
        self.dpi = dpi
        # ------------------------------------------------
        # FIGURE SETUP
        # ------------------------------------------------
        self.fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)

        self.ax = self.fig.subplots(1, 1)
        self.ax.set_ylabel('Absorbance')
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_xlim([self.doas_worker.start_fit_wave, self.doas_worker.end_fit_wave])
        self.ax.set_xlabel('Wavelength [nm]')
        self.ax.grid(True)
        self.plt_colours = ['b', 'r']
        for i in range(2):
            self.ax.plot([250, 400], [0, 0], self.plt_colours[i], linewidth=1)
        self.ax.legend(('Measured', 'Fitted reference'), loc=1, framealpha=1)
        self.ax.set_title('SO2 Column density [ppm.m]: N/A          STD Error: N/A')
        self.fig.tight_layout()

        self.canv = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canv.draw()
        self.canv.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_plot(self):
        """Updates doas plot"""
        # Update plot lines with new data
        # self.ax.lines[0].set_data(self.doas_worker.wavelengths_cut, self.doas_worker.abs_spec_cut[self.species])
        self.ax.lines[0].set_data(self.doas_worker.wavelengths_cut, self.doas_worker.abs_spec_species[self.species])

        if self.species != 'residual':
            self.ax.lines[1].set_data(self.doas_worker.wavelengths_cut, self.doas_worker.ref_spec_fit[self.species])

        # Set axis limits
        self.ax.set_xlim([self.doas_worker.wavelengths_cut[0], self.doas_worker.wavelengths_cut[-1]])
        ylims = np.amax(np.absolute(self.doas_worker.abs_spec_species[self.species]))
        ylims *= 1.15
        if ylims == 0:
            ylims = 0.05
        self.ax.set_ylim([-ylims, ylims])
        self.ax.set_title('SO2 Column density [ppm.m]: {}          STD Error: {}'.format(
            self.doas_worker.column_density['SO2'], self.doas_worker.std_err))


class ScatterCDs:
    """
    Generates scatter plot of column density vs apparent absorbances
    """
    def __init__(self):
        pass


class CalibrationWindow:
    """
    Generates top-level window containing calibration figures for DOAS instruments.
    """
    def __init__(self):
        # Setup reference spectra objects
        self.ref_frame = dict()
        for spec in species:
            species_id = 'ref_spec_{}'.format(spec)
            self.ref_frame[spec] = RefPlot(ref_spec_path=species[spec], species=spec)

    def generate_frame(self):
        self.frame = tk.Toplevel()
        self.frame.title('DOAS calibration')
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth() / 1.2),
                                                 int(self.frame.winfo_screenheight() / 1.2),
                                                 int(self.frame.winfo_screenwidth() / 10),
                                                 int(self.frame.winfo_screenheight() / 10)))

        for frame in self.ref_frame:
            self.ref_frame[frame].__setup_gui__(self.frame)
            self.ref_frame[frame].frame.pack(side=tk.TOP, anchor='nw', expand=1, fill=tk.BOTH, padx=5, pady=5)

class RefPlot:
    """
    Plots and allows interaction with the reference spectrum
    If no frame is passed initially, the settings are still loaded up. The frame can then be generated later.

    Parameters
    ----------
    frame: tk.Frame, ttk.Frame
        Frame to place widget in
    doas_worker: DOASWorker
        Main object for doas processing
    init_dir: str
        Directory used as starting directory when prompted at file search boxes
    ref_spec_path: str
        Path to reference spectrum for preloading
    species: str
        Defines the species for reference spectrum
    """
    def __init__(self, frame=None, doas_worker=DOASWorker(), init_dir='.\\', ref_spec_path=None, species='SO2'):
        self.frame = frame
        self.doas_worker = doas_worker  # DOAS processor
        self.species = species          # Gas species
        self.init_dir = init_dir        # Inital directory for loading dialog box
        self.fig_size = gui_setts.fig_ref_spec    # Figure size (tuple)
        self.dpi = gui_setts.dpi                  # DPI of figure

        self.setts = gui_setts
        self.pdx = 5
        self.pdy = 5

        self.min_wavelength = 250       # Minimum wavelength shown on plot
        self.max_wavelength = 350       # Maximum wavelength shown on plot

        # If we have a pre-loaded reference spectrum plot it
        self.ref_spec_path = ref_spec_path
        if self.ref_spec_path is not None:
            self.load_ref_spec(init_load=True)

        # Setup gui
        if self.frame is not None:
            self.__setup_gui__(self.frame)

    def __setup_gui__(self, frame):
        # -------------------------
        # Reference Spectrum Setup
        # -------------------------
        self.frame = ttk.LabelFrame(frame, text='{} Reference Spectrum'.format(self.species),
                                    relief=tk.RAISED, borderwidth=2)
        self.frame.pack(side=tk.LEFT, padx=self.pdx, pady=self.pdy)

        self.loadRefFrame = ttk.Frame(self.frame)
        self.loadRefFrame.grid(row=0, column=0, sticky='w')
        label = ttk.Label(self.loadRefFrame, text='Filename:')
        label.grid(row=0, column=0, padx=self.pdx, pady=self.pdy)
        self.nameRef = ttk.Label(self.loadRefFrame, text=self.ref_spec_path)
        self.nameRef.grid(row=0, column=1, padx=self.pdx, pady=self.pdy)
        self.selectRef = ttk.Button(self.loadRefFrame, text='Load Spectrum', command=self.choose_ref_spec)
        self.selectRef.grid(row=0, column=2, padx=self.pdx, pady=self.pdy)

        self.conv_button = ttk.Button(self.frame, text='Convolve with ILS', command=self.conv_ref)
        self.conv_button.grid(row=0, column=1)

        # --------------------------------------------
        # FIGURE SETUP
        # --------------------------------------------
        self.FigRef = plt.Figure(figsize=self.fig_size, dpi=self.dpi)
        self.ax_SO2 = self.FigRef.add_subplot(111)

        self.ax_SO2.set_title('Reference spectrum: None')
        self.ax_SO2.set_ylabel(r'Absorption Cross Section [cm$^2$/molecule]')
        self.ax_SO2.set_xlabel('Wavelength [nm]')
        self.ax_SO2.tick_params(axis='both', direction='in', top='on', right='on')


        for i in range(2):
            self.ax_SO2.plot([250, 400], [0, 0], linewidth=1)
        self.ax_SO2.legend(('Reference', 'ILS-convolved'), loc=1, framealpha=1)

        self.refCanvas = FigureCanvasTkAgg(self.FigRef, master=self.frame)
        self.refCanvas.draw()
        self.refCanvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

        # Plot initial reference spectrum
        self.plot_ref_spec()
        # --------------------------------------------------------------------------------------------------------------

    def choose_ref_spec(self):
        """Load reference spectrum"""
        self.ref_spec_path = filedialog.askopenfilename(initialdir=self.init_dir,
                                                        title='Select reference spectrum',
                                                        filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        self.load_ref_spec()

    def load_ref_spec(self, init_load=False):
        """Loads reference spectrum"""
        if not self.ref_spec_path:
            return
        if not init_load:
            if len(self.ref_spec_path) > 53:
                self.nameRef.configure(text='...' + self.ref_spec_path[-50:])
            else:
                self.nameRef.configure(text=self.ref_spec_path)
        self.doas_worker.load_ref_spec(self.ref_spec_path, self.species)

        self.ref_spec_file = self.ref_spec_path.split('/')[-1]

        if not init_load:
            # Set convolution plot to zero then update
            self.ax_SO2.lines[1].set_data([self.doas_worker.ref_spec[self.species][0, 0],
                                           self.doas_worker.ref_spec[self.species][-1,0]], [0,0])

            # Plot reference spectrum
            self.plot_ref_spec()

    def plot_ref_spec(self):
        """Plot up reference spectrum"""
        self.ax_SO2.lines[0].set_data(self.doas_worker.ref_spec[self.species][:, 0], self.doas_worker.ref_spec[self.species][:, 1])
        self.ax_SO2.set_xlim(self.min_wavelength, self.max_wavelength)
        self.ax_SO2.set_ylim([0, np.amax(self.doas_worker.ref_spec[self.species][:, 1])])
        if len(self.ref_spec_path) > 53:
            ref_spec_abbr = '...' + self.ref_spec_path[-50:]
        else:
            ref_spec_abbr = self.ref_spec_path
        self.ax_SO2.set_title('Reference Spectrum: %s' % ref_spec_abbr)
        self.refCanvas.draw()

    def conv_ref(self):
        if self.doas_worker.ILS is None:
            print('No ILS to convolved reference spectrum with')
            return

        # If reference spectrum dictionary is empty - return
        if not self.doas_worker.ref_spec:
            print('No reference spectrum loaded')
            return

        # Convolve reference spectrum
        self.doas_worker.conv_ref_spec()

        # Update plot (only if convolution was successful - it fails if wavelength data is not present - need to acquire spectrum with spectrometer first)
        if self.species in self.doas_worker.ref_spec_conv:
            self.ax_SO2.lines[1].set_data(self.doas_worker.wavelengths, self.doas_worker.ref_spec_conv[self.species])
            self.refCanvas.draw()

