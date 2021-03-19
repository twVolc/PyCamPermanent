import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import queue
import datetime

from pycam.doas.doas_worker import DOASWorker
from pycam.doas.ifit_worker import IFitWorker
from pycam.doas.cfg import doas_worker, species
from pycam.gui.cfg import gui_setts
# from acquisition_gui import AcquisitionFrame
from pycam.cfg import pyplis_worker, process_settings
from pycam.setupclasses import SpecSpecs
from pycam.gui.settings import GUISettings

# plt.style.use('dark_background')
plt.style.use('default')

refresh_rate = 200


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
    def __init__(self, gui, root, frame, figsize=(10, 3), species='SO2'):
        self.gui = gui
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

        # Shift widgets
        label = tk.Label(self.frame2, text='Shift spectrum:').pack(side=tk.LEFT)
        # label.grid(row=0, column=0)
        self.shift = tk.IntVar()
        self.shift.set(self.doas_worker.shift)
        self.shift_box = ttk.Spinbox(self.frame2, from_=-20, to=20, increment=1, width=3,
                                             textvariable=self.shift, command=self.update_shift)
        # self.fit_wind_box_start.grid(row=0, column=1)
        self.shift_box.pack(side=tk.LEFT)

        # Shift tolerance widgets
        label = ttk.Label(self.frame2, text='Shift tolerance').pack(side=tk.LEFT)
        self._shift_tol = tk.IntVar()
        self.shift_tol = self.doas_worker.shift_tol
        self.shift_tol_box = ttk.Spinbox(self.frame2, from_=-20, to=20, increment=1, width=3,
                                     textvariable=self._shift_tol, command=self.update_shift_tol)
        self.shift_tol_box.pack(side=tk.LEFT)

        label2 = tk.Label(self.frame2, text='Stretch spectrum:').pack(side=tk.LEFT)
        # label2.grid(row=0, column=2)
        self.stretch = tk.IntVar()
        self.stretch.set(self.doas_worker.stretch)
        self.stretch_box = ttk.Spinbox(self.frame2, from_=-999, to=999, increment=1, width=4,
                                           textvariable=self.stretch, command=self.update_stretch)
        # self.fit_wind_box_end.grid(row=0, column=3)
        self.stretch_box.pack(side=tk.LEFT)

        # If we are working with ifit we don't have these options - it does it automatically
        if isinstance(self.doas_worker, IFitWorker):
            self.shift_box.configure(state=tk.DISABLED)
            self.shift_tol_box.configure(state=tk.DISABLED)
            self.stretch_box.configure(state=tk.DISABLED)

        # Save button
        self.save_butt = ttk.Button(self.frame2, text='Save spectra', command=self.save_spectra)
        self.save_butt.pack(side=tk.RIGHT, anchor='e')

        # ----------------------------------------------------------------
        # Tabbed figure setup for all species and residual
        # ----------------------------------------------------------------
        # Setup up tab wideget for each species
        style = self.gui.style
        style.configure('One.TNotebook.Tab', **self.gui.layout_old[0][1])
        self.tabs = ttk.Notebook(self.frame, style='One.TNotebook.Tab')
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

    @property
    def shift_tol(self):
        return self._shift_tol.get()

    @shift_tol.setter
    def shift_tol(self, value):
        self._shift_tol.set(value)
        self.doas_worker.shift_tol = value

    def update_shift_tol(self):
        """Updates DOASWorker shift value for aligning spectra"""
        # Set shift in DOASWorker object
        self.doas_worker.shift_tol = self.shift_tol

        # If we have a processed spectrum we now must update it
        if self.doas_worker.processed_data:
            self.doas_worker.process_doas()
            self.update_plot()

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
    def __init__(self, frame, doas_work, spec, figsize, dpi):
        self.frame = frame
        self.doas_worker = doas_work
        self.species = spec
        self.figsize = figsize
        self.dpi = dpi
        # ------------------------------------------------
        # FIGURE SETUP
        # ------------------------------------------------
        self.fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)

        self.ax = self.fig.subplots(1, 1)
        if isinstance(self.doas_worker, DOASWorker):
            self.ax.set_ylabel('Absorbance')
        else:
            if self.species == 'Total':
                self.ax.set_ylabel('Intensity')
            else:
                self.ax.set_ylabel('Optical depth')
        self.ax.set_ylim([-0.2, 0.2])
        self.ax.set_xlim([self.doas_worker.start_fit_wave, self.doas_worker.end_fit_wave])
        self.ax.set_xlabel('Wavelength [nm]')
        self.ax.grid(True)
        self.plt_colours = ['b', 'r']
        for i in range(2):
            self.ax.plot([250, 400], [0, 0], self.plt_colours[i], linewidth=1)
        if isinstance(self.doas_worker, IFitWorker) and self.species == 'Total':
            leg_loc = 2
        else:
            leg_loc = 1
        self.ax.legend(('Measured', 'Fitted reference'), loc=leg_loc, framealpha=1)
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

        # ylims depends on if we have filtered absorbance (DOAS) or ifit (shouldn't have OD < 0)
        try:    # If we are residual we have no ref_spec_fit key
            all_dat = [self.doas_worker.ref_spec_fit[self.species], self.doas_worker.abs_spec_species[self.species]]
        except KeyError:
            all_dat = self.doas_worker.abs_spec_species[self.species]

        if isinstance(self.doas_worker, DOASWorker):
            ylim = np.amax(np.absolute(all_dat))
            ylim *= 1.15
            if ylim == 0:
                ylim = 0.05
            ylims = [-ylim, ylim]
        elif isinstance(self.doas_worker, IFitWorker):
            min_val = np.amin(all_dat)
            if min_val >= 0:
                min_lim = min_val * 0.95
            else:
                min_lim = min_val * 1.05
            ylims = [min_lim, np.amax(all_dat) * 1.05]
        self.ax.set_ylim(ylims)

        if isinstance(self.doas_worker, DOASWorker):
            self.ax.set_title('SO2 Column density [ppm.m]: {}          STD Error: {}'.format(
                self.doas_worker.column_density['SO2'], self.doas_worker.std_err))
        elif isinstance(self.doas_worker, IFitWorker):
            if self.species != 'Total' and self.species != 'residual':
                self.ax.set_title('{} Column density [cm-2]: {:.2e}          STD Error: {:.2e}'.format(self.species,
                    self.doas_worker.column_density[self.species], self.doas_worker.fit_errs[self.species]))
            else:
                self.ax.set_title(self.species)


class CDSeries:
    """
    Generates time series of SO2 columns densities from spectrometer
    """
    def __init__(self):
        pass


class CalibrationWindow:
    """
    Generates top-level window containing calibration figures for DOAS instruments.
    :param  fig_setts: dict     Dictionary containing settings values for figures
    """
    def __init__(self, fig_setts=GUISettings()):
        # Setup reference spectra objects
        self.ref_frame = dict()
        for spec in species:
            species_id = 'ref_spec_{}'.format(spec)
            self.ref_frame[spec] = RefPlot(ref_spec_path=species[spec]['path'], species=spec, doas_work=doas_worker,
                                           fig_setts=fig_setts)

        self.ils_frame = ILSFrame(doas_work=doas_worker, fig_setts=fig_setts)

    def generate_frame(self):
        self.frame = tk.Toplevel()
        self.frame.title('DOAS calibration')
        self.frame.geometry('{}x{}+{}+{}'.format(int(self.frame.winfo_screenwidth() / 1.2),
                                                 int(self.frame.winfo_screenheight() / 1.2),
                                                 int(self.frame.winfo_screenwidth() / 10),
                                                 int(self.frame.winfo_screenheight() / 10)))

        # Reference spectra
        self.ref_spec_frame = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=4)
        self.ref_spec_frame.pack(side=tk.LEFT, anchor='nw')
        # TODO turn this into a notepad rather than packing frame by frame
        for frame in self.ref_frame:
            self.ref_frame[frame].__setup_gui__(self.ref_spec_frame)
            self.ref_frame[frame].frame.pack(side=tk.TOP, anchor='nw', fill=tk.BOTH, padx=5, pady=5)

        # ILS frame
        self.ils_frame.generate_frame(self.frame)
        self.ils_frame.frame.pack(side=tk.LEFT, anchor='nw')


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
    def __init__(self, frame=None, doas_work=DOASWorker(), init_dir='.\\', ref_spec_path=None, species='SO2',
                 fig_setts=GUISettings()):
        self.frame = frame
        self.doas_worker = doas_work  # DOAS processor
        self.species = species          # Gas species
        self.init_dir = init_dir        # Inital directory for loading dialog box
        self.fig_size = fig_setts.fig_ref_spec    # Figure size (tuple)
        self.dpi = fig_setts.dpi                  # DPI of figure

        self.setts = fig_setts
        self.pdx = 5
        self.pdy = 5
        self.max_str_len = 70

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
        self.loadRefFrame.grid(row=0, column=0, sticky='ew')
        label = ttk.Label(self.loadRefFrame, text='Filename:')
        label.grid(row=0, column=0, padx=self.pdx, pady=self.pdy)
        self.nameRef = ttk.Label(self.loadRefFrame, text=self.ref_spec_path_short)
        self.nameRef.grid(row=0, column=1, padx=self.pdx, pady=self.pdy)
        self.selectRef = ttk.Button(self.loadRefFrame, text='Load Spectrum', command=self.choose_ref_spec)
        self.selectRef.grid(row=0, column=2, sticky='e', padx=self.pdx, pady=self.pdy)

        self.conv_button = ttk.Button(self.loadRefFrame, text='Convolve with ILS', command=self.conv_ref)
        self.conv_button.grid(row=0, column=3, sticky='e')

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

    @property
    def ref_spec_path_short(self):
        try:
            return_str = '...' + self.ref_spec_path[-(self.max_str_len-3):]
        except IndexError:
            return_str = self.ref_spec_path
        return return_str

    def choose_ref_spec(self):
        """Load reference spectrum"""
        self.ref_spec_path = filedialog.askopenfilename(initialdir=self.init_dir,
                                                        title='Select reference spectrum',
                                                        filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        self.load_ref_spec()

    def load_ref_spec(self, init_load=False):
        """
        Loads reference spectrum
        :param init_load    bool    Flag for when this function is used for first time, requiring different actions
        """
        if not self.ref_spec_path:
            return
        if not init_load:
            self.nameRef.configure(self.ref_spec_path_short)
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


class ILSFrame:
    """
    Frame containing widgets for ILS extraction from a calibration spectrum
    """
    def __init__(self, parent=None, doas_work=DOASWorker(), spec_specs=SpecSpecs(), fig_setts=GUISettings(),
                 config=process_settings, save_path='C:\\'):
        # Setup some main variables
        self.parent = parent
        self.frame = None
        self.doas_worker = doas_work
        self.spec_specs = spec_specs
        self.save_path = save_path
        if 'ILS_path' in config.keys():
            self.ILS_path = config['ILS_path'].split('\'')[1]
        else:
            self.ILS_path = None

        # Unpack figure settings
        self.dpi = fig_setts.dpi
        self.cal_figsize = fig_setts.fig_cal_doas
        self.ILS_figsize = fig_setts.fig_ILS

        # Initiating some values
        self.start_int_time = 100              # Integration time GUI starts with (ms)
        self.max_DN = self.spec_specs._max_DN  # Max DN of spectrometer
        self.str_len_max = 25                  # Maximum string length allowed for filenames before they are abbreviated
        self.cal_spec_corr = None

        # If provided a parent frame on startup, we generate the frame
        if isinstance(parent, tk.Frame) or isinstance(parent, ttk.Frame):
            self.generate_frame(self.parent)
        else:
            self.in_frame = False
            # If we have a path to ILS data on startup, we load it

        if self.ILS_path is not None:
            self.load_ILS()

    def generate_frame(self, parent):
        """
        Creates widget
        :param parent:  tk.Frame    Parent frame to place widget in
        :return:
        """
        self.parent = parent
        self.frame = ttk.Frame(self.parent)
        self.in_frame = True

        self.frame_cal = ttk.Frame(self.frame, relief=tk.RAISED, borderwidth=4)
        self.frame_cal.pack(side=tk.TOP)
        self.frame_ILS = ttk.Frame(self.frame_cal, relief=tk.RAISED, borderwidth=4)
        self.frame_ILS.pack(side=tk.BOTTOM, anchor='n')

        # -----------------------------------------------
        # WIDGET SETUP
        # ------------------------------------------------
        self.frame2 = ttk.Frame(self.frame_cal)
        self.frame2.pack(side=tk.TOP, fill=tk.X, expand=1)

        # Dark and calibration spectrum acquisitions
        self.frame_acq = ttk.Frame(self.frame2, relief=tk.GROOVE, borderwidth=5)
        self.frame_acq.pack(side=tk.TOP, anchor='w')

        # Acquisition settings
        label = ttk.Label(self.frame_acq, text='Integration time (ms):').grid(row=0, column=0, sticky='w')
        self.int_time = tk.DoubleVar()  # Integration time holder
        self.int_time.set(self.start_int_time)
        self.int_entry = ttk.Entry(self.frame_acq, textvariable=self.int_time, width=5).grid(row=0, column=1,
                                                                                             sticky='w')

        label = ttk.Label(self.frame_acq, text='Dark spectrum file:').grid(row=1, column=0, sticky='w')
        self.dark_file_label = ttk.Label(self.frame_acq, text='N/A', width=25)
        self.dark_file_label.grid(row=1, column=1, sticky='w')
        self.dark_button = ttk.Button(self.frame_acq, text='Dark Capture',
                                      command=self.dark_capture).grid(row=1, column=2, sticky='nsew')
        label = ttk.Label(self.frame_acq, text='Calibration spectrum file:').grid(row=2, column=0, sticky='w')
        self.cal_file_label = ttk.Label(self.frame_acq, text='N/A', width=25)
        self.cal_file_label.grid(row=2, column=1, sticky='w')
        self.cal_button = ttk.Button(self.frame_acq, text='Calibration Capture',
                                     command=self.cal_capture).grid(row=2, column=2, sticky='nsew')

        # ILS extraction
        self.frame_lines = ttk.Frame(self.frame_cal, relief=tk.GROOVE, borderwidth=5)
        self.frame_lines.pack(side=tk.TOP, anchor='w')

        self.ILS_start = tk.DoubleVar()
        self.ILS_start.set(self.doas_worker.start_fit_wave)
        self.ILS_box_start = tk.Spinbox(self.frame_lines, from_=0, to=400, increment=0.1, width=5,
                                        textvariable=self.ILS_start, command=self.update_ILS_start)
        self.ILS_end = tk.DoubleVar()
        self.ILS_end.set(self.doas_worker.end_fit_wave)
        self.ILS_box_end = tk.Spinbox(self.frame_lines, from_=1, to=400, increment=0.1, width=5,
                                      textvariable=self.ILS_end, command=self.update_ILS_end)

        label = tk.Label(self.frame_lines, text='Emission line start:').pack(side=tk.LEFT)
        self.ILS_box_start.pack(side=tk.LEFT)
        label = tk.Label(self.frame_lines, text='Emission line end:').pack(side=tk.LEFT)
        self.ILS_box_end.pack(side=tk.LEFT)

        # ------------------------------------------------
        # FIGURE SETUP
        # ------------------------------------------------
        self.fig = plt.Figure(figsize=self.cal_figsize, dpi=self.dpi)

        self.ax = self.fig.subplots(1, 1)
        self.ax.set_ylabel('DN')
        self.ax.set_ylim([0, self.max_DN])
        self.ax.set_xlim([250, 400])
        self.ax.set_xlabel('Wavelength [nm]')
        self.ax.grid(True)
        self.plt_colours = ['b', 'r']
        for i in range(2):
            self.ax.plot([250, 400], [0, 0], self.plt_colours[i], linewidth=1)
        self.ax.legend(('Dark', 'Calibration'), loc=1, framealpha=1)
        self.fig.tight_layout()

        # Stray light
        self.min_ILS_line = self.ax.plot([self.doas_worker.start_fit_wave, self.doas_worker.start_fit_wave],
                                         [0, self.max_DN], 'y')
        self.max_ILS_line = self.ax.plot([self.doas_worker.end_fit_wave, self.doas_worker.end_fit_wave],
                                         [0, self.max_DN], 'y')

        width = self.doas_worker.end_fit_wave - self.doas_worker.start_fit_wave
        self.ILS_range = Rectangle((self.doas_worker.start_fit_wave, 0), width=width, height=self.max_DN,
                                   fill=True, facecolor='y', alpha=0.2)
        self.ILS_range_patch = self.ax.add_patch(self.ILS_range)

        # Organise and draw canvas
        self.canv = FigureCanvasTkAgg(self.fig, master=self.frame_cal)
        self.canv.draw()
        self.canv.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --------------------------------------------------------------------------------------------------------------
        # ILS FUCTIONS AND PLOT
        # --------------------------------------------------------------------------------------------------------------
        # FUNCTIONS
        self.frame_ILS_func = ttk.Frame(self.frame_ILS, relief=tk.GROOVE, borderwidth=5)
        self.frame_ILS_func.pack(side=tk.LEFT, anchor='n')

        label1 = tk.Label(self.frame_ILS_func, text='Loaded ILS file:')
        self.ILS_load_label = tk.Label(self.frame_ILS_func, text='N/A', width=self.str_len_max)
        self.load_ILS_button = ttk.Button(self.frame_ILS_func, text='Load ILS', command=self.choose_ILS)

        label2 = tk.Label(self.frame_ILS_func, text='Saved ILS file:')
        self.ILS_save_label = tk.Label(self.frame_ILS_func, text='N/A', width=self.str_len_max)
        self.save_ILS_button = ttk.Button(self.frame_ILS_func, text='Save ILS', command=self.save_ILS)

        label1.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.ILS_load_label.grid(row=0, column=1, padx=5, pady=5)
        self.load_ILS_button.grid(row=0, column=2, padx=5, pady=5)

        label2.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.ILS_save_label.grid(row=1, column=1, padx=5, pady=5)
        self.save_ILS_button.grid(row=1, column=2, padx=5, pady=5)

        # PLOT
        self.fig_ILS = plt.Figure(figsize=self.ILS_figsize, dpi=self.dpi)
        self.ax_ILS = self.fig_ILS.subplots(1, 1)
        self.ax_ILS.set_ylabel('Normalised intensity')
        self.ax_ILS.set_xlabel('Relative wavelength [nm]')
        self.ax_ILS.set_ylim([0, 1])
        self.ax_ILS.set_xlim([0, self.doas_worker.end_fit_wave - self.doas_worker.start_fit_wave])
        self.ax_ILS.grid(True)
        self.ax_ILS.plot([0, self.doas_worker.end_fit_wave - self.doas_worker.start_fit_wave], [0, 0], 'g', linewidth=1)
        self.ax_ILS.legend(('Instrument line shape',), loc=1, framealpha=1)
        self.fig_ILS.tight_layout()

        # Organise and draw canvas
        self.canv_ILS = FigureCanvasTkAgg(self.fig_ILS, master=self.frame_ILS)
        self.canv_ILS.draw()
        self.canv_ILS.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Update label and plot as the ILS should be preloaded on startup
        self.update_ILS_label()
        self.update_ILS_plot()

    def dark_capture(self):
        """Implements dark capture fom spectrometer"""
        pass

    def cal_capture(self):
        """Implements calibration lamp capture fom spectrometer"""
        pass

    def update_ILS_start(self):
        """Updates location of ILS extraction line, and performs extraction of ILS if calibration spectrum is present"""
        ILS_start = self.ILS_start.get()

        # Ensure the start of the stray range doesn't become less than the start
        if ILS_start >= self.ILS_end.get():
            ILS_start = self.ILS_end.get() - 0.1
            self.ILS_start.set(ILS_start)

        # Update plot
        self.min_ILS_line[0].set_data([ILS_start, ILS_start],
                                      [0, self.max_DN])
        self.ILS_range.set_x(ILS_start)
        self.ILS_range.set_width(self.ILS_end.get() - ILS_start)
        self.canv.draw()

        if self.cal_spec_corr is not None:
            self.extract_ILS()

    def update_ILS_end(self):
        """Updates location of ILS extraction line, and performs extraction of ILS if calibration spectrum is present"""
        ILS_end = self.ILS_end.get()

        # Ensure the end of the stray range doesn't become less than the start
        if ILS_end <= self.ILS_start.get():
            ILS_end = self.ILS_start.get() + 0.1
            self.ILS_end.set(ILS_end)

        # Update plot
        self.max_ILS_line[0].set_data([ILS_end, ILS_end],
                                        [0, self.max_DN])
        self.ILS_range.set_width(ILS_end - self.ILS_start.get())
        self.canv.draw()

        if self.cal_spec_corr is not None:
            self.extract_ILS()

    def choose_ILS(self):
        """Bring up filedialog to select ILS, then instigates loading"""
        # Bring up dialog to find file
        self.ILS_path = filedialog.askopenfilename(initialdir=self.save_path,
                                                   title='Select ILS file',
                                                   filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        self.load_ILS()

    def update_ILS_label(self):
        """Updates ILS label"""
        # Update label in widget
        ILS_filename = self.ILS_path.split('/')[-1].split('\\')[-1]
        if len(ILS_filename) > self.str_len_max:
            self.ILS_load_label.configure(text='...' + ILS_filename[-(self.str_len_max - 3):])
        else:
            self.ILS_load_label.configure(text=ILS_filename)

    def load_ILS(self):
        """Loads ILS from text file"""
        if not self.ILS_path:
            return

        # Extract save path for next filedialog
        self.save_path = self.ILS_path.rsplit('\\', 1)[0].rsplit('/', 1)[0]

        # Extract data
        self.doas_worker.load_ils(self.ILS_path)

        # Only update frame if we have a frame setup
        if self.in_frame:
            # Update ILS label
            self.update_ILS_label()

            # Update ILS plot
            self.update_ILS_plot()

    def save_ILS(self):
        """Saves ILS to text file"""
        if self.doas_worker.ILS is None or self.doas_worker.ILS_wavelengths is None:
            print('No instrument line shape data to save')
            return

        time = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        ILS_filename = '{}_ILS.txt'.format(time)

        # Save file to numpy array
        np.savetxt(self.save_path + ILS_filename,
                   np.transpose([self.doas_worker.ILS_wavelengths, self.doas_worker.ILS]),
                   header='Instrument line shape\n'
                          'Line start [nm]: {}\tLine End [nm]: {}\n'
                          'Wavelength [nm]\tIntensity [DN]'.format(self.ILS_start.get(), self.ILS_end.get()))

        # Update widget label
        self.ILS_save_label.configure(text=ILS_filename)

    def extract_ILS(self):
        """Extracts instrument line shape from the calibration spectrum, then draws it"""
        # Determine indices for extraction of ILS
        start_idx = np.argmin(np.absolute(self.doas_worker.wavelengths - self.ILS_start.get()))
        end_idx = np.argmin(np.absolute(self.doas_worker.wavelengths - self.ILS_end.get())) + 1

        # Extract ILS and normalise it
        # Need to use copy() here, otherwise we end up modifying cal_spec_corr when modifying ILS
        ILS = np.copy(self.cal_spec_corr[start_idx:end_idx])
        self.doas_worker.ILS /= np.amax(ILS)

        # Extract wavelengths then set to start at 0 nm
        self.doas_worker.ILS_wavelengths = self.doas_worker.wavelengths[start_idx:end_idx] - self.doas_worker.wavelengths[start_idx]

        # Need to update the ifit analyser if we are using this
        if isinstance(self.doas_worker, IFitWorker):
            self.doas_worker.update_ils()

        # Update ILS plot
        self.update_ILS_plot()

    def update_ILS_plot(self):
        """Update ILS plot"""
        # Update plot
        self.ax_ILS.lines[0].set_data(self.doas_worker.ILS_wavelengths, self.doas_worker.ILS)
        self.ax_ILS.set_xlim([0, self.doas_worker.ILS_wavelengths[-1]])
        self.canv_ILS.draw()