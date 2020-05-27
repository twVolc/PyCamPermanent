# -*- codingL utf-8 -*-

"""Holds all classes for generating matplotlib figures for the pycam GUI"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.spines


class ImageFigure:
    """
    Class for plotting an image and associated widgets, such as cross-sectinal DNs
    """
    def __init(self, frame):
        self.frame = frame

        # --------------------------------------------------------------------------------------------------------------
        # Setup figure
        # --------------------------------------------------------------------------------------------------------------
        self.fig = plt.Figure(figsize=self.img_fig_size, dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, width_ratios=[648, 200], height_ratios=[486, 200])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.set_aspect(1)
        self.pltColA = self.fig.add_subplot(gs[1], sharey=self.ax)
        self.pltRowA = self.fig.add_subplot(gs[2], sharex=self.ax)
        # self.ax.set_adjustable('box-forced')
        # self.pltColA_test.set_adjustable('box-forced')
        # self.pltRowA_test.set_adjustable('box-forced')

        plt.setp(self.pltColA.get_yticklabels(), visible=False)
        plt.setp(self.ax.get_xticklabels(), visible=False)
        # self.pltColA.set_xticklabels(self.pltColA.xaxis.get_majorticklabels(), rotation=180)
        # self.fig.subplots_adjust(hspace=0.1, wspace=486/6480)

        self.fig.set_facecolor('black')

        for child in self.ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.ax.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        self.imgDispA = self.ax.imshow(self.imgArrayA, cmap=cm.Greys_r, interpolation='none', vmin=0,
                                       vmax=self.maxDN, aspect='auto')  # FOR GREYSCALE
        self.imgDispRowA, = self.ax.plot([0, self.imgSizeX], [self._init_imgRow, self._init_imgRow],
                                         color=self._row_colour, lw=2)
        self.imgDispColA, = self.ax.plot([self._init_imgCol, self._init_imgCol], [0, self.imgSizeY],
                                         color=self._col_colour, lw=2)
        self.ax.set_xlim([0, self.imgSizeX])
        self.ax.set_ylim([self.imgSizeY, 0])
        self.ax.set_title('Test Image A', color='white')
        self.ax.set_ylabel('Pixel', color='white')

        for child in self.pltRowA.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.pltRowA.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        for child in self.pltColA.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')
        self.pltColA.tick_params(axis='both', colors='white', direction='in', top='on', right='on')
        self.pltRowA.set_facecolor('black')
        self.pltColA.set_facecolor('black')

        self.pixRow = np.arange(0, self.imgSizeX, 1)
        self.pixCol = np.arange(0, self.imgSizeY, 1)
        self.rowDN_A = self.imgArrayA[self._init_imgRow, :]
        self.colDN_A = self.imgArrayA[:, self._init_imgCol]

        self.lineRow_A, = self.pltRowA.plot(self.pixRow, self.rowDN_A, color=self._row_colour)
        self.lineCol_A, = self.pltColA.plot(self.colDN_A, self.pixCol, color=self._col_colour)

        # ------------------------------------------------------
        # Plot settings
        # ------------------------------------------------------
        self.pltRowA.set_xlabel('Pixel', color='white')
        self.pltRowA.set_ylabel('DN', color='white')
        self.pltColA.set_xlabel('DN', color='white')
        # self.pltColA.set_ylabel('Pixel', color='white')
        # self.pltRowA.set_xlim(0, self.imgSizeX)
        self.pltRowA.set_ylim(0, self.maxDN)
        self.pltColA.set_xlim(0, self.maxDN)
        self.pltColA.set_ylim(self.imgSizeY, 0)

        self.fig.tight_layout()  # Make plots extend right to edges of figure (or at least to a good fit)
        self.fig.subplots_adjust(hspace=0.1, wspace=486 / 6480)  # Make space between subplots equal

        # -------------------------------------
        self.imgCanvasA = FigureCanvasTkAgg(self.fig, master=self.frameImgA)
        self.imgCanvasA.show()
        self.imgCanvasA.get_tk_widget().pack(side=tk.TOP)