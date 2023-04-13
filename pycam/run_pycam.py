# -*- coding: utf-8 -*-

"""Main file for running the PyCam GUI"""

import sys
import os
# # Make it possible to import iFit by updating path
dir_path = os.path.dirname(os.path.realpath(__file__))
ifit_path = os.path.join(dir_path, 'ifit')
sys.path.append(ifit_path)
from pycam.gui.pycam_gui import run_GUI

run_GUI()