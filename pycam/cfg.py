# -*- coding: utf-8 -*-

"""Main file for holding Pyplis processing object"""

from pycam.so2_camera_processor import PyplisWorker
from pycam.setupclasses import FileLocator

default_process_settings = FileLocator.PROCESS_DEFAULTS

# Instantiate main pyplis worker
pyplis_worker = PyplisWorker(config_path=default_process_settings)
