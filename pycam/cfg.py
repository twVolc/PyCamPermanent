# -*- coding: utf-8 -*-

"""Main file for holding Pyplis processing object"""

from pycam.so2_camera_processor import PyplisWorker
from pycam.setupclasses import FileLocator
from pathlib import Path

process_defaults_loc = Path(FileLocator.PROCESS_DEFAULTS_LOC)
if process_defaults_loc.exists():
    process_settings_path = process_defaults_loc.read_text()
else:
    process_settings_path = FileLocator.PROCESS_DEFAULTS

# Instantiate main pyplis worker
pyplis_worker = PyplisWorker(config_path=process_settings_path)
