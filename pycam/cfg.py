# -*- coding: utf-8 -*-

"""Main file for holding Pyplis processing object"""

from pycam.so2_camera_processor import PyplisWorker
from pycam.utils import read_file
from pycam.setupclasses import FileLocator

process_settings = read_file(FileLocator.PROCESS_DEFAULTS)

init_dir = process_settings['init_img_dir'].split('\'')[1]
default_process_settings = "C://Users//Daniel Brady//Documents//Lascar//processing_setting.yml"

# Instantiate main pyplis worker
pyplis_worker = PyplisWorker(config_path=default_process_settings, img_dir=init_dir)
