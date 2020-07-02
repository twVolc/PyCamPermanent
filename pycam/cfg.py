# -*- coding: utf-8 -*-

"""Main file for holding Pyplis processing object"""

from pycam.so2_camera_processor import PyplisWorker
from pycam.utils import read_file
from pycam.setupclasses import FileLocator

process_settings = read_file(FileLocator.PROCESS_DEFAULTS)

init_dir = process_settings['init_dir'].split('\'')[1]
bg_img_A = {'bg_path': process_settings['bg_img_A'].split('\'')[1],
            'dark_path': process_settings['bg_dark_A'].split('\'')[1]}
bg_img_B = {'bg_path': process_settings['bg_img_B'].split('\'')[1],
            'dark_path': process_settings['bg_dark_B'].split('\'')[1]}

# Instantiate main pyplis worker
pyplis_worker = PyplisWorker(img_dir=init_dir, bg_img_A=bg_img_A, bg_img_B=bg_img_B)
