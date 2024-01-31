# -*- coding: utf-8 -*-

"""
This file is a simple way of writing a number of folders to videos without having to manually select each folder in the
GUI. i.e. it's just a short-cut to make your life easier if you have lots of days of data.
"""

from pycam.io_py import create_video
import os

# Define directory where data directories are located
parent_directory = 'X:/volcano_cameras/Shared/Cotopaxi/'

# Define band to write
band = 'on'

data_dirs = os.listdir(parent_directory)

for data_dir in data_dirs:

    dir_path = os.path.join(parent_directory, data_dir)
    if os.path.isdir(dir_path):
        create_video(dir_path, band=band, save_dir=dir_path, overwrite=False)