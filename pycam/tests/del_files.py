# -*- coding: utf-8 -*-

"""
Script to delete data files whilst pycam is runnning, so that check_run.py can be tested
"""

import sys
sys.path.append('/home/pi/')
import datetime
import os
import time

from pycam.utils import StorageMount
from pycam.setupclasses import SpecSpecs, CameraSpecs
from pycam.directory_watcher import create_dir_watcher


def del_file_spec(pathname, t):
    """Deletes spectrometer file"""
    spec_specs = SpecSpecs()
    if spec_specs.file_ext not in pathname:
        return
    lock_file = os.path.splitext(pathname)[0] + '.lock'
    while os.path.exists(lock_file):
        time.sleep(0.05)
    print('Deleting file: {}'.format(pathname))
    os.remove(pathname)


def del_file_cam(pathname, t):
    """Deletes camera file"""
    cam_specs = CameraSpecs()
    if cam_specs.file_ext not in pathname:
        return
    lock_file = os.path.splitext(pathname)[0] + '.lock'
    while os.path.exists(lock_file):
        time.sleep(0.05)
    print('Deleting file: {}'.format(pathname))
    os.remove(pathname)


# Setup storage mount for where data is saved
storage_mount = StorageMount()

# Format for date directory of data
date_fmt="%Y-%m-%d"

# Get current list of images in
date_1 = datetime.datetime.now().strftime(date_fmt)
data_path = os.path.join(storage_mount.data_path, date_1)

# Create watcher
watcher = create_dir_watcher(data_path, False, del_file_spec)
watcher.start()




# while True:
#     file_list = os.listdir(data_path)
#
#     # Loop thorugh files, deleting spectra
#     for file in file_list:
#         if spec_specs.file_ext in file:
#             full_path = os.path.join(data_path, file)
#             os.remove(full_path)
#
#     time.sleep(0.2)
