# -*- coding: utf-8 -*-

"""
Script to check disk space taken up by Pi images and spectra. If it exceeds a predfined threshold the oldest images are
deleted
"""
import subprocess
import os
import sys
import datetime
sys.path.append('/home/pi/')

from pycam.setupclasses import FileLocator

# Path to image directory
img_path = FileLocator.IMG_SPEC_PATH

del_days = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # Days on which all existing data is deleted on local microSD
date_now = datetime.datetime.now()
day = date_now.day

if day in del_days:
    file_list = os.listdir(img_path)
    file_list.sort()
    for filename in file_list:
        if len(file_list) < 5000:
            break
        # Catch exception just in case the file disappears before it can be removed
        # (may get transferred then deleted by other program)
        try:
            # If it is a lock file we just ignore it
            if '.lock' in filename:
                continue

            # Create full path
            file_path = os.path.join(img_path, filename)

            # Check file isn't locked, if it is we just leave it
            _, ext = os.path.splitext(file_path)
            pathname_lock = file_path.replace(ext, '.lock')
            if os.path.exists(pathname_lock):
                continue

            # Remove file
            os.remove(file_path)
            print('Deleting file: {}'.format(filename))
        except BaseException as e:
            with open(FileLocator.REMOVED_FILES_LOG_PI, 'a') as f:
                f.write('{}\n'.format(e))
