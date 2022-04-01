# -*- coding: utf-8 -*-

"""
Script to check disk space taken up by Pi images and spectra. If it exceeds a predfined threshold the oldest images are
deleted
"""
import subprocess
import os
import sys
sys.path.append('/home/pi/')

from pycam.setupclasses import FileLocator


# Threshold (in MB)
threshold = 100000   # 100 GB
threshold_kb = threshold * 1000

# Path to imag directory
img_path = FileLocator.IMG_SPEC_PATH


def get_storage_usage(img_path):
    # TODO change from du to df, as du only gives file size, it doesn't quote total disk space - which is the important
    # TODO thing
    proc = subprocess.Popen(['du -s {}'.format(img_path)], stdout=subprocess.PIPE, shell=True)
    stdout_value = proc.communicate()[0]
    stdout_str = stdout_value.decode("utf-8")
    stdout_lines = stdout_str.split('\n')

    # Get kB storage by splitting the output string
    storage_kb = int(stdout_lines[0].split()[0])
    return storage_kb


# Get storage
storage_kb = get_storage_usage(img_path)
print('Disk usage: {}'.format(storage_kb))
print('Disk threshold: {}'.format(threshold_kb))

while storage_kb > threshold_kb:
    file_list = os.listdir(img_path)
    file_list.sort()

    # Catch exception just in case the file disappears before it can be removed
    # (may get transferred then deleted by other program)
    try:
        file_path = os.path.join(img_path, file_list[0])
        os.remove(file_path)
        with open(FileLocator.REMOVED_FILES_LOG_PI, 'a') as f:
            f.write('Data exceeding threshold disk space: {}MB. Removing file: {}\n'.format(threshold, file_path))
    except BaseException as e:
        with open(FileLocator.REMOVED_FILES_LOG_PI, 'a') as f:
            f.write('{}\n'.format(e))

    # Update storage
    storage_kb = get_storage_usage(img_path)
