# -*- coding: utf-8 -*-

"""
Script to free up space on SSD.

Note: This script does not clear all data from SSD (for that use clear_ssd.py). This script deletes enough data to
leave a predefined amount of space lef ton the SSD. Oldest files are deleted first.

Script can be passed an argument which defines the amount of space to make free on the SSD (in GB).
"""

import sys
sys.path.append('/home/pi/')

from pycam.utils import StorageMount

# Default space to create on SSD
make_space = 50

# Check if argument is passed to script for amount of space to free up
if len(sys.argv) - 1 == 1:
    make_space = int(sys.argv[-1])

# Create storage mount object
storage_mount = StorageMount()

# Ensure it is mounted
storage_mount.mount_dev()

# Free up space
storage_mount.free_up_space(make_space=make_space)
