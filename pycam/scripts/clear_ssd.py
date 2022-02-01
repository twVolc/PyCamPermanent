# -*- coding: utf-8 -*-

"""Script to clear all data from the SSD"""

import sys
sys.path.append('/home/pi/')

from pycam.utils import StorageMount

# Create storage mount object
storage_mount = StorageMount()

# Ensure it is mounted
storage_mount.mount_dev()

# Delete all data
storage_mount.del_all_data()
