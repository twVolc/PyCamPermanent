# -*- coding: utf-8 -*-

"""Simple script to unmount SSD before shutdown"""

import sys
sys.path.append('/home/pi/')

from pycam.utils import StorageMount

mount = StorageMount()
mount.unmount_dev()
