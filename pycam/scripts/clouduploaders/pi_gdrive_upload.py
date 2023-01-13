# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator
from pycam.scripts.clouduploaders.gdrive_uploader import GoogleDriveUploader

gdrive = GoogleDriveUploader(watch_folder=FileLocator.IMG_SPEC_PATH)

