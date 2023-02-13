# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator
from pycam.scripts.clouduploaders.dropbox_io import DropboxIO
import time

# Endlessly loop around - if we ever catch an exception we just delete the dropbox uploader and create a new one
# This should deal with connection errors
while True:
    try:
        if 'dbx' not in locals():
            # Create dropbox object
            dbx = DropboxIO(watch_folder=FileLocator.IMG_SPEC_PATH, delete_after=True, recursive=True)
            # dbx = DropboxIO(watch_folder='C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\', delete_after=False)

            # Upload any existing files
            dbx.upload_existing_files()
c
            # Start directory watcher
            dbx.watcher.start()
        else:
            print('Uploader waiting...')
            time.sleep(0.5)
    except Exception:
        print('Deleting dropbox object')
        dbx.watcher.stop()
        del dbx





