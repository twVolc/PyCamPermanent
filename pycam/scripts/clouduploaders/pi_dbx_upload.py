# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator
from pycam.scripts.clouduploaders.dropbox_io import DropboxIO

# Create dropbox object
dbx = DropboxIO(watch_folder=FileLocator.IMG_SPEC_PATH, delete_on_upload=True, recursive=True)
# dbx = DropboxIO(watch_folder='C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\', delete_on_upload=False)

# Upload any existing files
dbx.upload_existing_files(delete_on_upload=True)

# Start directory watcher
dbx.watcher.start()


