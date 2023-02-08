# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator
from pycam.scripts.clouduploaders.dropbox_io import DropboxIO

# Create dropbox object
dbx = DropboxIO(refresh_token_path='./dbx_access.txt', root_folder='Sheffield',
                save_folder='./test/', download_to_datedirs=True, delete_after=True)
# dbx = DropboxIO(watch_folder='C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\', delete_after=False)

# Download data files
dbx.downloader()

