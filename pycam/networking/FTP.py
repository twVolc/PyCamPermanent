# -*- coding: utf-8 -*-

"""Contains FTP classes for controlling transfer of images and spectra from remote camera to local processing machine"""

import ftplib
from pycam.utils import read_file


# THIS CLASS IS UNDER CONSTRUCTION AND REQUIRES MUCH MORE WORK AT THIS POINT. ONLY THE SKELETON HAS BEEN MADE SO FAR!
class FTPClient:
    """Main class for controlling FTP transfer"""

    def __init__(self, network_file=None):

        # If we are given a filename of instantiation we need to read in the data to class
        if network_file is not None:
            self.config = read_file(network_file)
            # Do unpacking of data here

        else:
            self._default_specs()

        self.connection = None  # Attribute should contain FTP connection

    def _default_specs(self):
        """Load in default specs for FTP transfer"""
        self.host_ip = ''
        self.dir_img_remote = ''
        self.dir_spec_remote = ''
        self.dir_img_local = ''
        self.dir_spec_local = ''

    def open_connection(self, ip):
        """Opens FTP connection to host machine"""
        pass

    def close_connection(self):
        """Closes FTP connection"""
        pass

    def get_img(self):
        """Downloads image"""
        pass

    def get_spec(self):
        """Downloads spectrum"""
        pass

    def watch_dir(self, path, q, lock='.lock'):
        """Watches directory for new files and adds image to queue when they appear

        Parameters
        ----------
        path: str
            path on remote machine to be watched
        q: queue.Queue
            q to but filenames in once they are ready to be transferred
        lock: str
            Filename extension which defines when the file is not ready to be collected (in a locked state)"""
        pass





