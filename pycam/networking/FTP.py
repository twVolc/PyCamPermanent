# -*- coding: utf-8 -*-

"""Contains FTP classes for controlling transfer of images and spectra from remote camera to local processing machine"""

from pycam.utils import read_file
from pycam.setupclasses import CameraSpecs, SpecSpecs
import ftplib
import os
import time
import queue
import threading


# THIS CLASS IS UNDER CONSTRUCTION AND REQUIRES MUCH MORE WORK AT THIS POINT. ONLY THE SKELETON HAS BEEN MADE SO FAR!
class FTPClient:
    """
    Main class for controlling FTP transfer

    :param network_info:    dict    Contains network parameters defining information for FTP transfer
    """

    def __init__(self, network_info=None):
        self.refresh_time = 1   # Length of time directory watcher sleeps before listing server images again
        self.cam_specs = CameraSpecs()
        self.spec_specs = SpecSpecs()
        self.watch_q = queue.Queue()
        self.thread = None
        self.watching_dir = False

        # If we are given a filename on instantiation we need to read in the data to class
        if network_info is not None:
            self.config = network_info

            # Do unpacking of config dictionary here
            self.host_ip = self.config['host_ip']
            self.user = self.config['uname']
            self.pwd = self.config['pwd']
            self.dir_data_remote = self.config['data_dir']
            self.local_dir = self.config['local_dir']
            self.dir_img_local = os.path.join(self.local_dir, 'Images/')
            if not os.path.exists(self.dir_img_local):
                os.mkdir(self.dir_img_local)
            self.dir_spec_local = os.path.join(self.local_dir, 'Spectra/')
            if not os.path.exists(self.dir_spec_local):
                os.mkdir(self.dir_spec_local)
        else:
            self._default_specs()

        self.connection = None  # Attribute should contain FTP connection

        # Open connection if we have a host_ip
        if len(self.host_ip) > 0:
            self.open_connection(self.host_ip, self.user, self.pwd)

    def _default_specs(self):
        """Load in default specs for FTP transfer"""
        self.host_ip = ''
        self.user = ''
        self.pwd = ''
        self.dir_data_remote = ''       # Spectra and Images are in a single directory for transferring
        self.dir_img_local = ''
        self.dir_spec_local = ''

    def open_connection(self, ip, username=None, password=None):
        """Opens FTP connection to host machine and moves to correct working directory"""
        try:
            self.connection = ftplib.FTP(ip)
            self.connection.login(user=username, passwd=password)
            self.connection.cwd(self.dir_data_remote)
            print('Got FTP connection from {}. File transfer now available.'.format(ip))
        except ftplib.all_errors as e:
            print('FTP connection encountered error - file transfer is inactive')
            print(e)

    def close_connection(self):
        """Closes FTP connection"""
        self.connection.close()

    def get_file(self, img, local_dir, rm=True):
        """Downloads image
        :param img:         str     Filename of image on remote machine
        :param local_dir:   str     Local directory to save image. Image will keep the same filename
        :param rm:          bool    If True, the file is deleted from the host once it has been transferred
        """
        # Filename organisation for local machine
        # File always goes into dated directory (date is extracted from filename)
        filename = os.path.split(img)[-1]
        date = filename.split('_')[self.cam_specs.file_date_loc].split('T')[0]
        local_date_dir = os.path.join(local_dir, date)
        if not os.path.exists(local_date_dir):
            os.mkdir(local_date_dir)
        local_name = os.path.join(local_date_dir, filename)

        # Download file
        with open(local_name, 'wb') as f:
            start_time = time.time()
            self.connection.retrbinary('RETR ' + img, f.write)
            elapsed_time = time.time() - start_time
            print('Transferred file {} from instrument to {}. Transfer time: {:.4f}s'.format(filename, local_date_dir,
                                                                                             elapsed_time))

        # Delete file after it has been transferred
        if rm:
            try:
                self.connection.delete(img)
                print('Deleted file {} from instrument'.format(filename))
            except ftplib.error_perm as e:
                print(e)

    def watch_dir(self):
        """Public access thread starter for _watch_dir"""
        self.watch_thread = threading.Thread(target=self._watch_dir, args=())
        self.watch_thread.daemon = True
        self.watch_thread.start()
        self.watching_dir = True

    def _watch_dir(self, lock='.lock'):
        """
        Watches directory for new files and adds image to queue when they appear. Watches the directory defined by
        self.dir_data_remote.

        :param lock: str
            Filename extension which defines when the file is not ready to be collected (in a locked state)
        """
        while True:
            # First check we haven't had a stop request
            try:
                mess = self.watch_q.get(block=False)
                if mess:
                    self.watching_dir = False
                    return
            except queue.Empty:
                pass

            # Get file list from host machine
            try:
                file_list = self.connection.nlst()
            except ftplib.error_perm as e:
                print(e)
                continue

            # Loop through files and decide what to do with them
            for file in file_list:
                # Always check if a quit has been requested - this ensures we don't get stuck here if transferring a lot
                # of files
                try:
                    mess = self.watch_q.get(block=False)
                    if mess:
                        self.watching_dir = False
                        return
                except queue.Empty:
                    pass

                filename, ext = os.path.splitext(file)

                # Camera image
                if ext == self.cam_specs.file_ext:
                    lock_file = filename + lock
                    if lock_file in file_list:      # Don't download image if it is still locked
                        continue
                    else:
                        self.get_file(os.path.join(self.dir_data_remote, file), self.dir_img_local)

                # Spectrum
                if ext == self.spec_specs.file_ext:
                    lock_file = filename + lock
                    if lock_file in file_list:      # Don't download image if it is still locked
                        continue
                    else:
                        self.get_file(os.path.join(self.dir_data_remote, file), self.dir_spec_local)

            # Sleep for designated time, so I'm not constantly polling the host
            time.sleep(self.refresh_time)

    def stop_watch(self):
        """Stops FTP file transfer"""
        if self.watching_dir:
            self.watch_q.put(1)







