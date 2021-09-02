# -*- coding: utf-8 -*-

"""Contains FTP classes for controlling transfer of images and spectra from remote camera to local processing machine"""

from pycam.utils import read_file
from pycam.setupclasses import CameraSpecs, SpecSpecs, FileLocator, ConfigInfo
import ftplib
import os
import time
import queue
import threading
import datetime
import pathlib


class CurrentDirectories:
    """
    Class holding the current active directories for file transfer
    """
    def __init__(self, root=FileLocator.IMG_SPEC_PATH_WINDOWS, specs=CameraSpecs()):
        self.root_dir = root
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        self.specs = specs

        self.date_fmt = "%Y-%m-%d"
        self.date_dir = None

        self.seq_dir = None
        self.cal_dir = None
        self.dark_dir = None    # For holding a list of dark images - calibration dirs can also hold dark images
        self.test_dir = None

        self.cal_dir_fmt = 'Cal_{}/'
        self.seq_dir_fmt = 'Seq_{}/'
        self.test_fmt = 'test_images/'
        self.dark_fmt = 'dark_images/'

        # If auto mode the object will find the most recent folder for that date and add files - if False (manual)
        # no editing of sequences is done beyond changing the date directory
        self.auto_mode = True

    def set_date_dir(self, date=None, edit_all=True):
        """
        Gets current directory based on date and root

        :param edit_all:    bool    If True, all directories will be updated if the date directory changes here.
                                    Otherwise they remain unchanged - False is used when run from inside a setting func
        """
        if date is None:
            date = datetime.datetime.now().strftime(self.date_fmt)
        date_dir_old = self.date_dir
        self.date_dir = os.path.join(self.root_dir, date)
        if not os.path.exists(self.date_dir):
            os.mkdir(self.date_dir)

        # Update subdirectories, but don't create new sequences - append to old. Only need to do this if the new
        # date_dir is actually different to the previous (i.e. if the date changes)
        if date_dir_old != self.date_dir and edit_all:
            self.set_test_dir(set_date=False)
            self.set_seq_dir(set_date=False, new=False)
            self.set_cal_dir(set_date=False, new=False)

    def set_test_dir(self, set_date=True):
        """Setup directory for test images"""
        if set_date:
            self.set_date_dir(edit_all=False)
        self.test_dir = os.path.join(self.date_dir, 'test_images/')

    def set_seq_dir(self, set_date=True, new=True):
        """Setup directory for sequence images"""
        if set_date:
            self.set_date_dir(edit_all=False)

        # If new requested we make a new sequence directory, otherwise find the most recent of that date, if one exists
        if new:
            seq_num = 1
            self.seq_dir = os.path.join(self.date_dir, self.seq_dir_fmt.format(seq_num))
            while os.path.exists(self.seq_dir):
                seq_num += 1
                self.seq_dir = os.path.join(self.date_dir, self.seq_dir_fmt.format(seq_num))
            # os.mkdir(self.seq_dir)
        else:
            id = self.seq_dir_fmt.split('{}')[0]
            seq_dirs = [x for x in os.listdir(self.date_dir) if id in x]
            if len(seq_dirs) == 0:
                self.seq_dir = os.path.join(self.date_dir, self.seq_dir_fmt.format(1))
            else:
                seq_dirs.sort()
                self.seq_dir = os.path.join(self.date_dir, seq_dirs[-1])

    def set_cal_dir(self, set_date=True, new=True):
        """Setup directory for calibration images"""
        # Change calibration directory so new images are saved to correct place
        if set_date:
            self.set_date_dir(edit_all=False)

        # If new requested we make a new sequence directory, otherwise find the most recent of that date, if one exists
        if new:
            cal_num = 1
            self.cal_dir = os.path.join(self.date_dir, self.cal_dir_fmt.format(cal_num))
            while os.path.exists(self.cal_dir):
                cal_num += 1
                self.cal_dir = os.path.join(self.date_dir, self.cal_dir_fmt.format(cal_num))
            # os.mkdir(self.cal_dir)
        else:
            id = self.cal_dir_fmt.split('{}')[0]
            cal_dirs = [x for x in os.listdir(self.date_dir) if id in x]
            if len(cal_dirs) == 0:
                self.cal_dir = os.path.join(self.date_dir, self.cal_dir_fmt.format(1))
            else:
                cal_dirs.sort()
                self.cal_dir = os.path.join(self.date_dir, cal_dirs[-1])

    def get_file_dir(self, filename):
        """Gets the directory to save the file to"""
        # Extract date if in auto mode, otherwise the directories should already be set from the manual acquisition obj
        if self.auto_mode:
            date = filename.split('_')[self.specs.file_date_loc].split('T')[0]
            self.set_date_dir(date)

        # Check image type from filename and then place it in the correct folder from that
        img_type = filename.split('_')[self.specs.file_type_loc]
        print('FTP transferring image type: {}'.format(img_type))
        if img_type == self.specs.file_type['test']:
            local_date_dir = self.test_dir
            if local_date_dir is None:
                self.set_test_dir()
        elif img_type == self.specs.file_type['meas']:
            local_date_dir = self.seq_dir
            if local_date_dir is None:
                self.set_seq_dir()
                local_date_dir = self.seq_dir
        elif img_type == self.specs.file_type['clear']:
            local_date_dir = self.cal_dir
            if local_date_dir is None:
                self.set_cal_dir()
                local_date_dir = self.cal_dir
        elif self.specs.file_type['cal'] in img_type:
            local_date_dir = self.cal_dir
            if local_date_dir is None:
                self.set_cal_dir()
                local_date_dir = self.cal_dir
        # TODO dark could be in either cal_x or could go in the dark_list dir if I do that? Maybe just have dark_dir as cal_dir?
        elif img_type == self.specs.file_type['dark']:
            # local_date_dir = self.dark_dir
            local_date_dir = self.cal_dir
            if local_date_dir is None:
                self.set_cal_dir()
                local_date_dir =self.cal_dir
        # If unrecognised file type the image is just put straight in the date directory
        else:
            if self.date_dir is None:
                self.date_dir = self.set_date_dir()
            local_date_dir = self.date_dir

        # Make directory if it doesn't already exist
        if not os.path.exists(local_date_dir):
            os.mkdir(local_date_dir)

        return local_date_dir


class FTPClient:
    """
    Main class for controlling FTP transfer

    :param img_dir:         CurrentDirectories  Object holding info on all the current directories for img data storage
    :param spec_dir:        CurrentDirectories  Object holding info on all the current directories for spec data storage
    :param network_info:    dict                Contains network parameters defining information for FTP transfer
    """

    def __init__(self, img_dir, spec_dir, network_info=None):
        # TODO Need a way of changing the IP address this connects to - this should be linked to sock ip somehow

        self.refresh_time = 1   # Length of time directory watcher sleeps before listing server images again
        self.cam_specs = CameraSpecs()
        self.spec_specs = SpecSpecs()
        self.watch_q = queue.Queue()
        self.thread = None
        self.watching_dir = False

        # Objects containing current working directories
        self.img_dir = img_dir
        self.spec_dir = spec_dir

        # If we are given a filename on instantiation we need to read in the data to class
        if network_info is not None:
            self.config = network_info

            # Do unpacking of config dictionary here
            self.host_ip = self.config['host_ip']
            self.user = self.config['uname']
            self.pwd = self.config['pwd']
            self.dir_data_remote = self.config['data_dir']
            self.local_dir = self.config[ConfigInfo.local_data_dir]
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
            return True
        except BaseException as e:
            print('FTP connection encountered error - file transfer is inactive')
            print(e)
            return False

    def close_connection(self):
        """Closes FTP connection"""
        self.connection.close()

    def test_connection(self):
        """Tests connection is still active, and if not it attempts to reconnect. If not possible, it returns False"""
        try:
            self.connection.voidcmd('NOOP')
        except BaseException as e:
            conn = self.open_connection(self.host_ip, username=self.user, password=self.pwd)
            return conn
        return True

    def move_file_to_instrument(self, local_file, remote_file):
        """Move specific file from local_file location to remote_file location"""
        if not os.path.exists(local_file):
            print('File does not exist, cannot perform FTP transfer: {}'.format(local_file))
            return

        # Test FTP connection
        if not self.test_connection():
            print('Cannot establish FTP connection. File cannot be transferred')

        # Move file to location
        with open(local_file, 'rb') as f:
            self.connection.storbinary('STOR ' + remote_file, f)

        print('FTP moved file from {} to {}'.format(local_file, remote_file))

    def get_file(self, remote, local, rm=True):
        """
        Gets file from remote location and places it in local location
        :param remote:  str     Path to remote file
        :param local:   str     Path to local location
        :param rm:      bool    If true the file is deleted from the remote computer
        :return:
        """
        # Test FTP connection
        if not self.test_connection():
            print('Cannot establish FTP connection. File cannot be transferred')
            return

        filename = os.path.split(remote)[-1]

        # Download file
        with open(local, 'wb') as f:
            start_time = time.time()
            self.connection.retrbinary('RETR ' + remote, f.write)
            elapsed_time = time.time() - start_time
            print('Transferred file {} from instrument to {}. Transfer time: {:.4f}s'.format(filename, local,
                                                                                             elapsed_time))
        # Delete file after it has been transferred
        if rm:
            try:
                self.connection.delete(remote)
                print('Deleted file {} from instrument'.format(filename))
            except ftplib.error_perm as e:
                print(e)

    def get_data(self, data_name, rm=True):
        """Downloads image/spectrum
        :param data_name:   str     Filename of image/spectrum on remote machine
        :param rm:          bool    If True, the file is deleted from the host once it has been transferred
                                    If False, transfer will keep running indefinitely as the transfer does not directly
                                    identify files it has already transferred previously
        """
        # Test FTP connection
        if not self.test_connection():
            print('Cannot establish FTP connection. File cannot be transferred')
            return

        # Filename organisation for local machine
        # File always goes into dated directory (date is extracted from filename)
        file = os.path.split(data_name)[-1]
        filename, ext = os.path.splitext(file)

        # Get correct directory to save to from the directory handler object
        if ext == self.cam_specs.file_ext:
            local_date_dir = self.img_dir.get_file_dir(filename)
        elif ext == self.spec_specs.file_ext:
            local_date_dir = self.spec_dir.get_file_dir(filename)
        else:
            return

        # Create full path to save to
        local_name = os.path.join(local_date_dir, file)
        local_name = local_name.replace(os.sep, '/')

        # Check if file exists - dont overwrite it if so
        if os.path.exists(local_name):
            print('File already exists on local machine, transfer aborted: {}'.format(file))
        else:
            # Download file
            lock_file = local_name.replace(ext, '.lock')
            open(lock_file, 'a').close()
            with open(local_name, 'wb') as f:
                start_time = time.time()
                self.connection.retrbinary('RETR ' + data_name, f.write)
                elapsed_time = time.time() - start_time
            os.remove(lock_file)
            print('Transferred file {} from instrument to {}. Transfer time: {:.4f}s'.format(filename, local_date_dir,
                                                                                             elapsed_time))

        # Delete file after it has been transferred
        if rm:
            try:
                self.connection.delete(data_name)
                print('Deleted file {} from instrument'.format(filename))
            except ftplib.error_perm as e:
                print(e)

    def watch_dir(self, lock='.lock', new_only=False):
        """Public access thread starter for _watch_dir"""
        if not self.test_connection():
            print('FTP connection could not be established, directory watching is not possible')
            return

        self.watch_thread = threading.Thread(target=self._watch_dir, args=(lock, new_only,))
        self.watch_thread.daemon = True
        self.watch_thread.start()
        self.watching_dir = True

    def _watch_dir(self, lock='.lock', new_only=False):
        """
        Watches directory for new files and adds image to queue when they appear. Watches the directory defined by
        self.dir_data_remote.

        :param lock: str
            Filename extension which defines when the file is not ready to be collected (in a locked state)
        :param new_only: bool
            If True, only new files are transfered across - files already on the instrument are ignored - used for
            manual acquisitions where we don't want to transfer old images, we want to start with any new ones taken
            during manual acquisition
        """
        # Empty old queue
        with self.watch_q.mutex:
            self.watch_q.queue.clear()

        # If new_only we need to get a list of all current images, so we can ignore all of these
        if new_only:
            # Get file list from host machine
            try:
                ignore_list = self.connection.nlst()
                ignore_list.sort()
                # print('Ignoring files already on instrument: {}'.format(ignore_list))
            except ftplib.error_perm as e:
                print(e)
                ignore_list = []

        while True:
            # First check we haven't had a stop request
            try:
                mess = self.watch_q.get(block=False)
                if mess:
                    self.watching_dir = False
                    print('Stopped FTP file watching ')
                    return
            except queue.Empty:
                pass

            # Get file list from host machine
            try:
                file_list = self.connection.nlst()
            except ftplib.error_perm as e:
                print(e)
                continue

            # Sort file list so oldest times come first (so they will be transferred first)
            file_list.sort()

            # Loop through files and decide what to do with them
            for file in file_list:
                # Always check if a quit has been requested - this ensures we don't get stuck here if transferring a lot
                # of files
                try:
                    mess = self.watch_q.get(block=False)
                    if mess:
                        self.watching_dir = False
                        print('Stopped FTP file watching')
                        return
                except queue.Empty:
                    pass

                # Check if this is a file that we want to ignore
                if new_only:
                    if file in ignore_list:
                        continue

                # Extract filename to generate lock file
                filename, ext = os.path.splitext(file)
                lock_file = filename + lock
                if lock_file in file_list:      # Don't download image if it is still locked
                    continue
                else:
                    print('Getting file: {}'.format(file))
                    self.get_data(os.path.join(self.dir_data_remote, file))

            # Sleep for designated time, so I'm not constantly polling the host
            time.sleep(self.refresh_time)

    def stop_watch(self):
        """Stops FTP file transfer"""
        if self.watching_dir:
            self.watch_q.put(1)







