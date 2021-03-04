# -*- coding: utf-8 -*-

"""Utilities for pycam"""
import os
import numpy as np
import subprocess
import datetime
import threading


def check_filename(filename, ext):
    """Checks filename to ensure it is as expected

    Parameters
    ----------
    filename: str
        full filename, expected to contain file extension <ext>
    ext: str
        expected filename extension to be checked
    """
    # Ensure filename is string
    if not isinstance(filename, str):
        raise ValueError('Filename must be in string format')

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    # Split filename by .
    split_name = filename.split('.')

    # # Ensure filename contains exactly one . for file extension
    # if len(split_name) != 2:
    #     raise ValueError('Filename is not in the correct format. Name contained {} points'.format(len(split_name)-1))

    # Compare file extension to expected extension
    if split_name[-1] != ext:
        raise ValueError('Wrong file extension encountered')

    return


def write_file(filename, my_dict):
    """Writes all attributes of dictionary to file

    Parameters
    ----------
    filename: str
        file name to be written to
    my_dict: dict
        Dictionary of all data
    """
    # Check filename is legal
    try:
        check_filename(filename, 'txt')
    except ValueError:
        raise

    with open(filename, 'w') as f:
        # Loop through dictionary and write to file
        for key in my_dict:
            string = '{}={}\n'.format(key, my_dict[key])
            f.write(string)
    return


def read_file(filename, separator='=', ignore='#'):
    """Reads all lines of file separating into keys using the separator

        Parameters
        ----------
        filename: str
            file name to be written to
        separator: str
            string used to separate the key from its attribute
        ignore: str
            lines beginning with this string are ignored
            
        :returns
        data: dict
            dictionary of all attributes in file
    """
    # Check we are working with a text file
    check_filename(filename, 'txt')

    # Create empty dictionary to be filled
    data = dict()

    with open(filename, 'r') as f:

        # Loop through file line by line
        for line in f:

            # If line is start with ignore string then ignore line
            if line[0:len(ignore)] == ignore:
                continue

            try:
                # Split line into key and the key attribute
                key, attr = line.split(separator)[0:2]
            # ValueError will be thrown if nothing is after (or before) the equals sign. So we ignore these lines
            except ValueError:
                continue

            # Add attribute to dictionary, first removing any unwanted information at the end of the line
            # (including whitespace and #)
            data[key] = attr.split(ignore)[0].strip('\n').strip()

    return data


def format_time(time_obj, fmt):
    """Formats datetime object to string for use in filenames

    Parameters
    ----------
    time_obj: datetime.datetime
        Time to be converted to string"""
    return time_obj.strftime(fmt)
    # # Remove microseconds
    # time_obj = time_obj.replace(microsecond=0)
    #
    # # Return string format
    # return time_obj.isoformat().replace(':', '')


def kill_process(process='pycam_camera'):
    """Kills process on raspberry pi machine

    Parameters
    ----------
    process: str
        String for process to be killed, this may kill any process containing this as a substring, so use with caution
    """
    proc = subprocess.Popen(['ps axg'], stdout=subprocess.PIPE, shell=True)
    stdout_value = proc.communicate()[0]
    stdout_str = stdout_value.decode("utf-8")
    stdout_lines = stdout_str.split('\n')

    # Check ps axg output lines to see whether pycam is actually running
    for line in stdout_lines:
        if process in line:
            subprocess.call(['kill', line.split()[0]])


def make_circular_mask_line(h, w, cx, cy, radius, tol=0.008):
    """Create a circular access mask for accessing certain pixels in an image. T
    aken from pyplis.helpers.make_circular_mask and adapted to only produce a line mask, rather than a filled circle

    Parameters
    ----------
    h : int
        height of mask
    w : int
        width of mask
    cx : int
        x-coordinate of center pixel of disk
    cy : int
        y-coordinate of center pixel of disk
    radius : int
        radius of disk
    tol : int
        Tolerance % (+/-) for accepted true values around radius value

    Returns
    -------
    ndarray
        the pixel access mask

    """
    y, x = np.ogrid[:h, :w]
    rad_grid = np.round((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    rad_square_min = radius ** 2
    rad_square_min *= 1 - tol
    rad_square_max = radius ** 2
    rad_square_max *= 1 + tol

    return  np.where((rad_grid >= rad_square_min) & (rad_grid <= rad_square_max), True, False)


def calc_dt(img_prev, img_curr):
    """
    Calculates time difference between two pyplis.Img objects
    :param img_prev: pyplis.Img
    :param img_curr: pyplis.Img
    :return: Time difference in seconds between the two images
    """
    t_delt = img_curr["start_acq"] - img_prev["start_acq"]

    return t_delt.total_seconds()


def get_img_time(filename, date_loc=0, date_fmt="%Y-%m-%dT%H%M%S"):
    """
    Gets time from filename and converts it to datetime object
    :param filename:
    :return img_time:
    """
    # Make sure filename only contains file and not larger pathname
    filename = filename.split('\\')[-1].split('/')[-1]

    # Extract time string from filename
    time_str = filename.split('_')[date_loc]

    # Turn time string into datetime object
    img_time = datetime.datetime.strptime(time_str, date_fmt)

    return img_time


def get_spec_time(filename, date_loc=0, date_fmt="%Y-%m-%dT%H%M%S"):
    """
    Gets time from filename and converts it to datetime object
    :param filename:
    :return spec_time:
    """
    # Make sure filename only contains file and not larger pathname
    filename = filename.split('\\')[-1].split('/')[-1]

    # Extract time string from filename
    time_str = filename.split('_')[date_loc]

    # Turn time string into datetime object
    spec_time = datetime.datetime.strptime(time_str, date_fmt)

    return spec_time


class StorageMount:
    """
    Basic class to control the handling of mounting external memory and storing details of mounted drive
    """
    def __init__(self, mount_path='/mnt/pycam/', dev_path=None):
        self.dev_path = dev_path
        self.mount_path = mount_path
        self.lock = threading.Lock()

        if self.dev_path is None:
            self.find_dev()

    @property
    def is_mounted(self):
        """Check whether device is already mounted"""
        mnt_output = subprocess.check_output('mount')
        if self.dev_path is None:
            return False
        mnt_stat = mnt_output.find(self.dev_path.encode())
        if mnt_stat == -1:
            return False
        else:
            return True

    def find_dev(self):
        """
        Finds device location based on it being /dev/sda of some kind (not necessarily sda1) and sets self.dev_path
        """
        proc = subprocess.Popen(['sudo fdisk -l /dev/sda'], stdout=subprocess.PIPE, shell=True)
        stdout_value = proc.communicate()[0]
        stdout_str = stdout_value.decode("utf-8")
        stdout_lines = stdout_str.split('\n')

        # Check output to find sda
        for line in stdout_lines:
            if 'HPFS/NTFS' in line:
                # TODO do some line splitting to get sda path
                sda_path = line.split()[0]
                self.dev_path = sda_path

    def mount_dev(self):
        """Mount device located at self.dev_path to self.mount_path destination"""
        if not os.path.exists(self.mount_path):
            subprocess.call(['sudo', 'mkdir', self.mount_path])

        # For better compatibility, Should probably use fdisk -l /dev/sda to find all devices
        # then search this string to determine what value X takes in /dev/sdaX. Then use this
        # in the mounting process, rather than just assuming the device is at '/dev/sda1'. Should
        # probably use this in the mntOutput.find() expression too, so I'm searching for the right device.
        # SHould use try: or something that catches if the /dev/sda1 doesn't exist - i.e. no USB stick plugged in.
        # THen print - please plug in device.
        if not self.is_mounted:
            subprocess.call(['sudo', 'mount', '-o', 'uid=pi,gid=pi', self.dev_path, self.mount_path])

    def unmount_dev(self):
        """Unmount device located at self.dev_path"""
        # Unmounting through /dev and not /mnt will ensure usb is unmounted
        # even if it has been manually mounted to a different directory. However, this method does mean I may
        # unmount the wrong device - so this needs to be thought about some more.
        if self.is_mounted:
            subprocess.call(['sudo', 'umount', self.dev_path])