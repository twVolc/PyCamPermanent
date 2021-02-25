# -*- coding: utf-8 -*-

"""Utilities for pycam"""
import os
import numpy as np
import subprocess


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
    cmd = ['ps axg | grep {}'.format(process)]
    p = subprocess.check_output(cmd, shell=True)
    print(p)
    subprocess.call(['kill', p.split()[0]])


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