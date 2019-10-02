# -*- coding: utf-8 -*-

"""Utilities for pycam"""


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

    # Split filename by .
    split_name = filename.split('.')

    # Ensure filename contains exactly one . for file extension
    if len(split_name) != 2:
        raise ValueError('Filename is not in the correct format. Name contained {} points'.format(len(split_name)-1))

    # Compare file extension to expected extension
    if split_name[-1] != ext:
        raise ValueError('Wrong file extension encountered')


def format_time(time_obj):
    """Formats datetime object to string for use in filenames

    Parameters
    ----------
    time_obj: datetime.datetime
        Time to be converted to string"""
    # Remove microseconds
    time_obj = time_obj.replace(microsecond=0)

    # Return string format
    return time_obj.isoformat().replace(':','')
