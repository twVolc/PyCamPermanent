# -*- coding: utf-8 -*-

"""Holds simple classes for information on the come of the commands in networking"""


class AcquisitionComms:
    """Contains acquisition attributes and their associated comm flag"""
    cam_dict = {'SSA': 'shutter_speed',
                'SSB': 'shutter_speed',
                'FRC': 'framerate',
                'ATA': 'auto_ss',
                'ATB': 'auto_ss',
                'SMN': 'min_saturation',
                'SMX': 'max_saturation',
                'PXC': 'saturation_pixels',
                'RWC': 'saturation_rows'
                }

    spec_dict = {'SSS': 'int_time',
                 'ATS': 'auto_int',
                 'FRS': 'framerate',
                 'CAD': 'coadd',
                 'PXS': 'saturation_pixels',
                 'SNS': 'min_saturation',
                 'SXS': 'max_saturation',
                 'WMN': 'wavelength_min',
                 'WMX': 'wavelength_max'
                 }
