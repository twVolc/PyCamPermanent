# -*- coding: utf-8 -*-

from .doas_worker import DOASWorker

# ======================================================================================================================
# PROCESSING
# ======================================================================================================================
# Dictionary for species in DOAS retrieval, holding pathnames to reference spectra
species = {'SO2': 'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\doas\\calibration\\Vandaele (2009) x-section in wavelength.txt',
           'O3': 'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\doas\\calibration\\Serdyuchenko_O3_223K.txt'}

doas_worker = DOASWorker()     # Global DOAS worker