# -*- coding: utf-8 -*-

from .doas_worker import DOASWorker
from pycam.cfg import pyplis_worker
from pycam.utils import read_file
from pycam.setupclasses import FileLocator

# ======================================================================================================================
# PROCESSING
# ======================================================================================================================
# Dictionary for species in DOAS retrieval, holding pathnames to reference spectra
species = {'SO2': 'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\doas\\calibration\\Vandaele (2009) x-section in wavelength.txt',
           'O3': 'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\doas\\calibration\\Serdyuchenko_O3_223K.txt'}

# Load startup settings
process_settings = read_file(FileLocator.PROCESS_DEFAULTS)

# Global DOAS worker
doas_worker = DOASWorker(spec_dir=process_settings['init_spec_dir'].split('\'')[1],
                         dark_dir=process_settings['dark_spec_dir'].split('\'')[1],
                         q_doas=pyplis_worker.q_doas,
                         species=list(species.keys()))
