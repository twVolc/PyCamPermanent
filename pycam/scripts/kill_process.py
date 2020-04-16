#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""Kills previous scripts running on Pis which may interfere with new run"""

import sys
sys.path.append('/home/pi/')

from pycam.utils import kill_process

kill_process()

kill_process('pycam_spectrometer')