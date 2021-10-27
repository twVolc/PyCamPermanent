#!/bin/bash

# Script starts picam without automatic capture
# Used when directly requesting script start from computer SSH

/usr/bin/python3 /home/pi/pycam/scripts/pycam_masterpi.py "$@" & > /home/pi/pycam/logs/run.log
