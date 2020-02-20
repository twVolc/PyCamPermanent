# -*- coding: utf-8 -*-

"""To be run on a raspberry pi
This script tests the interactive_capture() method of <object Camera>"""

from pycam.controllers import Camera
import threading

# Create camera object
cam = Camera()

# Initialise camera
cam.initialise_camera()

# Run dark mode capture
cam.capture_darks()