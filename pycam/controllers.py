# -*- coding: utf-8 -*-

"""
Main controller classes for the PiCam and OO Flame spectrometer
"""

import warnings
import queue
import io
import os
import time
import datetime
import numpy as np
import cv2

from .setupclasses import CameraSpecs, SpecSpecs
from .utils import format_time

try:
    import picamera
except ModuleNotFoundError:
    warnings.warn('Working on machine without picamera, functionality of some classes will be lost')


class Camera(CameraSpecs):
    """Main class for camera control

    subclass of: class: CameraSpecs
    """
    def __init__(self, band='on', filename=None):
        super().__init__(filename)
        self.band = band                    # 'on' or 'off' band camera
        self.capture_q = queue.Queue()      # Queue for requesting images
        self.cam = picamera.PiCamera()      # picamera object for control of camera acquisitions

        self.image = np.array([self.pix_size_x, self.pix_size_y])   # Image array
        self.filename = None                                        # Image filename

        self._analog_gain = 1
        self.exposure_speed = None  # True camera exposure speed retrieved from picamera object

    @property
    def analog_gain(self):
        return self._analog_gain

    @analog_gain.setter
    def analog_gain(self, ag):
        """Set camera analog gain"""
        self._analog_gain = ag
        self.cam.analog_gain = ag

    def initialise_camera(self):
        """Initialises PiCam by setting appropriate settings"""
        # Set camera shutter speed if possible, if not raise error
        if self.shutter_speed is None:
            raise ValueError('Shutter speed must be defined before camera can be initialised')
        else:
            self.set_shutter_speed(self.shutter_speed)

        # Turn exposure mode off, to prevent auto adjustments
        self.cam.exposure_mode = 'off'

        # Set analog gain (may want to think about most succinct way to hold/control this parameter)
        self.analog_gain = self._analog_gain

    def set_shutter_speed(self, ss):
        """Sets camera shutter speed and will wait until exposure speed has settled close to requested shutter speed

        Parameters
        ----------
        ss: int
            Shutter speed (in micro-seconds)"""
        self.shutter_speed = ss

        # Set framerate as this affects shutter speed
        self.set_cam_framerate()

        # Set shutter speed
        self.cam.shutter_speed = ss

        # Wait for camera exposure speed to settle on new value
        self.exposure_speed = self.check_exposure_speed()

    def set_cam_framerate(self):
        """Determines appropriate framerate based on current shutter speed (framerate limits shutter speed so must be
        set appropriately)"""
        framerate = 0.95 / (self.shutter_speed / 1000000.0)
        if framerate > 20:
            framerate = 20
        self.cam.framerate = framerate

    def check_exposure_speed(self):
        """Checks that exposure speed is within reasonable limits of shutter speed"""
        while self.cam.exposure_speed < 0.95 * self.shutter_speed or self.cam.exposure_speed > self.shutter_speed:
            time.sleep(0.01)  # Sleep until camera exposure speed is set close enough to requested ss

        # Return the camera's exact exposure speed
        return '{}'.format(self.cam.exposure_speed)

    def image_saturation(self):
        """Check image saturation
        return -1: if saturation exceeds the maximum allowed
        return 1:  if saturation is below minimum allowed
        return 0:  otherwise"""
        if np.amax(self.image) / self._max_DN > self.max_saturation:
            return -1
        elif np.amax(self.image) / self._max_DN < self.min_saturation:
            return 1
        else:
            return 0

    def generate_filename(self, time_str, type):
        """Generates the image filename

        Parameters
        ----------
        time_str: str
            Time string containing date and time
        type: str
            Type of image. Value should be retrieved from one of dictionary options in <self.file_img_type>"""
        return time_str + '_' + \
               self.file_filterids[self.band] + '_' + \
               str(self.analog_gain) + 'ag_' + \
               str(self.exposure_speed) + 'ss_' + \
               type + self.file_ext

    def capture(self):
        """Controls main capturing process on PiCam"""

        # Set up bytes stream
        with io.BytesIO() as stream:

            # Capture image to stream
            self.cam.capture(stream, format='jpeg', bayer=True)

            # ====================
            # RAW data extraction
            # ====================
            data = stream.getvalue()[-6404096:]
            assert data[:4] == b'BRCM'
            data = data[32768:]
            data = np.fromstring(data, dtype=np.uint8)

            reshape, crop = (1952, 3264), (1944, 3240)

            data = data.reshape(reshape)[:crop[0], :crop[1]]

            data = data.astype(np.uint16) << 2
            for byte in range(4):
                data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
            data = np.delete(data, np.s_[4::5], 1)
            # ====================

            # Resize image to requested size
            self.image = cv2.resize(data, (self.pix_size_x, self.pix_size_y), interpolation=cv2.INTER_AREA)

    def save_current_image(self, filename):
        """Saves image

        Parameters
        ----------
        filename: str
            Name of file to save image to"""
        # Generate lock file to prevent image from being accessed before capture has finished
        lock = filename.replace(self.file_ext, '.lock')
        open(lock, 'a').close()

        # Save image
        cv2.imwrite(filename, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        self.filename = filename

        # Remove lock to free image for transfer
        os.remove(lock)

    def capture_sequence(self):
        """Main capturing sequence"""
        # Set shutter speed to start
        self.set_shutter_speed(self.shutter_speed)

        # Get acquisition rate in seconds
        frame_rep = round(1 / self.framerate)

        # Previous second value for check that we don't take 2 images in one second
        prev_sec = None

        while True:

            # Rethink this later - how to react perhaps depends on what is sent to the queue?
            if self.capture_q.get(block=False):
                return

            # Get current time
            time_obj = datetime.datetime.now()

            # Only capture an image if we are at the right time
            if time_obj.second % frame_rep == 0 and time_obj != prev_sec:

                # Generate time string
                time_str = format_time(time_obj)

                # Acquire image
                self.capture()

                # Generate filename for image and save it
                self.save_current_image(self.generate_filename(time_str, self.file_img_type['meas']))

                # Check image saturation and adjust shutter speed if required
                if self.auto_ss:
                    adj_saturation = self.image_saturation()
                    if adj_saturation:
                        self.ss_idx += adj_saturation
                        self.set_shutter_speed(self.ss_list[self.ss_idx])

                # Set seconds value (used as check to prevent 2 images being acquired in same second)
                prev_sec = time_obj.second

    def capture_darks(self):
        """Capture dark images from all shutter speeds in <self.ss_list>"""
        # Loop through shutter speeds in ss_list
        for ss in self.ss_list:

            # Set camera shutter speed
            self.set_shutter_speed(ss)

            # Get time for stamping
            time_str = format_time(datetime.datetime.now())

            # Acquire image
            self.capture()

            # Generate filename for image and save it
            self.save_current_image(self.generate_filename(time_str, self.file_img_type['dark']))






class Spectrometer(SpecSpecs):
    """Main class for spectrometer control

    subclass of :class: SpecSpecs
    """
    def __init__(self, filename):
        super().__init__(filename)

        self.capture_q = queue.Queue()       # Queue for requesting spectra

