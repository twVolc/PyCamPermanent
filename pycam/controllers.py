# -*- coding: utf-8 -*-

"""
Main controller classes for the PiCam and OO Flame spectrometer
"""

import warnings
import queue
import multiprocessing.managers
import io
import os
import time
import datetime
import numpy as np
import cv2

from .setupclasses import CameraSpecs, SpecSpecs
from .utils import format_time

try:
    import seabreeze.spectrometers as sb
except ModuleNotFoundError:
    warnings.warn('Working on machine without seabreeze, functionality of some classes will be lost')
try:
    import picamera
except ModuleNotFoundError:
    warnings.warn('Working on machine without picamera, functionality of some classes will be lost')


class Camera(CameraSpecs):
    """Main class for camera control

    subclass of: class: CameraSpecs
    """
    def __init__(self, band='on', filename=None):
        self.band = band                    # 'on' or 'off' band camera
        self.capture_q = queue.Queue()      # Queue for requesting images
        self.img_q = queue.Queue()          # Queue where images are put for extraction
        self.cam = picamera.PiCamera()      # picamera object for control of camera acquisitions

        self.filename = None                                        # Image filename
        self._analog_gain = 1
        self.exposure_speed = None  # True camera exposure speed retrieved from picamera object

        # Get default specs from parent class and any other attributes
        super().__init__(filename)

        # Create empty image array after we have got pix_num_x/y from super()
        self.image = np.array([self.pix_num_x, self.pix_num_y])  # Image array

    @property
    def analog_gain(self):
        return self._analog_gain

    @analog_gain.setter
    def analog_gain(self, ag):
        """Set camera analog gain"""
        self._analog_gain = ag
        self.cam.analog_gain = ag

    def _q_check(self, q, q_type='capt'):
        """Checks type of queue object and returns queue (ret_q). Sets queue to default queue if none is provided"""
        if isinstance(q, multiprocessing.managers.BaseProxy):
            print('Using multiprocessing queue')
            ret_q = q
        elif isinstance(q, queue.Queue):
            print('Using Queue queue')
            ret_q = q
        else:
            print('Unrecognized queue object, reverting to default')
            if q_type == 'capt':
                ret_q = self.capture_q
            elif q_type == 'img':
                ret_q = self.img_q
            else:
                ret_q = queue.Queue()

        return ret_q


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
        framerate = 1 / (self.shutter_speed / 1000000.0)
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

    def generate_filename(self, time_str, img_type):
        """Generates the image filename

        Parameters
        ----------
        time_str: str
            Time string containing date and time
        img_type: str
            Type of image. Value should be retrieved from one of dictionary options in <self.file_img_type>"""
        return time_str + '_' + \
               self.file_filterids[self.band] + '_' + \
               str(self.analog_gain) + 'ag_' + \
               str(self.exposure_speed) + 'ss_' + \
               img_type + self.file_ext

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
            self.image = cv2.resize(data, (self.pix_num_x, self.pix_num_y), interpolation=cv2.INTER_AREA)

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

    def interactive_capture(self, capt_q=None):
        """Interactive capturing by requesting captures through img_q

        Parameters
        ---------
        capt_q: Queue-like object
            Capture commands are passed to this object using its put() method
        """
        # Setup queue (code repeated in capture_sequence() - possibly turn this check into a function)
        capt_q = self._q_check(capt_q, q_type='capt')

        while True:

            # Wait for imaging command (expecting a dictionary containing information for acquisition)
            command = capt_q.get(block=True)

            # Extract img queue
            if 'img_q' not in command or command['img_q'] is None:
                img_q = self.img_q
            else:
                img_q = command['img_q']

            # Set shutter speed
            self.set_shutter_speed(command['ss'])

            # If a sequence isn't requested we take one typical image
            if command['type'] is not 'meas':

                # Get time and format
                time_str = format_time(datetime.datetime.now())

                # Capture image
                self.capture()

                # Generate filename
                filename = self.generate_filename(time_str, command['type'])

                # Put filename and image in queue
                img_q.put([filename, self.image])

            # Otherwise we capture a sequence
            else:
                # If we have been provided with a queue for images we pass this to capture_sequence()
                if 'img_q' in command:
                    self.capture_sequence(command['img_q'])
                else:
                    self.capture_sequence()

    def capture_sequence(self, img_q=None):
        """Main capturing sequence

        Parameters
        ----------
        img_q: Queue-like object, such as <queue.Queue> or <multiprocessing.Queue>
            Filenames and images are passed to this object using its put() method"""
        # Setup queue
        img_q = self._q_check(img_q, q_type='img')

        # Set shutter speed to start
        self.set_shutter_speed(self.shutter_speed)

        # Get acquisition rate in seconds
        frame_rep = round(1 / self.framerate)

        # Previous second value for check that we don't take 2 images in one second
        prev_sec = None

        while True:

            # Rethink this later - how to react perhaps depends on what is sent to the queue?
            try:
                mess = self.capture_q.get(block=False)
                return
            except queue.Empty:
                pass

            # Get current time
            time_obj = datetime.datetime.now()

            # Only capture an image if we are at the right time
            if time_obj.second % frame_rep == 0 and time_obj != prev_sec:

                # Generate time string
                time_str = format_time(time_obj)

                # Acquire image
                self.capture()

                # Generate filename
                filename = self.generate_filename(time_str, self.file_img_type['meas'])

                # Generate filename for image and save it
                # self.save_current_image(self.generate_filename(time_str, self.file_img_type['meas']))

                # Put filename and image into q
                img_q.put([filename, self.image])

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

        self.capture_q = queue.Queue()      # Queue for requesting spectra
        self.spec_q = queue.Queue()         # Queue to put spectra in for access elsewhere

        # Discover spectrometer devices
        self.devices = None  # List of detected spectrometers
        self.spec = None  # Holds spectrometer for interfacing via seabreeze
        self.find_device()

        # Set integration time (ALL IN MICROSECONDS)
        self._int_limit_lower = 1000  # Lower integration time limit
        self._int_limit_upper = 20000000  # Upper integration time limit
        self._int_time = None  # Integration time attribute
        self.int_time = self.start_int_time

        self.min_coadd = 1
        self.max_coadd = 100
        self._coadd = None  # Controls coadding of spectra
        self.coadd = self.start_coadd

    def find_device(self):
        """Function to search for devices"""
        try:
            self.devices = sb.list_devices()
            self.spec = sb.Spectrometer(self.devices[0])
            self.spec.trigger_mode(0)

            # If we have a spectrometer we then retrieve its wavelength calibration and store it as an attribute
            self.get_wavelengths()

        except IndexError:
            self.devices = None
            self.spec = None
            raise SpectrometerConnectionError('No spectrometer found')

    @property
    def int_time(self):
        return self._int_time / 1000  # Return time in milliseconds

    @int_time.setter
    def int_time(self, int_time):
        """Set integration time

        Parameters
        ----------
        int_time: int
            Integration time for spectrometer, provided in milliseconds
        """
        # Adjust to work in microseconds (class takes time in milliseconds) and ensure we have an <int>
        int_time = int(int_time * 1000)

        # Check requested integration time is acceptable
        if int_time < self._int_limit_lower:
            raise ValueError('Integration time below %i us is not possible' % self._int_limit_lower)
        elif int_time > self._int_limit_upper:
            raise ValueError('Integration time above %i us is not possible' % self._int_limit_upper)

        self._int_time = int_time

        # Adjust _int_time_idx to reflect the closest integration time to the current int_time
        self._int_time_idx = np.argmin(np.abs(self.int_list - self.int_time))

        # Set spectrometer integration time
        self.spec.integration_time_micros(int_time)

    @property
    def int_time_idx(self):
        return self._int_time_idx

    @int_time_idx.setter
    def int_time_idx(self, value):
        """Update integration time to value in int_list defined by int_time_idx when int_time_idx is changed
        Accesses hidden variable _int_time directly to avoid causing property method being called"""
        self._int_time_idx = value
        self._int_time = self.int_list[self.int_time_idx]

    @property
    def coadd(self):
        return self._coadd

    @coadd.setter
    def coadd(self, coadd):
        """Set coadding property"""
        if coadd < self.min_coadd:
            coadd = self.min_coadd
        elif coadd > self.max_coadd:
            coadd = self.max_coadd
        self._coadd = int(coadd)

    def generate_filename(self, time_str, spec_type):
        """Generates the spectrum filename

        Parameters
        ----------
        time_str: str
            Time string containing date and time
        """
        return time_str + '_' + str(self.int_time) + 'ss_' + str(self.coadd) + 'coadd_' + spec_type + self.file_ext

    def get_spec(self):
        """Acquire spectrum from spectrometer"""
        # Set array for coadding spectra
        coadded_spectrum = np.zeros(len(self.wavelengths))

        # Loop through number of coadds
        for i in range(self.coadd):
            coadded_spectrum += self.spec.intensities()

        # Correct for number of coadds to result in a spectrum with correct digital numbers for bit-depth of device
        coadded_spectrum /= self.coadd
        self.spectrum = coadded_spectrum

    def get_spec_now(self):
        """Immediately acquire spectrum from spectrometer - does not discard first spectrum (probably never used)"""
        self.spectrum = self.spec.intensities()

    def get_wavelengths(self):
        """Returns wavelengths"""
        self.wavelengths = self.spec.wavelengths()

    def extract_subspec(self, wavelengths):
        """Extract and return wavelengths and spectrum data for subsection of spectrum defined by wavelengths

        Parameters
        ----------
        wavelengths: list, tuple

        Returns
        -------
        wavelengths: list
            wavelengths of spectrometer extracted between range requested
        spectrum: list
            intensities from spectrum extracted between requested range
        """
        # Check wavelengths have been provided correctly
        if len(wavelengths) != 2:
            raise ValueError('Expected list or tuple of length 2')

        # Determine indices of arrays where wavelengths are closest to requested extraction wavelengths
        min_idx = np.argmin(np.abs(wavelengths[0] - self.wavelengths))
        max_idx = np.argmax(np.abs(wavelengths[1] - self.wavelengths))

        return self.wavelengths[min_idx:max_idx+1], self.spectrum[min_idx:max_idx+1]

    def check_saturation(self):
        """Check spectrum saturation
        return -1: if saturation exceeds the maximum allowed
        return 1:  if saturation is below minimum allowed
        return 0:  otherwise
        """
        # Extract spectrum in specific wavelength range to be checked
        wavelengths, spectrum = self.extract_subspec(self.saturation_range)

        if np.amax(spectrum) / self._max_DN > self.max_saturation:
            return -1
        elif np.amax(spectrum) / self._max_DN < self.min_saturation:
            return 1
        else:
            return 0

    def capture_sequence(self):
        """Captures sequence of spectra"""
        if self.int_time is None:
            raise ValueError('Cannot acquire sequence until initial integration time is correctly set')

        # Get acquisition rate in seconds
        frame_rep = round(1 / self.framerate)

        # Previous second value for check that we don't take 2 images in one second
        prev_sec = None

        while True:

            # Rethink this later - how to react perhaps depends on what is sent to the queue?
            try:
                mess = self.capture_q.get(block=False)
                return
            except queue.Empty:
                # If there is nothing in the queue telling us to stop then we continue with acquisitions
                pass

            # Get current time
            time_obj = datetime.datetime.now()

            # Only capture an image if we are at the right time
            if time_obj.second % frame_rep == 0 and time_obj != prev_sec:

                # Generate time string
                time_str = format_time(time_obj)

                # Acquire spectra
                self.get_spec()

                # Generate filename
                filename = self.generate_filename(time_str, self.file_spec_type['meas'])

                # Add spectrum and filename to queue
                self.spec_q.put([filename, self.spectrum])

                # Check image saturation and adjust shutter speed if required
                if self.auto_int:
                    adj_saturation = self.check_saturation()
                    if adj_saturation:
                        self.int_time_idx += adj_saturation
                        self.int_time = self.int_list[self.int_time_idx]

                # Set seconds value (used as check to prevent 2 images being acquired in same second)
                prev_sec = time_obj.second

    def capture_darks(self):
        """Capture dark images from all shutter speeds in <self.ss_list>"""
        # Loop through shutter speeds in ss_list
        for int_time in self.int_list:

            # Set camera shutter speed
            self.int_time = int_time

            # Get time for stamping
            time_str = format_time(datetime.datetime.now())

            # Acquire image
            self.get_spec()

            # Generate filename for spectrum
            filename = self.generate_filename(time_str, self.file_spec_type['dark'])

            # Add data to queue
            self.spec_q.put(filename)
            self.spec_q.put(self.spectrum)


class SpectrometerConnectionError(Exception):
    """
    Error raised if no spectrometer is detected
    """
    pass
