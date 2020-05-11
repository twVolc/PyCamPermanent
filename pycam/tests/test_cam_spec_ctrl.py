# -*- coding: utf-8 -*-

"""
pycam test module for Camera and Spectrometer classes
"""

import cv2
from PIL import Image
import numpy as np

from pycam.controllers import Camera

class TestCamera:
    """Camera controller method tests"""

    camera_img = './test_data/2019-09-18T074335_fltrA_1ag_999904ss_Plume.png'

    def test_check_saturation(self):
        # INstantiate object
        cam = Camera()

        # Read image into numpy array
        cam.image = np.array(Image.open(self.camera_img))

        # Check saturation of image
        val = cam.check_saturation()

        assert val == 0

        # Adjust max saturation value to be below the max values in the image, so our image should be flagged as oversaturated
        cam.max_saturation = 0.7

        val = cam.check_saturation()

        assert val == -1

        # Adjust min saturation value to be above the max values in the image, so our image should be flagged as undersaturated

        cam.max_saturation = 1.0
        cam.min_saturation = 0.9

        val = cam.check_saturation()

        assert val == 1


