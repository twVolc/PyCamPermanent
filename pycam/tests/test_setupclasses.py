# -*- coding: utf-8 -*-

"""
pycam test module for setupclasses
"""

from pycam.setupclasses import CameraSpecs

class TestSpecs:
    def cam_specs_io(self):
        """Test basic IO functionality of the CameraSpecs object"""
        filename = 'temp_io_test.txt'

        # Instantiate CameraSpecs object
        cam_1 = CameraSpecs()

        # Attempt to save camera specs
        cam_1.save_specs(filename)

        # Reload camera specs
        cam_2 = CameraSpecs(filename)

        # Check cam_1 against cam_2 to ensure load and save has been performed as anticipated
        assert cam_1.__dict__ == cam_2.__dict__

