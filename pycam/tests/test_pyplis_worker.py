# -*- coding: utf-8 -*-

"""
pycam test module for PyplisWorker
"""

from pycam.so2_camera_processor import PyplisWorker
import unittest
from parameterized import parameterized


class TestPyplisWorker(unittest.TestCase):

    @parameterized.expand([
        (300, False),
        (350, False),
        (360, True),
        (400, True)
    ])
    def test_buffer_size(self, buff_size, expected):
        pyplis_worker = PyplisWorker()
        pyplis_worker.img_dir = "./pycam/tests/test_data"
        pyplis_worker.img_list = pyplis_worker.get_img_list()
        pyplis_worker.doas_fov_recal_mins = 30
        pyplis_worker.img_buff_size = buff_size

        self.assertIs(pyplis_worker.check_buffer_size(), expected)
