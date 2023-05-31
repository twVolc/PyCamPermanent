# -*- coding: utf-8 -*-

"""
pycam test module for PyplisWorker
"""

from pycam.so2_camera_processor import PyplisWorker
from pycam.doas.doas_worker import DOASWorker
from pydoas.analysis import DoasResults
import datetime
import time
import numpy as np


class TestPyplisWorker:

    doas_worker = DOASWorker()
    pyplis_worker = PyplisWorker()

    def add_results(self):
        """Just adding some results with different times, for tests"""
        for i in range(12):
            results = {'time': datetime.datetime(year=2020, month=11, day=30, hour=i),
                       'column_density': 1000,
                       'std_err': 20}
            self.doas_worker.add_doas_results(results)

    def test_doas_results_generation(self):
        """Generates new doas_results"""
        column_densities = [100, 200, 500]
        times = [datetime.datetime(year=2020, month=11, day=30, hour=12, minute=30),
                 datetime.datetime(year=2020, month=11, day=30, hour=12, minute=35),
                 datetime.datetime(year=2020, month=11, day=30, hour=12, minute=40)]
        stds = [10, 20, 50]
        species = 'SO2'

        self.doas_worker.results = DoasResults(column_densities, index=times, fit_errs=stds, species_id=species)
        print(self.doas_worker.results.index)

    def test_doas_results(self):
        """Tests adding and removing doas results"""
        results = {'time': datetime.datetime(year=2020, month=11, day=30, hour=12),
                   'column_density': 1000,
                   'std_err': 20}
        t = time.time()

        for i in range(100):
            self.doas_worker.add_doas_results(results)

        print('Elapsed time: {}'.format(time.time() - t))

    def test_doas_rem_results(self):
        """Test removing results from DoasResults"""
        self.add_results()
        time_obj = self.doas_worker.results.index[4]
        print('Removing time and earlier: {}'.format(time_obj))
        print('Times: {}'.format(self.doas_worker.results))
        results = self.doas_worker.rem_doas_results(time_obj, inplace=False)
        print('Times: {}'.format(results))
        self.doas_worker.rem_doas_results(time_obj, inplace=True)
        assert np.array_equal(self.doas_worker.results, results)
        assert np.array_equal(self.doas_worker.results.fit_errs, results.fit_errs)