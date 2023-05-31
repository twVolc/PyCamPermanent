from pycam.doas.cfg import doas_worker, process_settings
import pandas as pd
import numpy as np
import unittest

class TestIfitWorker(unittest.TestCase):

    def test_ifit_regression(self):
        # provide specific settings 
        ils_path = './pycam/doas/calibration/2019-07-03_302nm_ILS.txt'

        doas_worker.load_ils(ils_path)
        doas_worker.corr_light_dilution = True

        doas_worker.load_ld_lookup(process_settings["ld_lookup_1"][1:-1], fit_num=0)
        doas_worker.load_ld_lookup(process_settings["ld_lookup_2"][1:-1], fit_num=1)

        # Run doas_worker
        doas_worker.start_processing_threadless()
        # Load output(s) from new run
        # Load gold standard output(s)
        expected_output = pd.read_csv("./pycam/tests/test_data/test_spectra/ifit_test_standard/doas_results_2019-05-29T124401_124412.csv")

        # Compare original and gold standard
        self.assertTrue(all(expected_output["Time"] == doas_worker.results.index))
        self.assertTrue(np.allclose(doas_worker.results.values, expected_output["Column density"], equal_nan = True))
        self.assertTrue(np.allclose(doas_worker.results.fit_errs, expected_output["CD error"], equal_nan = True))
        self.assertTrue(np.allclose(doas_worker.results.ldfs, expected_output["LDF"], equal_nan = True))