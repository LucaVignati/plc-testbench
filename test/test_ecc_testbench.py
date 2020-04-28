import unittest
from numpy import zeros
from alohaecc.ecc_testbench import ecc_testbench as tb
from alohaecc.ecc_algorithm import zeros_ecc
import glob

wave_files = glob.glob('../*/**/*.wav', recursive=True)


class test_ecc_testbench(unittest.TestCase):

    def setUp(self):
        self.ea = zeros_ecc
        self.testbench = tb(ecc_algorithm=self.ea)
        self.assertTrue(True)

    def testRun(self):

        expected_results = zeros(len(wave_files))
        results = self.testbench.run(wave_files)
        self.assertEqual(expected_results, results)
        print(results)


if __name__ == '__main__':

    unittest.main()
