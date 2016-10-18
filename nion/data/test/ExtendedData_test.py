# standard libraries
import logging
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import DataAndMetadata


class TestExtendedData(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_rgb_data_constructs_with_default_calibrations(self):
        data = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertEqual(len(xdata.dimensional_shape), len(xdata.dimensional_calibrations))

    def test_rgb_data_slice_works_correctly(self):
        data = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertTrue(xdata.is_data_rgb_type)
        xdata_slice = xdata[2:6, 2:6]
        self.assertTrue(xdata_slice.is_data_rgb_type)
        self.assertTrue(xdata_slice.dimensional_shape, (4, 4))

    def test_data_slice_calibrates_correctly(self):
        data = numpy.zeros((100, 100), dtype=numpy.float)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        calibrations = xdata[40:60, 40:60].dimensional_calibrations
        self.assertAlmostEqual(calibrations[0].offset, 40)
        self.assertAlmostEqual(calibrations[0].scale, 1)
        self.assertAlmostEqual(calibrations[1].offset, 40)
        self.assertAlmostEqual(calibrations[1].scale, 1)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
