# standard libraries
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import xdata_1_0 as xdata


class TestCore(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_data_slice_typing(self) -> None:
        xd = xdata.new_with_data(numpy.zeros((100, 100), dtype=numpy.float32))
        xdata.data_slice(xd, (slice(10, 20), slice(20, 40)))
