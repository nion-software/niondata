# standard libraries
import logging
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import Context
from nion.data import Core
from nion.data import DataAndMetadata


class TestCore(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_something(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.uint32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = ctx.line_profile(a, ctx.vector(ctx.normalized_point(0.25, 0.25), ctx.normalized_point(0.5, 0.5)), 2)
        # print(b.data)

    def test_something_else(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.int32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = ctx.amin(a) - a
        # print(a.data)
        # print(b.data_shape)
        # print(b.data)

    def test_something_again(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.uint32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = a - ctx.amin(a)
        # print(a.data)
        # print(b.data_shape)
        # print(b.data)

    def test_fft_produces_correct_calibration(self):
        src_data = ((numpy.abs(numpy.random.randn(16, 16)) + 1) * 10).astype(numpy.float)
        dimensional_calibrations = (Calibration.Calibration(offset=3), Calibration.Calibration(offset=2))
        a = DataAndMetadata.DataAndMetadata.from_data(src_data, dimensional_calibrations=dimensional_calibrations)
        fft = Core.function_fft(a)
        self.assertAlmostEqual(fft.dimensional_calibrations[0].offset, -0.5)
        self.assertAlmostEqual(fft.dimensional_calibrations[1].offset, -0.5)
        ifft = Core.function_ifft(fft)
        self.assertAlmostEqual(ifft.dimensional_calibrations[0].offset, 0.0)
        self.assertAlmostEqual(ifft.dimensional_calibrations[1].offset, 0.0)

    def test_concatenate_works_with_1d_inputs(self):
        src_data1 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float)
        src_data2 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float)
        dimensional_calibrations = [Calibration.Calibration(offset=3)]
        a1 = DataAndMetadata.DataAndMetadata.from_data(src_data1, dimensional_calibrations=dimensional_calibrations)
        a2 = DataAndMetadata.DataAndMetadata.from_data(src_data2, dimensional_calibrations=dimensional_calibrations)
        c0 = Core.function_concatenate([a1, a2], 0)
        self.assertEqual(tuple(c0.data.shape), tuple(c0.data_shape))
        self.assertTrue(numpy.array_equal(c0.data, numpy.concatenate([src_data1, src_data2], 0)))

    def test_vstack_and_hstack_work_with_1d_inputs(self):
        src_data1 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float)
        src_data2 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float)
        dimensional_calibrations = [Calibration.Calibration(offset=3)]
        a1 = DataAndMetadata.DataAndMetadata.from_data(src_data1, dimensional_calibrations=dimensional_calibrations)
        a2 = DataAndMetadata.DataAndMetadata.from_data(src_data2, dimensional_calibrations=dimensional_calibrations)
        vstack = Core.function_vstack([a1, a2])
        self.assertEqual(tuple(vstack.data.shape), tuple(vstack.data_shape))
        self.assertTrue(numpy.array_equal(vstack.data, numpy.vstack([src_data1, src_data2])))
        hstack = Core.function_hstack([a1, a2])
        self.assertEqual(tuple(hstack.data.shape), tuple(hstack.data_shape))
        self.assertTrue(numpy.array_equal(hstack.data, numpy.hstack([src_data1, src_data2])))

    def test_sum_over_two_axes_returns_correct_shape(self):
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((16, 4, 4)))
        dst = Core.function_sum(src, (1, 2))
        self.assertEqual(dst.data_shape, dst.data.shape)

    def test_fourier_filter_gives_sensible_units(self):
        dimensional_calibrations = [Calibration.Calibration(units="mm"), Calibration.Calibration(units="mm")]
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((32, 32)), dimensional_calibrations=dimensional_calibrations)
        dst = Core.function_ifft(Core.function_fft(src))
        self.assertEqual(dst.dimensional_calibrations[0].units, "mm")
        self.assertEqual(dst.dimensional_calibrations[1].units, "mm")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
