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
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((4, 4, 16)))
        dst = Core.function_sum(src, (0, 1))
        self.assertEqual(dst.data_shape, dst.data.shape)

    def test_fourier_filter_gives_sensible_units_when_source_has_units(self):
        dimensional_calibrations = [Calibration.Calibration(units="mm"), Calibration.Calibration(units="mm")]
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((32, 32)), dimensional_calibrations=dimensional_calibrations)
        dst = Core.function_ifft(Core.function_fft(src))
        self.assertEqual(dst.dimensional_calibrations[0].units, "mm")
        self.assertEqual(dst.dimensional_calibrations[1].units, "mm")

    def test_fourier_filter_gives_sensible_units_when_source_has_no_units(self):
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((32, 32)))
        dst = Core.function_ifft(Core.function_fft(src))
        self.assertEqual(dst.dimensional_calibrations[0].units, "")
        self.assertEqual(dst.dimensional_calibrations[1].units, "")

    def test_fourier_mask_works_with_all_dimensions(self):
        dimension_list = [(32, 32), (31, 30), (30, 31), (31, 31), (32, 31), (31, 32)]
        for h, w in dimension_list:
            data = DataAndMetadata.DataAndMetadata.from_data(numpy.random.randn(h, w))
            mask = DataAndMetadata.DataAndMetadata.from_data((numpy.random.randn(h, w) > 0).astype(numpy.float))
            fft = Core.function_fft(data)
            masked_data = Core.function_ifft(Core.function_fourier_mask(fft, mask)).data
            self.assertAlmostEqual(numpy.sum(numpy.imag(masked_data)), 0)

    def test_slice_sum_grabs_signal_index(self):
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        slice = Core.function_slice_sum(data, 2, 2)
        self.assertTrue(numpy.array_equal(numpy.sum(random_data[..., 1:3], 2), slice.data))
        self.assertEqual(slice.dimensional_shape, random_data.shape[0:2])
        self.assertEqual(slice.intensity_calibration, c0)
        self.assertEqual(slice.dimensional_calibrations[0], c1)
        self.assertEqual(slice.dimensional_calibrations[1], c2)


    def test_pick_grabs_signal_index(self):
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        pick = Core.function_pick(data, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[2, 1, :], pick.data))
        self.assertEqual(pick.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], c3)

    def test_sum_region_produces_correct_result(self):
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        mask_data = numpy.zeros((3, 4), numpy.int)
        mask_data[0, 1] = 1
        mask_data[2, 2] = 1
        mask = DataAndMetadata.new_data_and_metadata(mask_data)
        sum_region = Core.function_sum_region(data, mask)
        self.assertTrue(numpy.array_equal(random_data[0, 1, :] + random_data[2, 2, :], sum_region.data))
        self.assertEqual(sum_region.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(sum_region.intensity_calibration, c0)
        self.assertEqual(sum_region.dimensional_calibrations[0], c3)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
