# standard libraries
import copy
import logging
import random
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata


class TestCore(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_line_profile_uses_integer_coordinates(self):
        data = numpy.zeros((32, 32))
        data[16, 15] = 1
        data[16, 16] = 1
        data[16, 17] = 1
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        line_profile_data = Core.function_line_profile(xdata, ((8/32, 16/32), (24/32, 16/32)), 1.0).data
        self.assertTrue(numpy.array_equal(line_profile_data, data[8:24, 16]))
        line_profile_data = Core.function_line_profile(xdata, ((8/32 + 1/128, 16/32 + 1/128), (24/32 + 2/128, 16/32 + 2/128)), 1.0).data
        self.assertTrue(numpy.array_equal(line_profile_data, data[8:24, 16]))
        line_profile_xdata = Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertTrue(numpy.array_equal(line_profile_xdata.data, data[8:24, 16] * 3))

    def test_line_profile_width_adjusts_intensity_calibration(self):
        data = numpy.zeros((32, 32))
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        line_profile_xdata = Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertAlmostEqual(line_profile_xdata.intensity_calibration.scale, 1/3)

    def test_line_profile_width_computation_does_not_affect_source_intensity(self):
        data = numpy.zeros((32, 32))
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertAlmostEqual(xdata.intensity_calibration.scale, 1)

    def test_line_profile_produces_appropriate_data_type(self):
        # valid for 'nearest' mode only. ignores overflow issues.
        vector = (0.1, 0.2), (0.3, 0.4)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.int32)), vector, 3.0).data_dtype, numpy.int)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.uint32)), vector, 3.0).data_dtype, numpy.uint)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.float32)), vector, 3.0).data_dtype, numpy.float32)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.float64)), vector, 3.0).data_dtype, numpy.float64)

    def test_line_profile_rejects_complex_data(self):
        vector = (0.1, 0.2), (0.3, 0.4)
        with self.assertRaises(Exception):
            Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.complex128)), vector, 3.0)

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

    def test_fft_forward_and_back_is_consistent(self):
        src_data = numpy.random.randn(256, 256)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        fft = Core.function_fft(a)
        ifft = Core.function_ifft(fft)
        src_data_2 = ifft.data
        # error increases for size of data
        self.assertLess(numpy.amax(numpy.absolute(src_data - src_data_2)), 1E-12)
        self.assertLess(numpy.absolute(numpy.sum(src_data - src_data_2)), 1E-12)

    def test_fft_rms_is_same_as_original(self):
        src_data = numpy.random.randn(256, 256)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        fft = Core.function_fft(a)
        src_data_2 = fft.data
        self.assertLess(numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data)))) - numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data_2)))), 1E-12)

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

    def test_sum_over_rgb_produces_correct_data(self):
        data = numpy.zeros((3, 3, 4), numpy.uint8)
        data[1, 0] = (3, 3, 3, 3)
        src = DataAndMetadata.DataAndMetadata.from_data(data)
        dst0 = Core.function_sum(src, 0)
        dst1 = Core.function_sum(src, 1)
        self.assertEqual(dst0.data_shape, dst0.data.shape)
        self.assertEqual(dst1.data_shape, dst1.data.shape)
        self.assertTrue(numpy.array_equal(dst0.data[0], (1, 1, 1, 1)))
        self.assertTrue(numpy.array_equal(dst0.data[1], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst0.data[2], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst1.data[0], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst1.data[1], (1, 1, 1, 1)))
        self.assertTrue(numpy.array_equal(dst1.data[2], (0, 0, 0, 0)))

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
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        slice = Core.function_slice_sum(data_and_metadata, 2, 2)
        self.assertTrue(numpy.array_equal(numpy.sum(random_data[..., 1:3], 2), slice.data))
        self.assertEqual(slice.dimensional_shape, random_data.shape[0:2])
        self.assertEqual(slice.intensity_calibration, c0)
        self.assertEqual(slice.dimensional_calibrations[0], c1)
        self.assertEqual(slice.dimensional_calibrations[1], c2)

    def test_pick_grabs_datum_index_from_3d(self):
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[2, 1, :], pick.data))
        self.assertEqual(pick.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], c3)

    def test_pick_grabs_datum_index_from_4d(self):
        random_data = numpy.random.randn(3, 4, 5, 6)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        c4 = Calibration.Calibration(units="e")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3, c4], data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[2, 1, ...], pick.data))
        self.assertEqual(pick.dimensional_shape, random_data.shape[2:4])
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], c3)
        self.assertEqual(pick.dimensional_calibrations[1], c4)

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

    def test_slice_sum_works_on_2d_data(self):
        random_data = numpy.random.randn(4, 10)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2])  # last index is signal
        result = Core.function_slice_sum(data_and_metadata, 5, 3)
        self.assertTrue(numpy.array_equal(numpy.sum(random_data[..., 4:7], -1), result.data))
        self.assertEqual(result.intensity_calibration, data_and_metadata.intensity_calibration)
        self.assertEqual(result.dimensional_calibrations[0], data_and_metadata.dimensional_calibrations[0])

    def test_fft_works_on_rgba_data(self):
        random_data = numpy.random.randint(0, 256, (32, 32, 4), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        Core.function_fft(data_and_metadata)

    def test_display_data_2d_not_a_view(self):
        random_data = numpy.random.randint(0, 256, (2, 2), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        display_xdata = Core.function_display_data(data_and_metadata)
        display_xdata_copy = copy.deepcopy(display_xdata)
        data_and_metadata.data[:] = 0
        self.assertTrue(numpy.array_equal(display_xdata.data, display_xdata_copy.data))

    def test_display_rgba_with_1d_rgba(self):
        random_data = numpy.random.randint(0, 256, (32, 4), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        Core.function_display_rgba(data_and_metadata)

    def test_align_works_on_2d_data(self):
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4, 1.2)
        xdata_shifted = Core.function_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, 100, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], 1)
        self.assertAlmostEqual(shift[1], measured_shift[1], 1)
        result = Core.function_align(data, xdata_shifted, 100) - xdata_shifted
        self.assertAlmostEqual(result.data.mean(), 0)

    def test_align_works_on_1d_data(self):
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4,)
        xdata_shifted = Core.function_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, 100, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], 1)
        result = Core.function_align(data, xdata_shifted, 100) - xdata_shifted
        self.assertAlmostEqual(result.data.mean(), 0)

    def test_sequence_register_works_on_2d_data(self):
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64, 64))
        for p in range(sdata.shape[0]):
            shift = (p / (sdata.shape[0] - 1) * -3.4, p / (sdata.shape[0] - 1) * 1.2)
            sdata[p, ...] = Core.function_shift(xdata, shift).data
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        shifts = Core.function_sequence_register_translation(sxdata, 100, True).data
        self.assertEqual(shifts.shape, (sdata.shape[0] - 1, 2))
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][0], 1 / (sdata.shape[0] - 1) * 3.4, 1)
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][1], 1 / (sdata.shape[0] - 1) * -1.2, 1)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[0], 3.4, 0)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[1], -1.2, 0)

    def test_sequence_register_works_on_1d_data(self):
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64))
        for p in range(sdata.shape[0]):
            shift = [(p / (sdata.shape[0] - 1) * -3.4)]
            sdata[p, ...] = Core.function_shift(xdata, shift).data
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))
        shifts = Core.function_sequence_register_translation(sxdata, 100, True).data
        self.assertEqual(shifts.shape, (sdata.shape[0] - 1, 1))
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][0], 1 / (sdata.shape[0] - 1) * 3.4, 1)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[0], 3.4, 0)

    def test_sequence_align_works_on_2d_data_without_errors(self):
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64, 64))
        for p in range(sdata.shape[0]):
            shift = (p / (sdata.shape[0] - 1) * -3.4, p / (sdata.shape[0] - 1) * 1.2)
            sdata[p, ...] = Core.function_shift(xdata, shift).data
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        aligned_sxdata = Core.function_sequence_align(sxdata, 100)
        shifts = Core.function_sequence_register_translation(aligned_sxdata, 100, True).data
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0)
        self.assertAlmostEqual(shifts_total[1], 0.0)
        numpy.random.set_state(random_state)

    def test_sequence_align_works_on_1d_data_without_errors(self):
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64))
        for p in range(sdata.shape[0]):
            shift = [(p / (sdata.shape[0] - 1) * -3.4)]
            sdata[p, ...] = Core.function_shift(xdata, shift).data
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))
        aligned_sxdata = Core.function_sequence_align(sxdata, 100)
        shifts = Core.function_sequence_register_translation(aligned_sxdata, 100, True).data
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0)
        numpy.random.set_state(random_state)

    def test_resize_works_to_make_one_dimension_larger_and_one_smaller(self):
        data = numpy.random.randn(64, 64)
        c0 = Calibration.Calibration(offset=1, scale=2)
        c1 = Calibration.Calibration(offset=1, scale=2)
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_resize(xdata, (60, 68))
        self.assertEqual(xdata2.data_shape, (60, 68))
        self.assertTrue(numpy.array_equal(xdata2.data[:, 0:2], numpy.full((60, 2), numpy.mean(data))))
        self.assertTrue(numpy.array_equal(xdata2.data[:, -2:], numpy.full((60, 2), numpy.mean(data))))
        self.assertTrue(numpy.array_equal(xdata2.data[:, 2:-2], xdata.data[2:-2, :]))
        self.assertEqual(xdata.dimensional_calibrations[0].convert_to_calibrated_value(2), xdata2.dimensional_calibrations[0].convert_to_calibrated_value(0))
        self.assertEqual(xdata.dimensional_calibrations[1].convert_to_calibrated_value(0), xdata2.dimensional_calibrations[1].convert_to_calibrated_value(2))

    def test_resize_works_to_make_one_dimension_larger_and_one_smaller_with_odd_dimensions(self):
        data = numpy.random.randn(65, 67)
        c0 = Calibration.Calibration(offset=1, scale=2)
        c1 = Calibration.Calibration(offset=1, scale=2)
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_resize(xdata, (61, 70))
        self.assertEqual(xdata2.data_shape, (61, 70))
        self.assertEqual(xdata.dimensional_calibrations[0].convert_to_calibrated_value(2), xdata2.dimensional_calibrations[0].convert_to_calibrated_value(0))
        self.assertEqual(xdata.dimensional_calibrations[1].convert_to_calibrated_value(0), xdata2.dimensional_calibrations[1].convert_to_calibrated_value(2))

    def test_squeeze_removes_datum_dimension(self):
        # first dimension
        data = numpy.random.randn(1, 4)
        c0 = Calibration.Calibration(offset=1, scale=2, units="a")
        c1 = Calibration.Calibration(offset=1, scale=2, units="b")
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_squeeze(xdata)
        self.assertEqual(xdata2.data_shape, (4, ))
        self.assertEqual(xdata2.dimensional_calibrations[0].units, "b")
        self.assertEqual(xdata2.datum_dimension_count, 1)
        # second dimension
        data = numpy.random.randn(5, 1)
        c0 = Calibration.Calibration(offset=1, scale=2, units="a")
        c1 = Calibration.Calibration(offset=1, scale=2, units="b")
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_squeeze(xdata)
        self.assertEqual(xdata2.data_shape, (5, ))
        self.assertEqual(xdata2.dimensional_calibrations[0].units, "a")
        self.assertEqual(xdata2.datum_dimension_count, 1)

    def test_squeeze_removes_collection_dimension(self):
        # first dimension
        data = numpy.random.randn(1, 4, 3)
        c0 = Calibration.Calibration(offset=1, scale=2, units="a")
        c1 = Calibration.Calibration(offset=1, scale=2, units="b")
        c3 = Calibration.Calibration(offset=1, scale=2, units="c")
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1, c3], data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))
        xdata2 = Core.function_squeeze(xdata)
        self.assertEqual(xdata2.data_shape, (4, 3))
        self.assertEqual(xdata2.dimensional_calibrations[0].units, "b")
        self.assertEqual(xdata2.dimensional_calibrations[1].units, "c")
        self.assertEqual(xdata2.collection_dimension_count, 1)
        self.assertEqual(xdata2.datum_dimension_count, 1)
        # second dimension
        data = numpy.random.randn(5, 1, 6)
        c0 = Calibration.Calibration(offset=1, scale=2, units="a")
        c1 = Calibration.Calibration(offset=1, scale=2, units="b")
        c3 = Calibration.Calibration(offset=1, scale=2, units="c")
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1, c3], data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))
        xdata2 = Core.function_squeeze(xdata)
        self.assertEqual(xdata2.data_shape, (5, 6))
        self.assertEqual(xdata2.dimensional_calibrations[0].units, "a")
        self.assertEqual(xdata2.dimensional_calibrations[1].units, "c")
        self.assertEqual(xdata2.collection_dimension_count, 1)
        self.assertEqual(xdata2.datum_dimension_count, 1)

    def test_squeeze_removes_sequence_dimension(self):
        data = numpy.random.randn(1, 4, 3)
        c0 = Calibration.Calibration(offset=1, scale=2, units="a")
        c1 = Calibration.Calibration(offset=1, scale=2, units="b")
        c3 = Calibration.Calibration(offset=1, scale=2, units="c")
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1, c3], data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        xdata2 = Core.function_squeeze(xdata)
        self.assertEqual(xdata2.data_shape, (4, 3))
        self.assertEqual(xdata2.dimensional_calibrations[0].units, "b")
        self.assertEqual(xdata2.dimensional_calibrations[1].units, "c")
        self.assertFalse(xdata2.is_sequence)
        self.assertEqual(xdata2.datum_dimension_count, 2)

    def test_auto_correlation_keeps_calibration(self):
        # configure dimensions so that the pixels go from -16S to 16S
        dimensional_calibrations = [Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S")]
        xdata = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        result = Core.function_autocorrelate(xdata)
        self.assertIsNot(dimensional_calibrations, result.dimensional_calibrations)  # verify
        self.assertEqual(dimensional_calibrations, result.dimensional_calibrations)

    def test_cross_correlation_keeps_calibration(self):
        # configure dimensions so that the pixels go from -16S to 16S
        dimensional_calibrations = [Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S")]
        xdata1 = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        xdata2 = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        result = Core.function_crosscorrelate(xdata1, xdata2)
        self.assertIsNot(dimensional_calibrations, result.dimensional_calibrations)  # verify
        self.assertEqual(dimensional_calibrations, result.dimensional_calibrations)

    def test_histogram_calibrates_x_axis(self):
        dimensional_calibrations = [Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S")]
        intensity_calibration = Calibration.Calibration(2, 3, units="L")
        data = numpy.ones((16, 16), numpy.uint32)
        data[:2, :2] = 4
        data[-2:, -2:] = 8
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
        result = Core.function_histogram(xdata, 16)
        self.assertEqual(1, len(result.dimensional_calibrations))
        x_calibration = result.dimensional_calibrations[-1]
        self.assertEqual(x_calibration.units, intensity_calibration.units)
        self.assertEqual(result.intensity_calibration, Calibration.Calibration())
        self.assertEqual(5, x_calibration.convert_to_calibrated_value(0))
        self.assertEqual(26, x_calibration.convert_to_calibrated_value(16))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
