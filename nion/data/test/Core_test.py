# standard libraries
import copy
import io
import logging
import math
import os
import typing
import unittest

# third party libraries
import h5py
import numpy
import scipy
import scipy.ndimage

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.data.DataAndMetadata import _ImageDataType


class TestCore(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_line_profile_uses_integer_coordinates(self) -> None:
        data = numpy.zeros((32, 32))
        data[16, 15] = 1
        data[16, 16] = 1
        data[16, 17] = 1
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        line_profile_data = Core.function_line_profile(xdata, ((8/32, 16/32), (24/32, 16/32)), 1.0)._data_ex
        self.assertTrue(numpy.array_equal(line_profile_data, data[8:24, 16]))
        line_profile_data = Core.function_line_profile(xdata, ((8/32 + 1/128, 16/32 + 1/128), (24/32 + 2/128, 16/32 + 2/128)), 1.0)._data_ex
        self.assertTrue(numpy.array_equal(line_profile_data, data[8:24, 16]))
        line_profile_xdata = Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertTrue(numpy.array_equal(line_profile_xdata._data_ex, data[8:24, 16] * 3))

    def test_line_profile_width_adjusts_intensity_calibration(self) -> None:
        data = numpy.zeros((32, 32))
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        line_profile_xdata = Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertAlmostEqual(line_profile_xdata.intensity_calibration.scale, 1/3)

    def test_line_profile_width_computation_does_not_affect_source_intensity(self) -> None:
        data = numpy.zeros((32, 32))
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="e"))
        Core.function_line_profile(xdata, ((8 / 32, 16 / 32), (24 / 32, 16 / 32)), 3.0)
        self.assertAlmostEqual(xdata.intensity_calibration.scale, 1)

    def test_line_profile_produces_appropriate_data_type(self) -> None:
        # valid for 'nearest' mode only. ignores overflow issues.
        vector = (0.1, 0.2), (0.3, 0.4)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.int32)), vector, 3.0).data_dtype, numpy.int32)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.uint32)), vector, 3.0).data_dtype, numpy.uint32)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.float32)), vector, 3.0).data_dtype, numpy.float32)
        self.assertEqual(Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.float64)), vector, 3.0).data_dtype, numpy.float64)

    def test_line_profile_accepts_complex_data(self) -> None:
        if tuple(map(int, (scipy.version.version.split(".")))) > (1, 6):
            vector = (0.1, 0.2), (0.3, 0.4)
            Core.function_line_profile(DataAndMetadata.new_data_and_metadata(numpy.zeros((32, 32), numpy.complex128)), vector, 3.0)

    def test_fft_produces_correct_calibration(self) -> None:
        src_data = ((numpy.abs(numpy.random.randn(16, 16)) + 1) * 10).astype(numpy.float32)
        dimensional_calibrations = (Calibration.Calibration(offset=3), Calibration.Calibration(offset=2))
        a = DataAndMetadata.DataAndMetadata.from_data(src_data, dimensional_calibrations=dimensional_calibrations)
        fft = Core.function_fft(a)
        self.assertAlmostEqual(fft.dimensional_calibrations[0].offset, -0.5 - 1/32)
        self.assertAlmostEqual(fft.dimensional_calibrations[1].offset, -0.5 - 1/32)
        ifft = Core.function_ifft(fft)
        self.assertAlmostEqual(ifft.dimensional_calibrations[0].offset, 0.0)
        self.assertAlmostEqual(ifft.dimensional_calibrations[1].offset, 0.0)

    def test_fft_forward_and_back_is_consistent(self) -> None:
        d = numpy.zeros((256, 256))
        src = Core.function_squeeze(Core.radius(d))
        fft = Core.function_fft(src)
        ifft = Core.function_ifft(fft)
        # error increases for size of data
        self.assertLess(numpy.amax(numpy.absolute(src._data_ex - ifft._data_ex)), 1E-11)
        self.assertLess(numpy.absolute(numpy.sum(src._data_ex - ifft._data_ex)), 1E-11)

    def test_fft_1d_forward_and_back_is_consistent(self) -> None:
        d = numpy.zeros((256, 1))
        src = Core.function_squeeze(Core.radius(d)) + numpy.array(range(d.shape[0]))
        fft = Core.function_fft(src)
        ifft = Core.function_ifft(fft)
        # error increases for size of data
        self.assertLess(numpy.amax(numpy.absolute(src._data_ex - ifft._data_ex)), 1E-11)
        self.assertLess(numpy.absolute(numpy.sum(src._data_ex - ifft._data_ex)), 1E-11)

    def test_fft_rms_is_same_as_original(self) -> None:
        d = numpy.random.randn(256, 256)
        src_data = Core.radius(d)
        fft = Core.function_fft(src_data)
        src_data_2 = fft.data
        self.assertLess(numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data)))) - numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data_2)))), 1E-12)  # type: ignore

    def test_fft_1d_rms_is_same_as_original(self) -> None:
        d = numpy.random.randn(256, 1)
        src_data = Core.function_squeeze(Core.radius(d))
        fft = Core.function_fft(src_data)
        src_data_2 = fft._data_ex
        self.assertLess(numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data)))) - numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(src_data_2)))), 1E-12)

    def test_concatenate_works_with_1d_inputs(self) -> None:
        src_data1 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float32)
        src_data2 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float32)
        dimensional_calibrations = [Calibration.Calibration(offset=3)]
        a1 = DataAndMetadata.DataAndMetadata.from_data(src_data1, dimensional_calibrations=dimensional_calibrations)
        a2 = DataAndMetadata.DataAndMetadata.from_data(src_data2, dimensional_calibrations=dimensional_calibrations)
        c0 = Core.function_concatenate([a1, a2], 0)
        self.assertEqual(tuple(c0._data_ex.shape), tuple(c0.data_shape))
        self.assertTrue(numpy.array_equal(c0._data_ex, numpy.concatenate([src_data1, src_data2], 0)))  # type: ignore

    def test_concatenate_propagates_data_descriptor(self) -> None:
        data1 = numpy.ones((16, 32))
        data2 = numpy.ones((8, 32))

        data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 1)
        xdata1 = DataAndMetadata.new_data_and_metadata(data1, data_descriptor=data_descriptor)
        xdata2 = DataAndMetadata.new_data_and_metadata(data2, data_descriptor=data_descriptor)
        concatenated = Core.function_concatenate([xdata1, xdata2])
        self.assertTrue(concatenated.is_sequence)
        self.assertFalse(concatenated.is_collection)
        self.assertEqual(concatenated.datum_dimension_count, 1)

        data_descriptor = DataAndMetadata.DataDescriptor(False, 1, 1)
        xdata1 = DataAndMetadata.new_data_and_metadata(data1, data_descriptor=data_descriptor)
        xdata2 = DataAndMetadata.new_data_and_metadata(data2, data_descriptor=data_descriptor)
        concatenated = Core.function_concatenate([xdata1, xdata2])
        self.assertFalse(concatenated.is_sequence)
        self.assertTrue(concatenated.is_collection)
        self.assertEqual(concatenated.datum_dimension_count, 1)

        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
        xdata1 = DataAndMetadata.new_data_and_metadata(data1, data_descriptor=data_descriptor)
        xdata2 = DataAndMetadata.new_data_and_metadata(data2, data_descriptor=data_descriptor)
        concatenated = Core.function_concatenate([xdata1, xdata2])
        self.assertFalse(concatenated.is_sequence)
        self.assertFalse(concatenated.is_collection)
        self.assertEqual(concatenated.datum_dimension_count, 2)

    def test_concatenate_calibrations(self) -> None:
        src_data1 = numpy.zeros((4, 8, 16))
        src_data2 = numpy.zeros((4, 8, 16))
        dimensional_calibrations = (Calibration.Calibration(units="a"), Calibration.Calibration(units="b"), Calibration.Calibration(units="c"))
        a1 = DataAndMetadata.DataAndMetadata.from_data(src_data1, dimensional_calibrations=dimensional_calibrations)
        a2 = DataAndMetadata.DataAndMetadata.from_data(src_data2, dimensional_calibrations=dimensional_calibrations)
        vstack = Core.function_concatenate([a1, a2], axis=0)
        self.assertEqual("a", vstack.dimensional_calibrations[0].units)
        self.assertEqual("b", vstack.dimensional_calibrations[1].units)
        self.assertEqual("c", vstack.dimensional_calibrations[2].units)
        vstack = Core.function_concatenate([a1[0:2], a2[2:4]], axis=0)
        self.assertFalse(vstack.dimensional_calibrations[0].units)
        self.assertEqual("b", vstack.dimensional_calibrations[1].units)
        self.assertEqual("c", vstack.dimensional_calibrations[2].units)

    def test_vstack_and_hstack_work_with_1d_inputs(self) -> None:
        src_data1 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float32)
        src_data2 = ((numpy.abs(numpy.random.randn(16)) + 1) * 10).astype(numpy.float32)
        dimensional_calibrations = [Calibration.Calibration(offset=3)]
        a1 = DataAndMetadata.DataAndMetadata.from_data(src_data1, dimensional_calibrations=dimensional_calibrations)
        a2 = DataAndMetadata.DataAndMetadata.from_data(src_data2, dimensional_calibrations=dimensional_calibrations)
        vstack = Core.function_vstack([a1, a2])
        self.assertEqual(tuple(vstack._data_ex.shape), tuple(vstack.data_shape))
        self.assertTrue(numpy.array_equal(vstack._data_ex, numpy.vstack([src_data1, src_data2])))
        hstack = Core.function_hstack([a1, a2])
        self.assertEqual(tuple(hstack._data_ex.shape), tuple(hstack.data_shape))
        self.assertTrue(numpy.array_equal(hstack._data_ex, numpy.hstack([src_data1, src_data2])))

    def test_sum_over_two_axes_returns_correct_shape(self) -> None:
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((4, 4, 16)))
        dst = Core.function_sum(src, (0, 1))
        self.assertEqual(dst.data_shape, dst._data_ex.shape)

    def test_sum_over_two_axes_returns_correct_calibrations(self) -> None:
        dimensional_calibrations = [
            Calibration.Calibration(1, 11, "one"),
            Calibration.Calibration(2, 22, "two"),
            Calibration.Calibration(3, 33, "three"),
        ]
        src = DataAndMetadata.new_data_and_metadata(numpy.ones((4, 4, 16)), dimensional_calibrations=dimensional_calibrations)
        dst = Core.function_sum(src, 2)
        self.assertEqual(2, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[0], dst.dimensional_calibrations[0])
        self.assertEqual(dimensional_calibrations[1], dst.dimensional_calibrations[1])
        dst = Core.function_sum(src, (0, 1))
        self.assertEqual(1, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[2], dst.dimensional_calibrations[0])
        dst = Core.function_sum(src, -1)
        self.assertEqual(2, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[0], dst.dimensional_calibrations[0])
        self.assertEqual(dimensional_calibrations[1], dst.dimensional_calibrations[1])

    def test_mean_over_two_axes_returns_correct_calibrations(self) -> None:
        dimensional_calibrations = [
            Calibration.Calibration(1, 11, "one"),
            Calibration.Calibration(2, 22, "two"),
            Calibration.Calibration(3, 33, "three"),
        ]
        src = DataAndMetadata.new_data_and_metadata(numpy.ones((4, 4, 16)), dimensional_calibrations=dimensional_calibrations)
        dst = Core.function_mean(src, 2)
        self.assertEqual(2, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[0], dst.dimensional_calibrations[0])
        self.assertEqual(dimensional_calibrations[1], dst.dimensional_calibrations[1])
        dst = Core.function_mean(src, (0, 1))
        self.assertEqual(1, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[2], dst.dimensional_calibrations[0])
        dst = Core.function_mean(src, -1)
        self.assertEqual(2, len(dst.dimensional_calibrations))
        self.assertEqual(dimensional_calibrations[0], dst.dimensional_calibrations[0])
        self.assertEqual(dimensional_calibrations[1], dst.dimensional_calibrations[1])

    def test_sum_over_rgb_produces_correct_data(self) -> None:
        data: numpy.typing.NDArray[numpy.uint8] = numpy.zeros((3, 3, 4), numpy.uint8)
        data[1, 0] = (3, 3, 3, 3)
        src = DataAndMetadata.DataAndMetadata.from_data(data)
        dst0 = Core.function_sum(src, 0)
        dst1 = Core.function_sum(src, 1)
        self.assertEqual(dst0.data_shape, dst0._data_ex.shape)
        self.assertEqual(dst1.data_shape, dst1._data_ex.shape)
        self.assertTrue(numpy.array_equal(dst0._data_ex[0], (1, 1, 1, 1)))
        self.assertTrue(numpy.array_equal(dst0._data_ex[1], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst0._data_ex[2], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst1._data_ex[0], (0, 0, 0, 0)))
        self.assertTrue(numpy.array_equal(dst1._data_ex[1], (1, 1, 1, 1)))
        self.assertTrue(numpy.array_equal(dst1._data_ex[2], (0, 0, 0, 0)))

    def test_fourier_filter_gives_sensible_units_when_source_has_units(self) -> None:
        dimensional_calibrations = [Calibration.Calibration(units="mm"), Calibration.Calibration(units="mm")]
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((32, 32)), dimensional_calibrations=dimensional_calibrations)
        dst = Core.function_ifft(Core.function_fft(src))
        self.assertEqual(dst.dimensional_calibrations[0].units, "mm")
        self.assertEqual(dst.dimensional_calibrations[1].units, "mm")

    def test_fourier_filter_gives_sensible_units_when_source_has_no_units(self) -> None:
        src = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((32, 32)))
        dst = Core.function_ifft(Core.function_fft(src))
        self.assertEqual(dst.dimensional_calibrations[0].units, "")
        self.assertEqual(dst.dimensional_calibrations[1].units, "")

    def test_fourier_mask_works_with_all_dimensions(self) -> None:
        dimension_list = [(32, 32), (31, 30), (30, 31), (31, 31), (32, 31), (31, 32)]
        for h, w in dimension_list:
            data = DataAndMetadata.DataAndMetadata.from_data(numpy.random.randn(h, w))
            mask = DataAndMetadata.DataAndMetadata.from_data(typing.cast(_ImageDataType, numpy.random.randn(h, w) > 0).astype(numpy.float32))
            fft = Core.function_fft(data)
            masked_data = Core.function_ifft(Core.function_fourier_mask(fft, mask))._data_ex
            self.assertAlmostEqual(numpy.sum(numpy.imag(masked_data)), 0)  # type: ignore

    def test_slice_sum_grabs_signal_index(self) -> None:
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        slice = Core.function_slice_sum(data_and_metadata, 2, 2)
        self.assertTrue(numpy.array_equal(numpy.sum(random_data[..., 1:3], 2), slice._data_ex))
        self.assertEqual(slice.dimensional_shape, random_data.shape[0:2])
        self.assertEqual(slice.intensity_calibration, c0)
        self.assertEqual(slice.dimensional_calibrations[0], c1)
        self.assertEqual(slice.dimensional_calibrations[1], c2)

    def test_pick_grabs_datum_index_from_3d(self) -> None:
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[2, 1, :], pick._data_ex))
        self.assertEqual(pick.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], c3)

    def test_pick_grabs_datum_index_from_sequence_of_3d(self) -> None:
        random_data = numpy.random.randn(2, 3, 4, 5)
        cs = Calibration.Calibration(units="s")
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[cs, c1, c2, c3], data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))  # last index is signal
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[:, 2, 1, :], pick._data_ex))
        self.assertSequenceEqual(pick.dimensional_shape, (random_data.shape[0], random_data.shape[-1]))
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], cs)
        self.assertEqual(pick.dimensional_calibrations[1], c3)

    def test_pick_grabs_datum_index_from_4d(self) -> None:
        random_data = numpy.random.randn(3, 4, 5, 6)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        c4 = Calibration.Calibration(units="e")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3, c4], data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[2, 1, ...], pick._data_ex))
        self.assertEqual(pick.dimensional_shape, random_data.shape[2:4])
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], c3)
        self.assertEqual(pick.dimensional_calibrations[1], c4)

    def test_pick_grabs_datum_index_from_sequence_of_4d(self) -> None:
        random_data = numpy.random.randn(2, 3, 4, 5, 6)
        cs = Calibration.Calibration(units="s")
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        c4 = Calibration.Calibration(units="e")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[cs, c1, c2, c3, c4], data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))
        pick = Core.function_pick(data_and_metadata, (2/3, 1/4))
        self.assertTrue(numpy.array_equal(random_data[:, 2, 1, ...], pick._data_ex))
        self.assertSequenceEqual(pick.dimensional_shape, (random_data.shape[0], random_data.shape[3], random_data.shape[4]))
        self.assertEqual(pick.intensity_calibration, c0)
        self.assertEqual(pick.dimensional_calibrations[0], cs)
        self.assertEqual(pick.dimensional_calibrations[1], c3)
        self.assertEqual(pick.dimensional_calibrations[2], c4)

    def test_sum_region_produces_correct_result(self) -> None:
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        mask_data: numpy.typing.NDArray[numpy.int32] = numpy.zeros((3, 4), numpy.int32)
        mask_data[0, 1] = 1
        mask_data[2, 2] = 1
        mask = DataAndMetadata.new_data_and_metadata(mask_data)
        sum_region = Core.function_sum_region(data, mask)
        self.assertTrue(numpy.array_equal(random_data[0, 1, :] + random_data[2, 2, :], sum_region._data_ex))
        self.assertEqual(sum_region.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(sum_region.intensity_calibration, c0)
        self.assertEqual(sum_region.dimensional_calibrations[0], c3)

    def test_sum_region_produces_correct_result_for_sequence(self) -> None:
        random_data = numpy.random.randn(2, 3, 4, 5)
        cs = Calibration.Calibration(units="s")
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[cs, c1, c2, c3], data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))  # last index is signal
        mask_data: numpy.typing.NDArray[numpy.int32] = numpy.zeros((3, 4), numpy.int32)
        mask_data[0, 1] = 1
        mask_data[2, 2] = 1
        mask = DataAndMetadata.new_data_and_metadata(mask_data)
        sum_region = Core.function_sum_region(data, mask)
        self.assertTrue(numpy.array_equal(random_data[:, 0, 1, :] + random_data[:, 2, 2, :], sum_region._data_ex))
        self.assertEqual(sum_region.dimensional_shape, (random_data.shape[0], random_data.shape[-1]))
        self.assertEqual(sum_region.intensity_calibration, c0)
        self.assertEqual(sum_region.dimensional_calibrations[0], cs)
        self.assertEqual(sum_region.dimensional_calibrations[1], c3)

    def test_average_region_produces_correct_result(self) -> None:
        random_data = numpy.random.randn(3, 4, 5)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2, c3])  # last index is signal
        mask_data: numpy.typing.NDArray[numpy.int32] = numpy.zeros((3, 4), numpy.int32)
        mask_data[0, 1] = 1
        mask_data[2, 2] = 1
        mask = DataAndMetadata.new_data_and_metadata(mask_data)
        average_region = Core.function_average_region(data, mask)
        self.assertTrue(numpy.array_equal((random_data[0, 1, :] + random_data[2, 2, :])/2, average_region._data_ex))
        self.assertEqual(average_region.dimensional_shape, (random_data.shape[-1],))
        self.assertEqual(average_region.intensity_calibration, c0)
        self.assertEqual(average_region.dimensional_calibrations[0], c3)

    def test_average_region_produces_correct_result_for_sequence(self) -> None:
        random_data = numpy.random.randn(2, 3, 4, 5)
        cs = Calibration.Calibration(units="s")
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        c3 = Calibration.Calibration(units="d")
        data = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[cs, c1, c2, c3], data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))  # last index is signal
        mask_data: numpy.typing.NDArray[numpy.int32] = numpy.zeros((3, 4), numpy.int32)
        mask_data[0, 1] = 1
        mask_data[2, 2] = 1
        mask = DataAndMetadata.new_data_and_metadata(mask_data)
        average_region = Core.function_average_region(data, mask)
        self.assertTrue(numpy.array_equal((random_data[:, 0, 1, :] + random_data[:, 2, 2, :])/2, average_region._data_ex))
        self.assertEqual(average_region.dimensional_shape, (random_data.shape[0], random_data.shape[-1]))
        self.assertEqual(average_region.intensity_calibration, c0)
        self.assertEqual(average_region.dimensional_calibrations[0], cs)
        self.assertEqual(average_region.dimensional_calibrations[1], c3)

    def test_slice_sum_works_on_2d_data(self) -> None:
        random_data = numpy.random.randn(4, 10)
        c0 = Calibration.Calibration(units="a")
        c1 = Calibration.Calibration(units="b")
        c2 = Calibration.Calibration(units="c")
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data, intensity_calibration=c0, dimensional_calibrations=[c1, c2])  # last index is signal
        result = Core.function_slice_sum(data_and_metadata, 5, 3)
        self.assertTrue(numpy.array_equal(numpy.sum(random_data[..., 4:7], -1), result._data_ex))
        self.assertEqual(result.intensity_calibration, data_and_metadata.intensity_calibration)
        self.assertEqual(result.dimensional_calibrations[0], data_and_metadata.dimensional_calibrations[0])

    def test_fft_works_on_rgba_data(self) -> None:
        random_data = numpy.random.randint(0, 256, (32, 32, 4), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        Core.function_fft(data_and_metadata)

    def test_display_data_2d_not_a_view(self) -> None:
        random_data = numpy.random.randint(0, 256, (2, 2), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        display_xdata = Core.function_display_data(data_and_metadata)
        assert display_xdata
        display_xdata_copy = copy.deepcopy(display_xdata)
        data_and_metadata._data_ex[:] = 0
        self.assertTrue(numpy.array_equal(display_xdata._data_ex, display_xdata_copy._data_ex))

    def test_display_rgba_with_1d_rgba(self) -> None:
        random_data = numpy.random.randint(0, 256, (32, 4), numpy.uint8)
        data_and_metadata = DataAndMetadata.new_data_and_metadata(random_data)
        Core.function_display_rgba(data_and_metadata)

    def test_create_rgba_image_from_uint16(self) -> None:
        image = numpy.mgrid[22000:26096:256, 0:16][0].astype(numpy.uint16)
        display_rgba = Core.function_display_rgba(DataAndMetadata.new_data_and_metadata(image), display_range=(22000, 26096))
        assert display_rgba
        image_rgb = display_rgba._data_ex
        # image_rgb = Image.create_rgba_image_from_array(image, display_limits=(22000, 26096))
        self.assertGreater(image_rgb[15, 15], image_rgb[0, 0])

    def test_create_display_from_rgba_sequence_should_work(self) -> None:
        data: _ImageDataType = typing.cast(_ImageDataType, numpy.random.rand(4, 64, 64, 3) * 255).astype(numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        display_data, modified = Core.function_display_data_no_copy(xdata, 0)
        self.assertIsNotNone(display_data)
        self.assertTrue(modified)

    def test_ability_to_take_1d_slice_with_newaxis(self) -> None:
        data = numpy.random.rand(64)
        xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 1))
        self.assertTrue(numpy.array_equal(data[..., numpy.newaxis], xdata[..., numpy.newaxis]))

    def test_slice_of_2d_works(self) -> None:
        data = numpy.random.rand(64, 64)
        xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 2))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1, ...]))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1]))
        # slicing out a single value is not yet supported
        # self.assertTrue(numpy.array_equal(data[1, 2], xdata[1, 2]))

    def test_slice_of_sequence_works(self) -> None:
        data = numpy.random.rand(4, 64, 64)
        xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1, ...]))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1]))
        self.assertTrue(numpy.array_equal(data[1, 30, ...], xdata[1, 30, ...]))
        self.assertTrue(numpy.array_equal(data[1, 30, ...], xdata[1, 30]))
        # slicing out a single value is not yet supported
        # self.assertTrue(numpy.array_equal(data[1, 30, 20], xdata[1, 30, 20]))

    def test_rgb_slice_of_sequence_works(self) -> None:
        data: _ImageDataType = typing.cast(_ImageDataType, numpy.random.rand(4, 64, 64, 3) * 255).astype(numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1, ...]))
        self.assertTrue(numpy.array_equal(data[1, ...], xdata[1]))
        self.assertTrue(numpy.array_equal(data[1, 30, ...], xdata[1, 30, ...]))
        self.assertTrue(numpy.array_equal(data[1, 30, ...], xdata[1, 30]))
        # slicing out a single value is not yet supported
        # self.assertTrue(numpy.array_equal(data[1, 30, 20], xdata[1, 30, 20]))

    def test_align_works_on_2d_data(self) -> None:
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4, 1.2)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        self.assertAlmostEqual(shift[1], measured_shift[1], delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)

    def test_align_with_bounds_works_on_2d_data(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64, 64)
        data[10:20, 10:20] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4, 1.2)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        xdata._data_ex[40:50, 40:50] += 100
        xdata_shifted._data_ex[40:50, 40:50] += 10
        bounds = ((5/64, 5/64), (20/64, 20/64))
        measured_shift = Core.function_register(xdata_shifted, xdata, True, bounds=bounds)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        self.assertAlmostEqual(shift[1], measured_shift[1], delta=0.5)
        # Now test that without bounds we find no shift (because the more intense feature does not shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True, bounds=None)
        self.assertAlmostEqual(measured_shift[0], 0, delta=0.5)
        self.assertAlmostEqual(measured_shift[1], 0, delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted, bounds=bounds) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)
        numpy.random.set_state(random_state)

    def test_align_works_on_1d_data(self) -> None:
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4,)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)

    def test_shift_nx1_data_produces_nx1_data(self) -> None:
        data = numpy.random.randn(64)
        data[30:40,] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift: typing.Tuple[float, ...] = (-3.4, )
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        self.assertEqual(xdata.data_shape, xdata_shifted.data_shape)

        data = numpy.random.randn(64, 1)
        data[30:40, 0] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4, 0.0)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        self.assertEqual(xdata.data_shape, xdata_shifted.data_shape)

        data = numpy.random.randn(1, 64)
        data[0, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (0.0, -3.4)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        self.assertEqual(xdata.data_shape, xdata_shifted.data_shape)

    def test_align_works_on_nx1_data(self) -> None:
        data = numpy.random.randn(64, 1)
        data[30:40, 0] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4, 0.0)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        self.assertAlmostEqual(shift[1], measured_shift[1], delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)

        data = numpy.random.randn(1, 64)
        data[0, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (0.0, -3.4)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        self.assertAlmostEqual(shift[1], measured_shift[1], delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)

    def test_measure_works_on_navigable_data(self) -> None:
        sequence_collection_shapes = (
            (10, ()),
            (0, (10,)),
            (4, (4,)),
            (0, (4, 4)),
            (4, (4, 4))
        )
        data_shapes = (
            (64,),
            (1, 64),
            (64, 1),
            (16, 16)
        )
        shapes = (tuple(list(scs) + [ds]) for scs in sequence_collection_shapes for ds in data_shapes)
        for sequence_len_o, collection_shape_o, data_shape_o in shapes:
            # print(f"{sequence_len}, {collection_shape}, {data_shape}")
            data_shape = typing.cast(typing.Tuple[int, ...], data_shape_o)
            collection_shape = typing.cast(typing.Tuple[int, ...], collection_shape_o)
            sequence_len = typing.cast(int, sequence_len_o)
            s_shape = (sequence_len, *collection_shape) if sequence_len else collection_shape
            sequence_data = numpy.zeros((s_shape + data_shape))
            sequence_xdata = DataAndMetadata.new_data_and_metadata(sequence_data, data_descriptor=DataAndMetadata.DataDescriptor(sequence_len > 0, len(collection_shape), len(data_shape)))
            sequence_xdata = Core.function_squeeze(sequence_xdata)
            random_state = numpy.random.get_state()
            numpy.random.seed(1)
            data = numpy.random.randn(*data_shape)
            numpy.random.set_state(random_state)
            d_index = [slice(30, 40) for _ in range(len(data_shape))]
            data[tuple(d_index)] += 10
            xdata = DataAndMetadata.new_data_and_metadata(data)
            s_total = numpy.product(s_shape).item()
            for i in range(s_total):
                ii = numpy.unravel_index(i, s_shape)
                shift = 3.5 * i / s_total
                # construct shifts so that it is shifting the first data dimension where the dimension length > 1
                shifts = list()
                for dd in range(len(data_shape)):
                    if data_shape[dd] > 1:
                        shifts.append(shift)
                        shift = 0.0
                    else:
                        shifts.append(0.0)
                sequence_data[ii] = Core.function_fourier_shift(xdata, tuple(shifts))
            unravel_index = typing.cast(DataAndMetadata._SliceKeyType, numpy.unravel_index(0, sequence_xdata.navigation_dimension_shape))
            measured = Core.function_sequence_measure_relative_translation(sequence_xdata, sequence_xdata[unravel_index], False)
            self.assertEqual(sequence_xdata.is_sequence, measured.is_sequence)
            self.assertEqual(sequence_xdata.collection_dimension_shape, measured.collection_dimension_shape)
            self.assertEqual(1, measured.datum_dimension_count)
            self.assertAlmostEqual(0.0, numpy.amin(-measured), places=1)
            s_max = numpy.product(s_shape).item()
            expected_max = 3.5 * (s_max - 1) / s_max
            self.assertAlmostEqual(expected_max, numpy.amax(-measured).item(), delta=0.5)
            measured_squeezed = Core.function_squeeze_measurement(measured)

    def test_align_with_bounds_works_on_1d_data(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64)
        data[10:20] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        shift = (-3.4,)
        xdata_shifted = Core.function_fourier_shift(xdata, shift)
        xdata._data_ex[40:50] += 100
        xdata_shifted._data_ex[40:50] += 100
        bounds = (5/64, 25/64)
        measured_shift = Core.function_register(xdata_shifted, xdata, True, bounds=bounds)
        self.assertAlmostEqual(shift[0], measured_shift[0], delta=0.5)
        # Now test that without bounds we find no shift (because the more intense feature does not shift)
        measured_shift = Core.function_register(xdata_shifted, xdata, True, bounds=None)
        self.assertAlmostEqual(measured_shift[0], 0, delta=0.5)
        result = Core.function_fourier_align(data, xdata_shifted, bounds=bounds) - xdata_shifted
        self.assertAlmostEqual(result._data_ex.mean(), 0)
        numpy.random.set_state(random_state)

    def test_sequence_register_works_on_2d_data(self) -> None:
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64, 64))
        for p in range(sdata.shape[0]):
            shift = (p / (sdata.shape[0] - 1) * -3.4, p / (sdata.shape[0] - 1) * 1.2)
            sdata[p, ...] = Core.function_fourier_shift(xdata, shift)._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        shifts = Core.function_sequence_register_translation(sxdata, True)._data_ex
        self.assertEqual(shifts.shape, (sdata.shape[0], 2))
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][0], 1 / (sdata.shape[0] - 1) * 3.4, delta=0.1)
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][1], 1 / (sdata.shape[0] - 1) * -1.2, delta=0.1)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[0], 3.4, delta=2)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[1], -1.2, delta=2)

    def test_sequence_register_works_on_1d_data(self) -> None:
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64))
        for p in range(sdata.shape[0]):
            shift = [(p / (sdata.shape[0] - 1) * -3.4)]
            sdata[p, ...] = Core.function_fourier_shift(xdata, tuple(shift))._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))
        shifts = Core.function_sequence_register_translation(sxdata, True)._data_ex
        self.assertEqual(shifts.shape, (sdata.shape[0], 1))
        self.assertAlmostEqual(shifts[sdata.shape[0] // 2][0], 1 / (sdata.shape[0] - 1) * 3.4, delta=0.1)
        self.assertAlmostEqual(numpy.sum(shifts, axis=0)[0], 3.4, delta=2)

    def test_sequence_register_produces_correctly_shaped_output_on_2dx1d_data(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((6, 6, 64))
        for p in range(sdata.shape[0]):
            for q in range(sdata.shape[1]):
                shift = [((p + q) / 2 / (sdata.shape[0] - 1) * -3.4)]
                sdata[q, p, ...] = Core.function_shift(xdata, tuple(shift))._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))
        shifts = Core.function_sequence_measure_relative_translation(sxdata, sxdata[0, 0], True)._data_ex
        self.assertEqual(shifts.shape, (6, 6, 1))
        numpy.random.set_state(random_state)

    def test_sequence_register_produces_correctly_shaped_output_on_2dx2d_data(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((6, 6, 64, 64))
        for p in range(sdata.shape[0]):
            for q in range(sdata.shape[1]):
                shift = (p / (sdata.shape[0] - 1) * -3.4, q / (sdata.shape[0] - 1) * 1.2)
                sdata[q, p, ...] = Core.function_shift(xdata, shift)._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))
        shifts = Core.function_sequence_measure_relative_translation(sxdata, sxdata[0, 0], True)._data_ex
        self.assertEqual(shifts.shape, (6, 6, 2))
        numpy.random.set_state(random_state)

    def test_sequence_align_works_on_2d_data_without_errors(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64, 64))
        for p in range(sdata.shape[0]):
            shift = (p / (sdata.shape[0] - 1) * -3.4, p / (sdata.shape[0] - 1) * 1.2)
            sdata[p, ...] = Core.function_fourier_shift(xdata, shift)._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        aligned_sxdata = Core.function_sequence_fourier_align(sxdata)
        shifts = Core.function_sequence_register_translation(aligned_sxdata, True)._data_ex
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0, delta=0.5)
        self.assertAlmostEqual(shifts_total[1], 0.0, delta=0.5)
        numpy.random.set_state(random_state)

    def test_sequence_align_works_on_1d_data_without_errors(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((32, 64))
        for p in range(sdata.shape[0]):
            shift = [(p / (sdata.shape[0] - 1) * -3.4)]
            sdata[p, ...] = Core.function_shift(xdata, tuple(shift))._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))
        aligned_sxdata = Core.function_sequence_align(sxdata)
        shifts = Core.function_sequence_register_translation(aligned_sxdata, True)._data_ex
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0, delta=0.5)
        numpy.random.set_state(random_state)

    def test_sequence_align_works_on_2dx1d_data_without_errors(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64)
        data[30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((6, 6, 64))
        for p in range(sdata.shape[0]):
            for q in range(sdata.shape[1]):
                shift = [((p + q) / 2 / (sdata.shape[0] - 1) * -3.4)]
                sdata[q, p, ...] = Core.function_fourier_shift(xdata, tuple(shift))._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))
        aligned_sxdata = Core.function_sequence_fourier_align(sxdata)
        aligned_sxdata = DataAndMetadata.new_data_and_metadata(aligned_sxdata._data_ex.reshape(36, 64), data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))
        shifts = Core.function_sequence_register_translation(aligned_sxdata, True)._data_ex
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0, places=1)
        numpy.random.set_state(random_state)

    def test_sequence_align_works_on_2dx2d_data_without_errors(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(1)
        data = numpy.random.randn(64, 64)
        data[30:40, 30:40] += 10
        xdata = DataAndMetadata.new_data_and_metadata(data)
        sdata = numpy.empty((6, 6, 64, 64))
        for p in range(sdata.shape[0]):
            for q in range(sdata.shape[1]):
                shift = (p / (sdata.shape[0] - 1) * -3.4, q / (sdata.shape[0] - 1) * 1.2)
                sdata[q, p, ...] = Core.function_fourier_shift(xdata, shift)._data_ex
        sxdata = DataAndMetadata.new_data_and_metadata(sdata, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))
        aligned_sxdata = Core.function_sequence_fourier_align(sxdata)
        aligned_sxdata = DataAndMetadata.new_data_and_metadata(aligned_sxdata._data_ex.reshape(36, 64, 64), data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        shifts = Core.function_sequence_register_translation(aligned_sxdata, True)._data_ex
        shifts_total = numpy.sum(shifts, axis=0)
        self.assertAlmostEqual(shifts_total[0], 0.0, delta=0.5)
        self.assertAlmostEqual(shifts_total[1], 0.0, delta=0.5)
        numpy.random.set_state(random_state)

    def test_resize_works_to_make_one_dimension_larger_and_one_smaller(self) -> None:
        data = numpy.random.randn(64, 64)
        c0 = Calibration.Calibration(offset=1, scale=2)
        c1 = Calibration.Calibration(offset=1, scale=2)
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_resize(xdata, (60, 68))
        self.assertEqual(xdata2.data_shape, (60, 68))
        self.assertTrue(numpy.array_equal(xdata2._data_ex[:, 0:2], numpy.full((60, 2), numpy.mean(data))))
        self.assertTrue(numpy.array_equal(xdata2._data_ex[:, -2:], numpy.full((60, 2), numpy.mean(data))))
        self.assertTrue(numpy.array_equal(xdata2._data_ex[:, 2:-2], xdata._data_ex[2:-2, :]))
        self.assertEqual(xdata.dimensional_calibrations[0].convert_to_calibrated_value(2), xdata2.dimensional_calibrations[0].convert_to_calibrated_value(0))
        self.assertEqual(xdata.dimensional_calibrations[1].convert_to_calibrated_value(0), xdata2.dimensional_calibrations[1].convert_to_calibrated_value(2))

    def test_resize_works_to_make_one_dimension_larger_and_one_smaller_with_odd_dimensions(self) -> None:
        data = numpy.random.randn(65, 67)
        c0 = Calibration.Calibration(offset=1, scale=2)
        c1 = Calibration.Calibration(offset=1, scale=2)
        xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=[c0, c1])
        xdata2 = Core.function_resize(xdata, (61, 70))
        self.assertEqual(xdata2.data_shape, (61, 70))
        self.assertEqual(xdata.dimensional_calibrations[0].convert_to_calibrated_value(2), xdata2.dimensional_calibrations[0].convert_to_calibrated_value(0))
        self.assertEqual(xdata.dimensional_calibrations[1].convert_to_calibrated_value(0), xdata2.dimensional_calibrations[1].convert_to_calibrated_value(2))

    def test_squeeze_removes_datum_dimension(self) -> None:
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

    def test_squeeze_removes_collection_dimension(self) -> None:
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

    def test_squeeze_removes_sequence_dimension(self) -> None:
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

    def test_auto_correlation_keeps_calibration(self) -> None:
        # configure dimensions so that the pixels go from -16S to 16S
        dimensional_calibrations = (Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S"))
        xdata = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        result = Core.function_autocorrelate(xdata)
        self.assertIsNot(dimensional_calibrations, result.dimensional_calibrations)  # verify
        self.assertEqual(tuple(dimensional_calibrations), tuple(result.dimensional_calibrations))

    def test_cross_correlation_keeps_calibration(self) -> None:
        # configure dimensions so that the pixels go from -16S to 16S
        dimensional_calibrations = (Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S"))
        xdata1 = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        xdata2 = DataAndMetadata.new_data_and_metadata(numpy.random.randn(16, 16), dimensional_calibrations=dimensional_calibrations)
        result = Core.function_crosscorrelate(xdata1, xdata2)
        self.assertIsNot(dimensional_calibrations, result.dimensional_calibrations)  # verify
        self.assertEqual(tuple(dimensional_calibrations), tuple(result.dimensional_calibrations))

    def test_histogram_calibrates_x_axis(self) -> None:
        dimensional_calibrations = [Calibration.Calibration(-16, 2, "S"), Calibration.Calibration(-16, 2, "S")]
        intensity_calibration = Calibration.Calibration(2, 3, units="L")
        data: numpy.typing.NDArray[numpy.uint32] = numpy.ones((16, 16), numpy.uint32)
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

    def test_crop_out_of_bounds_produces_proper_size_data(self) -> None:
        data: numpy.typing.NDArray[numpy.uint32] = numpy.ones((16, 16), numpy.uint32)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        result = Core.function_crop(xdata, ((0.75, 0.75), (0.5, 0.5)))
        self.assertEqual((8, 8), result.data_shape)
        self.assertEqual(0, numpy.amin(result))
        self.assertEqual(1, numpy.amax(result))

    def test_crop_rotated_produces_proper_size_data(self) -> None:
        data: numpy.typing.NDArray[numpy.uint32] = numpy.ones((16, 16), numpy.uint32)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        result = Core.function_crop_rotated(xdata, ((0.75, 0.75), (0.5, 0.5)), 0.3)
        self.assertEqual((8, 8), result.data_shape)
        self.assertEqual(0, numpy.amin(result))
        self.assertEqual(1, numpy.amax(result))
        # test rounding. case from actual failing code.
        xdata = DataAndMetadata.new_data_and_metadata(numpy.ones((76, 256), numpy.uint32))
        result = Core.function_crop_rotated(xdata, ((0.5, 0.5), (1 / 76, math.sqrt(2) / 2)), math.radians(45))
        self.assertEqual((1, 181), result.data_shape)
        # another case where height was zero.
        xdata = DataAndMetadata.new_data_and_metadata(numpy.ones((49, 163), numpy.uint32))
        result = Core.function_crop_rotated(xdata, ((0.5, 0.5), (1 / 49, 115 / 163)), -0.8096358402621856)
        self.assertEqual((1, 115), result.data_shape)

    def test_redimension_basic_functionality(self) -> None:
        data: numpy.typing.NDArray[numpy.int32] = numpy.ones((100, 100), dtype=numpy.int32)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        xdata_redim = Core.function_redimension(xdata, DataAndMetadata.DataDescriptor(True, 0, 1))
        self.assertEqual(xdata.data_descriptor.expected_dimension_count, xdata_redim.data_descriptor.expected_dimension_count)

    def test_squeeze_does_not_remove_last_datum_dimension(self) -> None:
        data: numpy.typing.NDArray[numpy.int32] = numpy.ones((1, 1, 1, 1), dtype=numpy.int32)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        xdata_squeeze= Core.function_squeeze(xdata)
        self.assertEqual(1, xdata_squeeze.data_descriptor.expected_dimension_count)

    def test_match_template_for_1d_data(self) -> None:
        data = numpy.random.RandomState(42).randn(100)
        image_xdata = DataAndMetadata.new_data_and_metadata(data)
        template_xdata = DataAndMetadata.new_data_and_metadata(data[40:60])
        ccorr_xdata = Core.function_match_template(image_xdata, template_xdata)
        self.assertTrue(ccorr_xdata.is_data_1d)
        self.assertEqual(numpy.argmax(ccorr_xdata._data_ex), 50)
        self.assertAlmostEqual(numpy.amax(ccorr_xdata._data_ex), 1.0, places=1)

    def test_match_template_for_2d_data(self) -> None:
        data = numpy.random.RandomState(42).randn(100, 100)
        image_xdata = DataAndMetadata.new_data_and_metadata(data)
        template_xdata = DataAndMetadata.new_data_and_metadata(data[40:60, 15:20])
        ccorr_xdata = Core.function_match_template(image_xdata, template_xdata)
        self.assertTrue(ccorr_xdata.is_data_2d)
        self.assertTupleEqual(numpy.unravel_index(numpy.argmax(ccorr_xdata._data_ex), ccorr_xdata.data_shape), (50, 17))
        self.assertAlmostEqual(numpy.amax(ccorr_xdata._data_ex), 1.0, places=1)

    def test_register_template_for_1d_data(self) -> None:
        data = numpy.random.RandomState(42).randn(100)
        image_xdata = DataAndMetadata.new_data_and_metadata(data)
        template_xdata = DataAndMetadata.new_data_and_metadata(data[40:60])
        ccoeff, max_pos = Core.function_register_template(image_xdata, template_xdata)
        self.assertEqual(len(max_pos), 1)
        self.assertAlmostEqual(max_pos[0], 0, places=1)
        self.assertAlmostEqual(ccoeff, 1.0, places=1)

    def test_register_template_for_2d_data(self) -> None:
        data = numpy.random.RandomState(42).randn(100, 100)
        image_xdata = DataAndMetadata.new_data_and_metadata(data)
        template_xdata = DataAndMetadata.new_data_and_metadata(data[40:60, 15:20])
        ccoeff, max_pos = Core.function_register_template(image_xdata, template_xdata)
        self.assertEqual(len(max_pos), 2)
        self.assertTrue(numpy.allclose(max_pos, (0, -33), atol=0.1))
        self.assertAlmostEqual(ccoeff, 1.0, places=1)

    def test_register_template_for_2d_data_with_mask(self) -> None:
        data = numpy.zeros((100, 100))
        data[5::10, 5::10] = 1
        data = scipy.ndimage.gaussian_filter(data, 2)
        image_xdata = DataAndMetadata.new_data_and_metadata(data)
        template_xdata = DataAndMetadata.new_data_and_metadata(scipy.ndimage.shift(data, (-2.3, -3.7), order=1))
        mask: numpy.typing.NDArray[numpy.bool_] = numpy.zeros(data.shape, dtype=bool)
        y, x = numpy.mgrid[-data.shape[0]/2:data.shape[0]/2:1j*data.shape[0], -data.shape[0]/2:data.shape[0]/2:1j*data.shape[0]] # type: ignore
        mask[numpy.sqrt(x**2 + y**2) < 7] = True
        # We make a mask that is one lattice site offset in y-direction
        mask = numpy.roll(mask, (10, 0), axis=(0, 1)).astype(bool)
        ccoeff, max_pos = Core.function_register_template(image_xdata, template_xdata, ccorr_mask=mask)
        self.assertEqual(len(max_pos), 2)
        self.assertTrue(numpy.allclose(max_pos, (12.3, 3.7), atol=0.5))
        self.assertAlmostEqual(ccoeff, 1.0, delta=0.2)
        # Now test that we get the original shift without a mask
        ccoeff, max_pos = Core.function_register_template(image_xdata, template_xdata)
        self.assertTrue(numpy.allclose(max_pos, (2.3, 3.7), atol=0.5))

    def test_sequence_join(self) -> None:
        xdata_list = [DataAndMetadata.new_data_and_metadata(numpy.ones((16, 32)), data_descriptor=DataAndMetadata.DataDescriptor(False, 1, 1))]
        xdata_list.append(DataAndMetadata.new_data_and_metadata(numpy.ones((2, 16, 32)), data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 1)))
        xdata_list.append(DataAndMetadata.new_data_and_metadata(numpy.ones((1, 16, 32)), data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 1)))
        sequence_xdata = Core.function_sequence_join(xdata_list)
        self.assertTrue(sequence_xdata.is_sequence)
        self.assertTrue(sequence_xdata.is_collection)
        self.assertSequenceEqual(sequence_xdata.data_shape, (4, 16, 32))

    def test_sequence_split(self) -> None:
        sequence_xdata = DataAndMetadata.new_data_and_metadata(numpy.ones((3, 16, 32)), data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 1))
        xdata_list = Core.function_sequence_split(sequence_xdata)
        self.assertEqual(len(xdata_list), 3)
        for xdata in xdata_list:
            self.assertSequenceEqual(xdata.data_shape, (16, 32))
            self.assertTrue(xdata.is_collection)
            self.assertFalse(xdata.is_sequence)

    def test_affine_transform(self) -> None:
        data_shapes = [(5, 5)]#, (6, 6)]
        for data_shape in data_shapes:
            with self.subTest(data_shape=data_shape):
                original_data = numpy.zeros(data_shape)
                original_data[1:-1, 2:-2] = 1
                transformation_matrix: numpy.typing.NDArray[typing.Any] = numpy.array(((numpy.cos(numpy.pi/2), -numpy.sin(numpy.pi/2), 0),
                                                                          (numpy.sin(numpy.pi/2),  numpy.cos(numpy.pi/2), 0),
                                                                          (0,                      0,                     1)))
                transformed = Core.function_affine_transform(original_data, transformation_matrix, order=1)
                self.assertTrue(numpy.allclose(numpy.rot90(original_data), transformed._data_ex))  # type: ignore

    def test_affine_transform_does_identity_correctly(self) -> None:
        data_shapes = [(4, 4), (5, 5)]
        for data_shape in data_shapes:
            with self.subTest(data_shape=data_shape):
                original_data = numpy.random.rand(*data_shape)
                transformation_matrix: numpy.typing.NDArray[typing.Any] = numpy.array(((1, 0), (0, 1)))
                transformed = Core.function_affine_transform(original_data, transformation_matrix, order=1)
                self.assertTrue(numpy.allclose(original_data, transformed._data_ex))

    def test_operations_using_copy_on_h5py_array(self) -> None:
        bio = io.BytesIO()
        with h5py.File(bio, "w") as f:
            dataset = f.create_dataset("data", data=numpy.ones((4, 4), dtype=float))
            d = DataAndMetadata.new_data_and_metadata(dataset)
            Core.function_fft(d)
            Core.function_ifft(d)
            Core.function_autocorrelate(d)
            Core.function_transpose_flip(d)
            Core.function_transpose_flip(d, True, True, True)
            Core.function_rebin_2d(d, d.data_shape)
            Core.function_rebin_2d(d, (2, 2))
            Core.function_resample_2d(d, (3, 3))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
