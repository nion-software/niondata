import gettext
import unittest

import numpy
import scipy.ndimage

# local libraries
from nion.data import DataAndMetadata
from nion.data import MultiDimensionalProcessing

_ = gettext.gettext


class TestMultiDimensionalProcessing(unittest.TestCase):

    def test_function_apply_multi_dimensional_shifts_4d(self) -> None:
        with self.subTest("Test for a sequence of SIs, shift collection dimensions along sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.collection_dimension_indexes))

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1], 0.0], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 1D collections of 2D data, shift data dimensions along sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.datum_dimension_indexes))

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [0.0, shifts[i, 0], shifts[i, 1]], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of SIs, shift data dimensions along collection and sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))

            shifts = numpy.linspace(0, 3, num=int(numpy.prod(shape[:-1]))).reshape(shape[:-1])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.datum_dimension_indexes))

            shifted = numpy.empty_like(data)

            for k in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        shifted[k, i, j] = scipy.ndimage.shift(data[k, i, j], [shifts[k, i, j]], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 1D collections of 2D data, shift collection dimension along sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 2))

            shifts = numpy.linspace(0, 1, num=5)

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.collection_dimension_indexes))

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i], 0.0, 0.0], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

    def test_function_apply_multi_dimensional_shifts_5d(self) -> None:
        with self.subTest("Test for a sequence of 4D images, shift collection dimensions along sequence axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.collection_dimension_indexes))

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1], 0.0, 0.0], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 4D images, shift data dimensions along sequence axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, tuple(xdata.datum_dimension_indexes))

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [0.0, 0.0, shifts[i, 0], shifts[i, 1]], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 4D images, shift sequence dimension along collection axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(1., 1.5, 2.),
                                  (2.5, 3., 3.5)])

            assert xdata.sequence_dimension_index is not None
            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, (xdata.sequence_dimension_index,))

            shifted = numpy.empty_like(data)

            for k in range(shape[1]):
                for i in range(shape[2]):
                    shifted[:, k, i] = scipy.ndimage.shift(data[:, k, i], [shifts[k, i], 0., 0.], order=1)

            assert result is not None
            self.assertTrue(numpy.allclose(result.data, shifted))

    def test_function_measure_multi_dimensional_shifts_3d(self) -> None:
        with self.subTest("Test for a sequence of 2D data, measure shift of data dimensions along sequence axis"):
            shape = (5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[1:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(data[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.array([(0., 2.), (0., 5.), (0., 10.), (0., 2.5), (0., 3.)])

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.datum_dimension_indexes),
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[reference_index]), atol=0.5))

        with self.subTest("Test for a 2D collection of 1D data, measure shift of data dimensions along collection axis"):
            shape = (5, 5, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2]) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.datum_dimension_indexes),
                                                                                          reference_index=reference_index)
            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape)]), atol=0.5))

        with self.subTest("Test for a sequence of 2D data, measure shift of data dimensions along sequence axis relative to previous slice"):
            shape = (5, 100, 100)
            data = numpy.random.rand(*shape[1:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(data[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.array([(0., 2.), (0., 5.), (0., 10.), (0., 2.5), (0., 3.)])

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.datum_dimension_indexes),
                                                                                          reference_index=None)
            expected_res = -1.0 * (shifts[1:] - shifts[:-1])
            expected_res = numpy.append(numpy.zeros((1, 2)), expected_res, axis=0)
            expected_res = numpy.cumsum(expected_res, axis=0)

            self.assertTrue(numpy.allclose(result.data, expected_res, atol=0.5))

    def test_function_measure_multi_dimensional_shifts_3d_works_with_non_integer_bounds(self) -> None:
        with self.subTest("Test for a sequence of 2D data, measure shift of data dimensions along sequence axis (relative shifts)"):
            shape = (10, 100, 100)
            reference_index = None
            data = numpy.random.rand(*shape[1:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(data[numpy.newaxis, ...], shape[0], axis=0)
            bounds = ((0.0584, 0.0622), (0.9161, 0.8775))
            # bounds = [[0.05, 0.06], [0.9, 0.9]]

            shifts = numpy.array([(0., 0.), (0., 5.), (0., 10.), (0., 2.5), (0., 3.), (0.5, 2.), (1.8, 5.), (2.5, 10.), (6.1, 2.5), (3.5, 3.)])

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1]], order=5, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.datum_dimension_indexes),
                                                                                          reference_index=reference_index,
                                                                                          bounds=bounds)
            self.assertTrue(numpy.allclose(result.data, -1.0 * shifts, atol=1.5))

    def test_function_measure_multi_dimensional_shifts_4d(self) -> None:
        with self.subTest("Test for a 2D collection of 2D data, measure shift of data dimensions along collection axis"):
            shape = (5, 5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2], 2) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j, 0], shifts[i, j, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.datum_dimension_indexes),
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape[:-1])]), atol=0.5))

        with self.subTest("Test for a 2D collection of 2D data, measure shift of collection dimensions along data axis"):
            shape = (5, 5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2], 2) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j, 0], shifts[i, j, 1]], order=1, cval=numpy.mean(data))

            shifted = numpy.moveaxis(shifted, (2, 3), (0, 1))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          tuple(shifted_xdata.collection_dimension_indexes),
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape[:-1])]), atol=0.5))

    def test_function_integrate_along_axis_2d(self) -> None:
        with self.subTest("Test for an image that gets reduced to a single number"):
            data = numpy.ones((5, 3))

            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (0, 1))

            self.assertSequenceEqual(integrated.data_shape, (1,))
            self.assertTrue(numpy.allclose(integrated.data, 15.0))
            self.assertEqual(integrated.data_descriptor, DataAndMetadata.DataDescriptor(False, 0, 1))

    def test_function_integrate_along_axis_3d(self) -> None:
        with self.subTest("Test for a sequence of 2D images. Integrate sequence axis."):
            data = numpy.ones((5, 3, 4))

            data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (0,))

            self.assertSequenceEqual(integrated.data_shape, (3, 4))
            self.assertTrue(numpy.allclose(integrated.data, 5.0))

        with self.subTest("Test for a 2D collection of 1D data. Integrate collection axis."):
            data = numpy.ones((5, 3, 4))

            data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 1)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (0, 1))

            self.assertSequenceEqual(integrated.data_shape, (4,))
            self.assertTrue(numpy.allclose(integrated.data, 15.0))

        with self.subTest("Test for a sequence of 2D images. Integrate first data axis."):
            data = numpy.ones((5, 3, 4))

            data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (1,))

            self.assertSequenceEqual(integrated.data_shape, (5, 4))
            self.assertTrue(numpy.allclose(integrated.data, 3.0))

        with self.subTest("Test for a sequence of 2D images. Integrate data axes with mask."):
            data = numpy.ones((5, 7, 8))

            data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            mask = numpy.zeros((7, 8))
            mask[1:4, 2:6] = 1.0
            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (1, 2), integration_mask=mask)

            self.assertSequenceEqual(integrated.data_shape, (5,))
            self.assertTrue(numpy.allclose(integrated.data, 12.0))

    def test_function_integrate_along_axis_4d(self) -> None:
        with self.subTest("Test for a 2D collection of 2D images. Integrate all data axes."):
            data = numpy.ones((5, 3, 4, 2))

            data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (2, 3))

            self.assertSequenceEqual(integrated.data_shape, (5, 3))
            self.assertTrue(numpy.allclose(integrated.data, 8.0))

    def test_function_integrate_along_axis_5d(self) -> None:
        with self.subTest("Integrating sequence axis."):
            data = numpy.ones((5, 3, 4, 2, 6))

            data_descriptor = DataAndMetadata.DataDescriptor(True, 2, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (0,))

            self.assertSequenceEqual(integrated.data_shape, (3, 4, 2, 6))
            self.assertTrue(numpy.allclose(integrated.data, 5.0))

        with self.subTest("Integrating second collection axis."):
            data = numpy.ones((5, 3, 4, 2, 6))

            data_descriptor = DataAndMetadata.DataDescriptor(True, 2, 2)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)

            integrated = MultiDimensionalProcessing.function_integrate_along_axis(xdata, (2,))

            self.assertSequenceEqual(integrated.data_shape, (5, 3, 2, 6))
            self.assertTrue(numpy.allclose(integrated.data, 4.0))
