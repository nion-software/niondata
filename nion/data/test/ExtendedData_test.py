# standard libraries
import h5py
import logging
import os
import shutil
import unittest

# third party libraries
import numpy
import numpy.typing

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata


def db_make_directory_if_needed(directory_path: str) -> None:
    if os.path.exists(directory_path):
        if not os.path.isdir(directory_path):
            raise OSError("Path is not a directory:", directory_path)
    else:
        os.makedirs(directory_path)


class TestExtendedData(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rgb_data_constructs_with_default_calibrations(self) -> None:
        data: numpy.typing.NDArray[numpy.uint8] = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertEqual(len(xdata.dimensional_shape), len(xdata.dimensional_calibrations))

    def test_rgb_data_slice_works_correctly(self) -> None:
        data: numpy.typing.NDArray[numpy.uint8] = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertTrue(xdata.is_data_rgb_type)
        xdata_slice = xdata[2:6, 2:6]
        self.assertTrue(xdata_slice.is_data_rgb_type)
        self.assertTrue(xdata_slice.dimensional_shape, (4, 4))

    def test_data_slice_calibrates_correctly(self) -> None:
        data: numpy.typing.NDArray[numpy.float32] = numpy.zeros((100, 100), dtype=numpy.float32)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        calibrations = xdata[40:60, 40:60].dimensional_calibrations
        self.assertAlmostEqual(calibrations[0].offset, 40)
        self.assertAlmostEqual(calibrations[0].scale, 1)
        self.assertAlmostEqual(calibrations[1].offset, 40)
        self.assertAlmostEqual(calibrations[1].scale, 1)

    def test_data_slice_of_sequence_handles_calibrations(self) -> None:
        data: numpy.typing.NDArray[numpy.float32] = numpy.zeros((10, 100, 100), dtype=numpy.float32)
        intensity_calibration = Calibration.Calibration(0.1, 0.2, "I")
        dimensional_calibrations = [Calibration.Calibration(0.11, 0.22, "S"), Calibration.Calibration(0.11, 0.22, "A"), Calibration.Calibration(0.111, 0.222, "B")]
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        self.assertFalse(xdata[3].is_sequence)
        self.assertTrue(xdata[3:4].is_sequence)
        self.assertAlmostEqual(xdata[3].intensity_calibration.offset, xdata.intensity_calibration.offset)
        self.assertAlmostEqual(xdata[3].intensity_calibration.scale, xdata.intensity_calibration.scale)
        self.assertEqual(xdata[3].intensity_calibration.units, xdata.intensity_calibration.units)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[0].offset, xdata.dimensional_calibrations[1].offset)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale)
        self.assertEqual(xdata[3].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[1].offset, xdata.dimensional_calibrations[2].offset)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[1].scale, xdata.dimensional_calibrations[2].scale)
        self.assertEqual(xdata[3].dimensional_calibrations[1].units, xdata.dimensional_calibrations[2].units)

    def test_xdata_backed_by_ndarray_works_with_all_operators(self) -> None:
        xdata = DataAndMetadata.new_data_and_metadata(numpy.ones((4, 4)))
        results = list()
        results.append(abs(xdata))
        results.append(-xdata)
        results.append(+xdata)
        results.append(xdata + 5)
        results.append(5 + xdata)
        results.append(xdata - 5)
        results.append(5 - xdata)
        results.append(xdata * 5)
        results.append(5 * xdata)
        results.append(xdata / 5)
        results.append(5 / xdata)
        results.append(xdata // 5)
        results.append(5 // xdata)
        results.append(xdata % 5)
        results.append(5 % xdata)
        results.append(xdata ** 5)
        results.append(5 ** xdata)

    def test_xdata_backed_by_hdf5_dataset_works_with_all_operators(self) -> None:
        current_working_directory = os.getcwd()
        workspace_dir = os.path.join(current_working_directory, "__Test")
        db_make_directory_if_needed(workspace_dir)
        try:
            with h5py.File(os.path.join(workspace_dir, "file.h5"), "w") as f:
                dataset = f.create_dataset("data", data=numpy.ones((4, 4)))
                xdata = DataAndMetadata.new_data_and_metadata(dataset)
                results = list()
                results.append(abs(xdata))
                results.append(-xdata)
                results.append(+xdata)
                results.append(xdata + 5)
                results.append(5 + xdata)
                results.append(xdata - 5)
                results.append(5 - xdata)
                results.append(xdata * 5)
                results.append(5 * xdata)
                results.append(xdata / 5)
                results.append(5 / xdata)
                results.append(xdata // 5)
                results.append(5 // xdata)
                results.append(xdata % 5)
                results.append(5 % xdata)
                results.append(xdata ** 5)
                results.append(5 ** xdata)
        finally:
            # print(f"rmtree {workspace_dir}")
            shutil.rmtree(workspace_dir)

    def test_numpy_functions_work_directly_with_xdata(self) -> None:
        data: numpy.typing.NDArray[numpy.int32] = numpy.ones((100, 100), dtype=numpy.int32)
        data[50, 50] = 2
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertEqual(1, numpy.amin(xdata))
        self.assertEqual(2, numpy.amax(xdata))

    def test_data_descriptor_is_a_copy(self) -> None:
        data: numpy.typing.NDArray[numpy.int32] = numpy.ones((100, 100), dtype=numpy.int32)
        data[50, 50] = 2
        xdata = DataAndMetadata.new_data_and_metadata(data)
        xdata.data_descriptor.is_sequence = True
        self.assertFalse(xdata.data_descriptor.is_sequence)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
