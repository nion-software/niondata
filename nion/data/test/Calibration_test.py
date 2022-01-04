# standard libraries
import logging
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Calibration


class TestCalibrationClass(unittest.TestCase):

    def test_conversion(self) -> None:
        calibration = Calibration.Calibration(3.0, 2.0, "x")
        self.assertEqual(calibration.convert_to_calibrated_value_str(5.0), u"13 x")

    def test_calibration_should_work_for_complex_data(self) -> None:
        calibration = Calibration.Calibration(1.0, 2.0, "c")
        value_array: numpy.typing.NDArray[numpy.complex128] = numpy.zeros((1, ), dtype=numpy.complex128)
        value_array[0] = 3 + 4j
        self.assertEqual(calibration.convert_to_calibrated_value_str(value_array[0]), u"7+8j c")
        self.assertEqual(calibration.convert_to_calibrated_size_str(value_array[0]), u"6+8j c")

    def test_calibration_displays_inverted_units_when_requested(self) -> None:
        calibration = Calibration.Calibration(scale=0.5, units="1/c")
        self.assertEqual(calibration.convert_to_calibrated_value_str(4), u"2 1/c")
        self.assertEqual(calibration.convert_to_calibrated_value_str(4, display_inverted=True), u"0.5 c")

    def test_calibration_displays_non_inverted_units_when_close_to_zero(self) -> None:
        calibration = Calibration.Calibration(scale=0.5, units="1/c")
        self.assertEqual(calibration.convert_to_calibrated_value_str(0, display_inverted=True), u"0 1/c")
        self.assertEqual(calibration.convert_to_calibrated_value_str(1e-15, display_inverted=True), u"5e-16 1/c")

    def test_calibration_should_work_for_rgb_data(self) -> None:
        calibration = Calibration.Calibration(1.0, 2.0, "c")
        value: numpy.typing.NDArray[numpy.uint8] = numpy.zeros((4, ), dtype=numpy.uint8)
        self.assertEqual(calibration.convert_to_calibrated_value_str(value), "0, 0, 0, 0")
        self.assertEqual(calibration.convert_to_calibrated_size_str(value), "0, 0, 0, 0")

    def test_calibration_conversion_to_string_can_handle_numpy_types(self) -> None:
        calibration = Calibration.Calibration(1.0, 2.0, "c")
        # ignore type to make the test salient
        self.assertEqual(calibration.convert_to_calibrated_value_str(numpy.uint32(14)), "29 c")  # type: ignore

    def test_calibration_equality(self) -> None:
        calibration1 = Calibration.Calibration(1.0, 2.0, "c")
        calibration2 = Calibration.Calibration(1.0, 2.0, "c")
        self.assertEqual(calibration1, calibration2)
        calibration1 = Calibration.Calibration(1.0, 2.1, "c")
        calibration2 = Calibration.Calibration(1.0, 2.0, "c")
        self.assertNotEqual(calibration1, calibration2)
        self.assertNotEqual(calibration1, None)
        self.assertTrue(calibration1 != None)
        self.assertFalse(calibration1 == None)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
