"""Tests for annotated_array.primitives — FFT / IFFT.

These tests are the canonical human-readable specification of the expected
behaviour.  Each test focuses on one observable property and is written to
be self-explanatory without requiring any other context.
"""

import math
import typing
import unittest

import numpy
import numpy.testing

from nion.data import annotated_array
from nion.data.annotated_array import primitives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1d_array(
    data: numpy.ndarray,
    scale: float = 1.0,
    offset: float = 0.0,
    unit: str = "",
) -> annotated_array.AnnotatedArray:
    """Return a 1-D AnnotatedArray with a single affine spatial calibration.

    The value_type is inferred from the array dtype so that both real and
    complex test inputs are accepted without additional boilerplate.
    """
    calibration = annotated_array.CoordinateCalibration(
        calibrations=(annotated_array.AffineCalibration(scale=scale, offset=offset, unit=unit),)
    )
    signal_group = annotated_array.AxisGroup.from_1d_size(
        data.shape[0],
        coordinate_calibrations={"spatial": calibration},
        primary_calibration_key="spatial",
    )
    value_type = annotated_array.infer_value_type(data.dtype)
    descriptor = annotated_array.ArrayDescriptor((signal_group,), value_type=value_type)
    return annotated_array.AnnotatedArray(data=data, descriptor=descriptor)


def _make_2d_array(
    data: numpy.ndarray,
    scale: tuple[float, float] = (1.0, 1.0),
    unit: str = "",
) -> annotated_array.AnnotatedArray:
    """Return a 2-D AnnotatedArray with a single affine spatial calibration.

    The value_type is inferred from the array dtype.
    """
    calibration = annotated_array.CoordinateCalibration(
        calibrations=(
            annotated_array.AffineCalibration(scale=scale[0], unit=unit),
            annotated_array.AffineCalibration(scale=scale[1], unit=unit),
        )
    )
    signal_group = annotated_array.AxisGroup.from_2d_size(
        (data.shape[0], data.shape[1]),
        coordinate_calibrations={"spatial": calibration},
        primary_calibration_key="spatial",
    )
    value_type = annotated_array.infer_value_type(data.dtype)
    descriptor = annotated_array.ArrayDescriptor((signal_group,), value_type=value_type)
    return annotated_array.AnnotatedArray(data=data, descriptor=descriptor)


# ---------------------------------------------------------------------------
# FFT — output data
# ---------------------------------------------------------------------------

class TestFftOutputData(unittest.TestCase):

    def test_fft_1d_output_is_complex(self) -> None:
        """The output of fft on a real 1-D array must be complex."""
        src = _make_1d_array(numpy.ones(8, dtype=numpy.float64))
        result = primitives.fft(src)
        self.assertTrue(numpy.iscomplexobj(result.data))

    def test_fft_2d_output_is_complex(self) -> None:
        """The output of fft on a real 2-D array must be complex."""
        src = _make_2d_array(numpy.ones((4, 8), dtype=numpy.float64))
        result = primitives.fft(src)
        self.assertTrue(numpy.iscomplexobj(result.data))

    def test_fft_1d_preserves_rms_energy(self) -> None:
        """RMS of input equals RMS of output (Parseval / energy-normalised FFT)."""
        rng = numpy.random.default_rng(0)
        data = rng.standard_normal(64)
        src = _make_1d_array(data)
        result = primitives.fft(src)
        rms_in  = numpy.sqrt(numpy.mean(numpy.abs(data) ** 2))
        rms_out = numpy.sqrt(numpy.mean(numpy.abs(result.data) ** 2))
        numpy.testing.assert_allclose(rms_in, rms_out, rtol=1e-12)

    def test_fft_2d_preserves_rms_energy(self) -> None:
        """RMS of input equals RMS of output (Parseval / energy-normalised FFT)."""
        rng = numpy.random.default_rng(1)
        data = rng.standard_normal((16, 32))
        src = _make_2d_array(data)
        result = primitives.fft(src)
        rms_in  = numpy.sqrt(numpy.mean(numpy.abs(data) ** 2))
        rms_out = numpy.sqrt(numpy.mean(numpy.abs(result.data) ** 2))
        numpy.testing.assert_allclose(rms_in, rms_out, rtol=1e-12)

    def test_fft_1d_dc_component_is_at_centre(self) -> None:
        """For a constant (DC-only) signal the single non-zero bin must be at the array centre."""
        n = 16
        data = numpy.ones(n, dtype=numpy.float64)
        src = _make_1d_array(data)
        result = primitives.fft(src)
        magnitude = numpy.abs(numpy.asarray(result.data))
        centre = n // 2
        # N ones scaled by 1/sqrt(N) → DC bin = N/sqrt(N) = sqrt(N).
        self.assertAlmostEqual(magnitude[centre], math.sqrt(n))
        # All other bins must be (essentially) zero.
        mask = numpy.ones(n, dtype=bool)
        mask[centre] = False
        numpy.testing.assert_allclose(magnitude[mask], 0.0, atol=1e-12)

    def test_fft_2d_dc_component_is_at_centre(self) -> None:
        """For a constant 2-D image the only non-zero bin must be at the image centre."""
        rows, cols = 8, 12
        data = numpy.ones((rows, cols), dtype=numpy.float64)
        src = _make_2d_array(data)
        result = primitives.fft(src)
        magnitude = numpy.abs(numpy.asarray(result.data))
        cy, cx = rows // 2, cols // 2
        self.assertGreater(magnitude[cy, cx], 0.5)
        # All other bins must be (essentially) zero.
        mask = numpy.ones((rows, cols), dtype=bool)
        mask[cy, cx] = False
        numpy.testing.assert_allclose(magnitude[mask], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# FFT — output calibrations
# ---------------------------------------------------------------------------

class TestFftOutputCalibrations(unittest.TestCase):

    def test_fft_1d_frequency_scale_is_reciprocal(self) -> None:
        """freq_scale = 1 / (spatial_scale * N)."""
        n = 32
        s = 0.5          # spatial scale (e.g. 0.5 nm/pixel)
        src = _make_1d_array(numpy.zeros(n), scale=s, unit="nm")
        result = primitives.fft(src)

        signal_group = result.descriptor.axis_groups[-1]
        freq_cal = signal_group.get_calibration(0)   # primary (frequency) calibration
        self.assertIsInstance(freq_cal, annotated_array.AffineCalibration)
        freq_aff = typing.cast(annotated_array.AffineCalibration, freq_cal)
        expected_scale = 1.0 / (s * n)
        self.assertAlmostEqual(freq_aff.scale, expected_scale)

    def test_fft_1d_frequency_offset_places_dc_at_centre(self) -> None:
        """freq_offset = (-0.5 - N//2) / (spatial_scale * N)."""
        n = 32
        s = 0.5
        src = _make_1d_array(numpy.zeros(n), scale=s, unit="nm")
        result = primitives.fft(src)

        signal_group = result.descriptor.axis_groups[-1]
        freq_cal = signal_group.get_calibration(0)
        self.assertIsInstance(freq_cal, annotated_array.AffineCalibration)
        freq_aff = typing.cast(annotated_array.AffineCalibration, freq_cal)
        expected_offset = (-0.5 - n // 2) / (s * n)
        self.assertAlmostEqual(freq_aff.offset, expected_offset)

    def test_fft_1d_frequency_unit_is_reciprocal(self) -> None:
        """The frequency unit is '1/<spatial_unit>'."""
        src = _make_1d_array(numpy.zeros(16), scale=1.0, unit="nm")
        result = primitives.fft(src)
        signal_group = result.descriptor.axis_groups[-1]
        freq_cal = signal_group.get_calibration(0)
        self.assertIsInstance(freq_cal, annotated_array.AffineCalibration)
        freq_aff = typing.cast(annotated_array.AffineCalibration, freq_cal)
        self.assertEqual("1/nm", freq_aff.unit)

    def test_fft_2d_frequency_unit_is_reciprocal(self) -> None:
        """Both axes of a 2-D FFT result have reciprocal units."""
        src = _make_2d_array(numpy.zeros((8, 12)), scale=(1.0, 1.0), unit="nm")
        result = primitives.fft(src)
        signal_group = result.descriptor.axis_groups[-1]
        freq_cal_row = signal_group.get_calibration(0)
        freq_cal_col = signal_group.get_calibration(1)
        self.assertIsInstance(freq_cal_row, annotated_array.AffineCalibration)
        self.assertIsInstance(freq_cal_col, annotated_array.AffineCalibration)
        freq_row_aff = typing.cast(annotated_array.AffineCalibration, freq_cal_row)
        freq_col_aff = typing.cast(annotated_array.AffineCalibration, freq_cal_col)
        self.assertEqual("1/nm", freq_row_aff.unit)
        self.assertEqual("1/nm", freq_col_aff.unit)

    def test_fft_2d_frequency_calibration_applied_to_each_axis(self) -> None:
        """Both axes of a 2-D FFT result have independent frequency calibrations."""
        rows, cols = 8, 16
        s_row, s_col = 0.25, 0.5
        src = _make_2d_array(numpy.zeros((rows, cols)), scale=(s_row, s_col), unit="nm")
        result = primitives.fft(src)

        signal_group = result.descriptor.axis_groups[-1]
        freq_cal_row = signal_group.get_calibration(0)
        freq_cal_col = signal_group.get_calibration(1)
        self.assertIsInstance(freq_cal_row, annotated_array.AffineCalibration)
        self.assertIsInstance(freq_cal_col, annotated_array.AffineCalibration)
        freq_row_aff = typing.cast(annotated_array.AffineCalibration, freq_cal_row)
        freq_col_aff = typing.cast(annotated_array.AffineCalibration, freq_cal_col)
        self.assertAlmostEqual(freq_row_aff.scale, 1.0 / (s_row * rows))
        self.assertAlmostEqual(freq_col_aff.scale, 1.0 / (s_col * cols))

    def test_fft_primary_calibration_key_is_preserved(self) -> None:
        """After FFT the primary coordinate calibration key is unchanged."""
        src = _make_1d_array(numpy.zeros(16), scale=1.0, unit="nm")
        result = primitives.fft(src)
        signal_group = result.descriptor.axis_groups[-1]
        self.assertEqual("spatial", signal_group.primary_calibration_key)

    def test_fft_calibration_key_has_frequency_values(self) -> None:
        """After FFT each calibration key is present with frequency-domain values."""
        src = _make_1d_array(numpy.zeros(16), scale=0.3, unit="nm")
        result = primitives.fft(src)
        signal_group = result.descriptor.axis_groups[-1]
        self.assertIn("spatial", signal_group.calibration_keys)
        spatial_cal = signal_group.get_calibration(0, key="spatial")
        self.assertIsInstance(spatial_cal, annotated_array.AffineCalibration)
        spatial_aff = typing.cast(annotated_array.AffineCalibration, spatial_cal)
        self.assertAlmostEqual(spatial_aff.scale, 1.0 / (0.3 * 16))
        self.assertEqual("1/nm", spatial_aff.unit)

    def test_fft_intensity_calibration_is_unchanged(self) -> None:
        """FFT must not alter the intensity calibration."""
        from nion.data import Calibration
        from nion.data import DataAndMetadata
        xdata = DataAndMetadata.new_data_and_metadata(
            numpy.zeros(16),
            intensity_calibration=Calibration.Calibration(0.0, 2.5, "counts"),
        )
        aa = annotated_array.from_data_and_metadata(xdata)
        result = primitives.fft(aa)
        intensity = result.descriptor.get_intensity_calibration()
        self.assertIsInstance(intensity, annotated_array.AffineCalibration)
        intensity_aff = typing.cast(annotated_array.AffineCalibration, intensity)
        self.assertAlmostEqual(intensity_aff.scale, 2.5)
        self.assertEqual("counts", intensity_aff.unit)


# ---------------------------------------------------------------------------
# FFT — input validation
# ---------------------------------------------------------------------------

class TestFftInputValidation(unittest.TestCase):

    def test_fft_rejects_multiple_axis_groups(self) -> None:
        """FFT requires exactly one axis group."""
        from nion.data.annotated_array._implementation import Axis
        navigation_group = annotated_array.AxisGroup(axes=(Axis("n", 4),))
        signal_group = annotated_array.AxisGroup(axes=(Axis("x", 8),))
        descriptor = annotated_array.ArrayDescriptor((navigation_group, signal_group))
        array = annotated_array.AnnotatedArray(data=numpy.zeros((4, 8)), descriptor=descriptor)
        with self.assertRaises(ValueError):
            primitives.fft(array)

    def test_fft_rejects_signal_rank_other_than_1_or_2(self) -> None:
        """FFT on a 3-D signal axis group must raise ValueError."""
        from nion.data.annotated_array._implementation import Axis
        signal_group = annotated_array.AxisGroup(axes=(Axis("x", 4), Axis("y", 4), Axis("z", 4)))
        descriptor = annotated_array.ArrayDescriptor((signal_group,))
        array = annotated_array.AnnotatedArray(data=numpy.zeros((4, 4, 4)), descriptor=descriptor)
        with self.assertRaises(ValueError):
            primitives.fft(array)

    def test_fft_accepts_complex_input(self) -> None:
        """FFT on a complex input must succeed and produce a complex output."""
        data = numpy.ones(8, dtype=numpy.complex128)
        src = _make_1d_array(data)
        result = primitives.fft(src)
        self.assertTrue(numpy.iscomplexobj(result.data))


# ---------------------------------------------------------------------------
# IFFT — output data
# ---------------------------------------------------------------------------

class TestIfftOutputData(unittest.TestCase):

    def test_ifft_1d_round_trip_recovers_original_data(self) -> None:
        """ifft(fft(x)) must recover x to floating-point precision."""
        rng = numpy.random.default_rng(2)
        data = rng.standard_normal(64)
        src = _make_1d_array(data)
        recovered = primitives.ifft(primitives.fft(src))
        numpy.testing.assert_allclose(numpy.real(recovered.data), data, atol=1e-12)


class TestIfftInputValidation(unittest.TestCase):

    def test_ifft_rejects_multiple_axis_groups(self) -> None:
        """IFFT requires exactly one axis group."""
        from nion.data.annotated_array._implementation import Axis
        navigation_group = annotated_array.AxisGroup(axes=(Axis("n", 4),))
        signal_group = annotated_array.AxisGroup(axes=(Axis("x", 8),))
        descriptor = annotated_array.ArrayDescriptor((navigation_group, signal_group), value_type=annotated_array.ValueType.COMPLEX)
        array = annotated_array.AnnotatedArray(data=numpy.zeros((4, 8), dtype=numpy.complex128), descriptor=descriptor)
        with self.assertRaises(ValueError):
            primitives.ifft(array)

    def test_ifft_2d_round_trip_recovers_original_data(self) -> None:
        """ifft(fft(x)) must recover x to floating-point precision."""
        rng = numpy.random.default_rng(3)
        data = rng.standard_normal((16, 32))
        src = _make_2d_array(data)
        recovered = primitives.ifft(primitives.fft(src))
        numpy.testing.assert_allclose(numpy.real(recovered.data), data, atol=1e-12)


# ---------------------------------------------------------------------------
# IFFT — calibration round-trip
# ---------------------------------------------------------------------------

class TestIfftCalibrationRoundTrip(unittest.TestCase):

    def test_ifft_restores_spatial_scale(self) -> None:
        """After ifft(fft(x)) the signal AxisGroup primary scale must equal the original."""
        s = 0.4
        src = _make_1d_array(numpy.zeros(32), scale=s, unit="nm")
        recovered = primitives.ifft(primitives.fft(src))
        signal_group = recovered.descriptor.axis_groups[-1]
        cal = signal_group.get_calibration(0)
        self.assertIsInstance(cal, annotated_array.AffineCalibration)
        cal_aff = typing.cast(annotated_array.AffineCalibration, cal)
        self.assertAlmostEqual(cal_aff.scale, s)

    def test_ifft_restores_spatial_unit(self) -> None:
        """After ifft(fft(x)) the primary unit must match the original unit."""
        src = _make_1d_array(numpy.zeros(32), scale=0.4, unit="nm")
        recovered = primitives.ifft(primitives.fft(src))
        signal_group = recovered.descriptor.axis_groups[-1]
        cal = signal_group.get_calibration(0)
        self.assertIsInstance(cal, annotated_array.AffineCalibration)
        cal_aff = typing.cast(annotated_array.AffineCalibration, cal)
        self.assertEqual("nm", cal_aff.unit)

    def test_ifft_restores_original_calibration_key(self) -> None:
        """The primary calibration key is restored to its pre-FFT name."""
        src = _make_1d_array(numpy.zeros(32), scale=0.4, unit="nm")
        recovered = primitives.ifft(primitives.fft(src))
        signal_group = recovered.descriptor.axis_groups[-1]
        self.assertEqual("spatial", signal_group.primary_calibration_key)

    def test_ifft_2d_restores_both_axes(self) -> None:
        """Both axis calibrations are recovered correctly after a 2-D round-trip."""
        rows, cols = 8, 16
        s_row, s_col = 0.25, 0.5
        src = _make_2d_array(numpy.zeros((rows, cols)), scale=(s_row, s_col), unit="nm")
        recovered = primitives.ifft(primitives.fft(src))
        signal_group = recovered.descriptor.axis_groups[-1]
        cal_row = signal_group.get_calibration(0)
        cal_col = signal_group.get_calibration(1)
        self.assertIsInstance(cal_row, annotated_array.AffineCalibration)
        self.assertIsInstance(cal_col, annotated_array.AffineCalibration)
        cal_row_aff = typing.cast(annotated_array.AffineCalibration, cal_row)
        cal_col_aff = typing.cast(annotated_array.AffineCalibration, cal_col)
        self.assertAlmostEqual(cal_row_aff.scale, s_row)
        self.assertAlmostEqual(cal_col_aff.scale, s_col)

    def test_ifft_derived_spatial_scale_from_frequency_calibration(self) -> None:
        """When an AnnotatedArray with frequency calibration is constructed directly
        (not via fft) ifft must derive the spatial calibration as 1/(scale_freq * N)."""
        n = 32
        scale_freq = 0.1  # 1/nm per pixel
        freq_cal = annotated_array.CoordinateCalibration(
            calibrations=(annotated_array.AffineCalibration(scale=scale_freq, unit="1/nm"),)
        )
        signal_group = annotated_array.AxisGroup.from_1d_size(
            n,
            coordinate_calibrations={"frequency": freq_cal},
            primary_calibration_key="frequency",
        )
        descriptor = annotated_array.ArrayDescriptor((signal_group,), value_type=annotated_array.ValueType.COMPLEX)
        array = annotated_array.AnnotatedArray(
            data=numpy.zeros(n, dtype=numpy.complex128),
            descriptor=descriptor,
        )
        result = primitives.ifft(array)
        spatial_cal = result.descriptor.axis_groups[-1].get_calibration(0)
        self.assertIsInstance(spatial_cal, annotated_array.AffineCalibration)
        spatial_aff = typing.cast(annotated_array.AffineCalibration, spatial_cal)
        expected_scale = 1.0 / (scale_freq * n)
        self.assertAlmostEqual(spatial_aff.scale, expected_scale)
        self.assertEqual("nm", spatial_aff.unit)

    def test_fft_ifft_preserves_all_calibration_keys(self) -> None:
        """FFT and IFFT preserve the exact set of calibration keys (same keys in, same keys out).

        Each key's calibration is independently transformed to frequency domain on FFT
        and back to spatial domain on IFFT.
        """
        spatial_cal = annotated_array.AffineCalibration(scale=0.5, offset=0.0, unit="nm")
        angular_cal = annotated_array.AffineCalibration(scale=0.1, offset=0.0, unit="radians")
        coord_cals = {
            "spatial": annotated_array.CoordinateCalibration((spatial_cal,)),
            "angular": annotated_array.CoordinateCalibration((angular_cal,)),
        }
        signal_group = annotated_array.AxisGroup.from_1d_size(
            16, coordinate_calibrations=coord_cals, primary_calibration_key="spatial",
        )
        descriptor = annotated_array.ArrayDescriptor((signal_group,))
        array = annotated_array.AnnotatedArray(data=numpy.ones(16), descriptor=descriptor)

        # FFT: same two keys, now in frequency domain.
        result_fft = primitives.fft(array)
        fft_group = result_fft.descriptor.axis_groups[-1]
        self.assertEqual(set(fft_group.coordinate_calibrations), {"spatial", "angular"})
        self.assertEqual("spatial", fft_group.primary_calibration_key)

        spatial_freq = typing.cast(annotated_array.AffineCalibration, fft_group.get_calibration(0, key="spatial"))
        angular_freq = typing.cast(annotated_array.AffineCalibration, fft_group.get_calibration(0, key="angular"))
        self.assertAlmostEqual(spatial_freq.scale, 1.0 / (0.5 * 16))
        self.assertAlmostEqual(angular_freq.scale, 1.0 / (0.1 * 16))
        self.assertEqual("1/nm", spatial_freq.unit)
        self.assertEqual("1/radians", angular_freq.unit)

        # IFFT: same two keys, back to spatial domain.
        result_ifft = primitives.ifft(result_fft)
        ifft_group = result_ifft.descriptor.axis_groups[-1]
        self.assertEqual(set(ifft_group.coordinate_calibrations), {"spatial", "angular"})
        self.assertEqual("spatial", ifft_group.primary_calibration_key)

        spatial_back = typing.cast(annotated_array.AffineCalibration, ifft_group.get_calibration(0, key="spatial"))
        angular_back = typing.cast(annotated_array.AffineCalibration, ifft_group.get_calibration(0, key="angular"))
        self.assertAlmostEqual(spatial_back.scale, 0.5)
        self.assertAlmostEqual(angular_back.scale, 0.1)
        self.assertEqual("nm", spatial_back.unit)
        self.assertEqual("radians", angular_back.unit)


if __name__ == "__main__":
    unittest.main()
