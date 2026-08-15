"""Annotated-array processing primitives.

This module is the home for low-level processing operations on annotated arrays.
Operation implementations can be added here and re-exported from
`nion.data.annotated_array` as they are introduced.
"""

from __future__ import annotations

import math
import typing

import numpy
import numpy.typing
import scipy.fft

from nion.data.annotated_array._implementation import (
    AffineCalibration,
    AnnotatedArray,
    ArrayDescriptor,
    AxisGroup,
    CoordinateCalibration,
    ValueType,
)


# ---------------------------------------------------------------------------
# Internal calibration helpers
# ---------------------------------------------------------------------------

def _spatial_to_frequency_axis_group(axis_group: AxisGroup) -> AxisGroup:
    """Return a new AxisGroup with all calibrations transformed to frequency domain.

    For each axis of size *N* with spatial calibration ``(scale=s, unit=u)``
    the frequency calibration is::

        scale_freq   = 1 / (s * N)
        offset_freq  = (-0.5 - N // 2) / (s * N)   # DC at centre after fftshift
        unit_freq    = "1/" + u                      # reciprocal unit

    All calibration keys are preserved unchanged (same keys in, same keys out).
    Each calibration is independently transformed under its original key.
    The primary calibration key is unchanged.
    """
    def transform_coord_calibration(coord_cal: CoordinateCalibration) -> CoordinateCalibration:
        freq_calibrations: list[AffineCalibration] = []
        for axis_index, axis in enumerate(axis_group.axes):
            n = axis.size
            spatial_cal = coord_cal.calibrations[axis_index]
            s = spatial_cal.scale if isinstance(spatial_cal, AffineCalibration) else 1.0
            u = spatial_cal.unit  if isinstance(spatial_cal, AffineCalibration) else ""
            freq_calibrations.append(AffineCalibration(
                scale=1.0 / (s * n),
                offset=(-0.5 - n // 2) / (s * n),
                unit=("1/" + u) if u else "",
            ))
        return CoordinateCalibration(calibrations=tuple(freq_calibrations))

    new_calibrations = {
        key: transform_coord_calibration(coord_cal)
        for key, coord_cal in axis_group.coordinate_calibrations.items()
    }

    return AxisGroup(
        axes=axis_group.axes,
        coordinate_system_id=axis_group.coordinate_system_id,
        coordinate_calibrations=new_calibrations,
        primary_calibration_key=axis_group.primary_calibration_key,
    )


def _frequency_to_spatial_axis_group(axis_group: AxisGroup) -> AxisGroup:
    """Return a new AxisGroup with all calibrations transformed back to spatial domain.

    For each frequency calibration ``(scale=s_freq, unit=u_freq)``::

        scale_spatial   = 1 / (s_freq * N)
        offset_spatial  = 0
        unit_spatial    = u_freq[2:] if u_freq.startswith("1/") else ""

    All calibration keys are preserved unchanged (same keys in, same keys out).
    The primary calibration key is unchanged.
    """
    def transform_coord_calibration(coord_cal: CoordinateCalibration) -> CoordinateCalibration:
        spatial_calibrations: list[AffineCalibration] = []
        for axis_index, axis in enumerate(axis_group.axes):
            n = axis.size
            freq_cal = coord_cal.calibrations[axis_index]
            if isinstance(freq_cal, AffineCalibration) and freq_cal.scale != 0.0:
                s_freq = freq_cal.scale
                u_freq = freq_cal.unit
                s_spatial = 1.0 / (s_freq * n)
                u_spatial = u_freq[2:] if u_freq.startswith("1/") else ""
            else:
                s_spatial, u_spatial = 1.0, ""
            spatial_calibrations.append(AffineCalibration(scale=s_spatial, offset=0.0, unit=u_spatial))
        return CoordinateCalibration(calibrations=tuple(spatial_calibrations))

    new_calibrations = {
        key: transform_coord_calibration(coord_cal)
        for key, coord_cal in axis_group.coordinate_calibrations.items()
    }

    return AxisGroup(
        axes=axis_group.axes,
        coordinate_system_id=axis_group.coordinate_system_id,
        coordinate_calibrations=new_calibrations,
        primary_calibration_key=axis_group.primary_calibration_key,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fft(array: AnnotatedArray) -> AnnotatedArray:
    """Compute the forward FFT of an :class:`AnnotatedArray`.

    The transform is applied to the array's only axis group.

    Energy normalisation
    ~~~~~~~~~~~~~~~~~~~~
    The scaling factor ``1 / sqrt(N)`` (1-D) or ``1 / sqrt(N * M)`` (2-D) is
    applied so that the RMS value is preserved:

    .. code-block:: python

        numpy.sqrt(numpy.mean(numpy.abs(data)**2))
        == numpy.sqrt(numpy.mean(numpy.abs(fft(xdata).data)**2))

    DC at centre
    ~~~~~~~~~~~~
    The result is passed through :func:`scipy.fft.fftshift` along the signal axes
    so that the zero-frequency component lies at the array centre.

    Calibration
    ~~~~~~~~~~~
    All calibrations in the signal :class:`AxisGroup` are transformed to the
    frequency domain. The keys of all calibrations are preserved, so the user-defined
    calibration names remain unchanged. For example, if the input has calibrations
    keyed ``"spatial"`` and ``"angular"``, the output will have calibrations keyed
    ``"spatial"`` and ``"angular"`` (with their scales/offsets/units transformed to
    frequency). The primary calibration key is also preserved.

    Args:
        array: Input :class:`AnnotatedArray` with a scalar or complex datum.
               RGB and RGBA value types are not supported.

    Returns:
        :class:`AnnotatedArray` with complex datum (``complex128``) and
        frequency-domain calibrations on the signal :class:`AxisGroup`.

    Raises:
        ValueError: If there is not exactly one axis group, if that group rank
                    is not 1 or 2, or if the value type is not ``SCALAR`` or
                    ``COMPLEX``.
    """
    axis_groups = array.descriptor.axis_groups
    if len(axis_groups) != 1:
        raise ValueError(f"fft: expected exactly one axis group, got {len(axis_groups)}")

    signal_group = axis_groups[-1]
    rank = signal_group.rank

    if rank not in (1, 2):
        raise ValueError(f"fft: signal rank must be 1 or 2, got {rank}")

    value_type = array.descriptor.value_type
    if value_type not in (ValueType.SCALAR, ValueType.COMPLEX):
        raise ValueError(
            f"fft: unsupported value type {value_type!r}; "
            "only SCALAR and COMPLEX are supported"
        )

    data = numpy.asarray(array.data)
    signal_shape = signal_group.shape          # e.g. (N,) or (N, M)
    signal_axes = tuple(range(-rank, 0))      # e.g. (-1,) or (-2, -1)

    if rank == 1:
        n = signal_shape[0]
        scaling = 1.0 / math.sqrt(n)
        result_data: numpy.typing.NDArray[numpy.complexfloating[typing.Any, typing.Any]] = (
            scipy.fft.fftshift(scipy.fft.fft(data, axis=-1) * scaling, axes=signal_axes)
        )
    else:
        n, m = signal_shape
        scaling = 1.0 / math.sqrt(n * m)
        result_data = scipy.fft.fftshift(
            scipy.fft.fft2(data, axes=signal_axes) * scaling,
            axes=signal_axes,
        )

    new_signal_group = _spatial_to_frequency_axis_group(signal_group)
    new_axis_groups = axis_groups[:-1] + (new_signal_group,)
    new_descriptor = ArrayDescriptor(
        axis_groups=new_axis_groups,
        intensity_calibrations=array.descriptor.intensity_calibrations,
        value_type=ValueType.COMPLEX,
    )
    return AnnotatedArray(data=result_data, descriptor=new_descriptor, metadata=array.metadata)


def ifft(array: AnnotatedArray) -> AnnotatedArray:
    """Compute the inverse FFT of an :class:`AnnotatedArray`.

    The transform is applied to the array's only axis group.

    Energy normalisation
    ~~~~~~~~~~~~~~~~~~~~
    The inverse scaling factor ``sqrt(N)`` (1-D) or ``sqrt(N * M)`` (2-D) is
    applied to be the exact inverse of :func:`fft`.

    DC at centre
    ~~~~~~~~~~~~
    The input is assumed to have its DC component at the array centre
    (produced by :func:`fft`); :func:`scipy.fft.ifftshift` is applied along
    the signal axes before the inverse transform.

    Calibration round-trip
    ~~~~~~~~~~~~~~~~~~~~~~
    All calibrations in the signal :class:`AxisGroup` are transformed back to the
    spatial domain. The keys of all calibrations are preserved exactly, so if the
    frequency-domain array has calibrations keyed ``"spatial"`` and ``"angular"``,
    the result will also have calibrations with those same keys (now with spatial
    scales/offsets/units). The primary calibration key is also preserved.

    Args:
        array: :class:`AnnotatedArray` with a complex datum in frequency
               space (DC at centre).

    Returns:
        :class:`AnnotatedArray` with complex datum and spatial-domain
        calibrations on the signal :class:`AxisGroup`.

    Raises:
        ValueError: If there is not exactly one axis group, or if that group
                    rank is not 1 or 2.
    """
    axis_groups = array.descriptor.axis_groups
    if len(axis_groups) != 1:
        raise ValueError(f"ifft: expected exactly one axis group, got {len(axis_groups)}")

    signal_group = axis_groups[-1]
    rank = signal_group.rank

    if rank not in (1, 2):
        raise ValueError(f"ifft: signal rank must be 1 or 2, got {rank}")

    data = numpy.asarray(array.data)
    signal_shape = signal_group.shape
    signal_axes = tuple(range(-rank, 0))

    if rank == 1:
        n = signal_shape[0]
        scaling = math.sqrt(n)
        result_data = scipy.fft.ifft(
            scipy.fft.ifftshift(data, axes=signal_axes) * scaling,
            axis=-1,
        )
    else:
        n, m = signal_shape
        scaling = math.sqrt(n * m)
        result_data = scipy.fft.ifft2(
            scipy.fft.ifftshift(data, axes=signal_axes) * scaling,
            axes=signal_axes,
        )

    new_signal_group = _frequency_to_spatial_axis_group(signal_group)
    new_axis_groups = axis_groups[:-1] + (new_signal_group,)
    new_descriptor = ArrayDescriptor(
        axis_groups=new_axis_groups,
        intensity_calibrations=array.descriptor.intensity_calibrations,
        value_type=ValueType.COMPLEX,
    )
    return AnnotatedArray(data=result_data, descriptor=new_descriptor, metadata=array.metadata)
