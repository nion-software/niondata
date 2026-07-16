"""Annotated n-dimensional arrays with calibrated, labeled axes.

This module pairs a numpy array with a :class:`DataDescriptor` describing its
axes (grouped into :class:`AxisGroup`/:class:`BoundAxisGroup`), per-axis and
intensity :class:`Calibration` mappings between indices and physical
coordinates, and associated metadata.
"""

from __future__ import annotations

import dataclasses
import datetime
import typing
import types

import numpy
import tzlocal


DEFAULT_CALIBRATION_KEY = "default"
DEFAULT_COORDINATE_MAPPING_KEY = "default"


class Calibration(typing.Protocol):
    """Protocol converting between array indices and physical coordinates."""

    @property
    def unit(self) -> str: ...

    def to_coordinate(self, index: float) -> float: ...
    def to_index(self, coordinate: float) -> float: ...


@dataclasses.dataclass(frozen=True)
class AffineCalibration(Calibration):
    """Linear calibration of the form ``coordinate = offset + index * scale``."""

    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""

    def __post_init__(self) -> None:
        if self.scale == 0.0:
            raise ValueError("AffineCalibration scale must be non-zero")

    def to_coordinate(self, index: float) -> float:
        return self.offset + index * self.scale

    def to_index(self, coordinate: float) -> float:
        return (coordinate - self.offset) / self.scale


@dataclasses.dataclass(frozen=True)
class CalibrationSet:
    """Keyed calibrations with an explicit primary calibration key."""

    calibrations: typing.Mapping[str, Calibration] = dataclasses.field(default_factory=lambda: {DEFAULT_CALIBRATION_KEY: AffineCalibration()})
    primary_key: str = DEFAULT_CALIBRATION_KEY

    def __post_init__(self) -> None:
        calibrations = dict(self.calibrations)
        if not calibrations:
            raise ValueError("calibrations must not be empty")
        if self.primary_key not in calibrations:
            raise ValueError(f"primary_key {self.primary_key!r} is not present in calibrations")
        object.__setattr__(self, "calibrations", types.MappingProxyType(calibrations))

    @property
    def calibration_keys(self) -> tuple[str, ...]:
        return tuple(self.calibrations.keys())

    @property
    def primary(self) -> Calibration:
        return self.calibrations[self.primary_key]

    def has(self, key: str) -> bool:
        return key in self.calibrations

    def get(self, key: str | None = None) -> Calibration:
        target_key = self.primary_key if key is None else key
        calibration = self.calibrations.get(target_key)
        if calibration is None:
            raise KeyError(f"Unknown calibration {target_key!r}")
        return calibration

    @staticmethod
    def from_calibration(calibration: Calibration, key: str = DEFAULT_CALIBRATION_KEY) -> CalibrationSet:
        return CalibrationSet(calibrations={key: calibration}, primary_key=key)

    def with_primary(self, key: str) -> CalibrationSet:
        if key not in self.calibrations:
            raise KeyError(f"Unknown calibration {key!r}")
        return dataclasses.replace(self, primary_key=key)

    def with_calibration(self, key: str, calibration: Calibration, *, make_primary: bool = False) -> CalibrationSet:
        calibrations = dict(self.calibrations)
        calibrations[key] = calibration
        primary_key = key if make_primary else self.primary_key
        return CalibrationSet(calibrations=calibrations, primary_key=primary_key)


@dataclasses.dataclass(frozen=True)
class CoordinateSystemTransform:
    """Invertible affine transform between two coordinate systems."""

    source_coordinate_system_id: str
    destination_coordinate_system_id: str
    matrix: numpy.typing.NDArray[numpy.float64]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be a square 2D affine transform matrix")


@dataclasses.dataclass(frozen=True)
class Axis:
    """A named dimension with a primary calibration and optional auxiliaries."""

    label: str = ""
    calibrations: CalibrationSet = dataclasses.field(default_factory=CalibrationSet)

    def get_calibration(self, key: str | None = None) -> Calibration:
        return self.calibrations.get(key)

    def with_primary_calibration(self, key: str) -> Axis:
        return dataclasses.replace(self, calibrations=self.calibrations.with_primary(key))

    def with_calibration(self, key: str, calibration: Calibration, *, make_primary: bool = False) -> Axis:
        return dataclasses.replace(
            self,
            calibrations=self.calibrations.with_calibration(key, calibration, make_primary=make_primary),
        )

    @property
    def calibration_keys(self) -> tuple[str, ...]:
        return self.calibrations.calibration_keys

    @property
    def primary_calibration_key(self) -> str:
        return self.calibrations.primary_key

    @property
    def unit(self) -> str:
        return self.calibrations.primary.unit

    @classmethod
    def from_calibration(cls, label: str, calibration: Calibration, key: str = DEFAULT_CALIBRATION_KEY) -> Axis:
        return cls(label=label, calibrations=CalibrationSet(calibrations={key: calibration}, primary_key=key))


@dataclasses.dataclass(frozen=True)
class BoundAxis:
    """An :class:`Axis` bound to a concrete (positive) length."""

    axis: Axis
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"BoundAxis size must be positive, got {self.size}")


@dataclasses.dataclass(frozen=True)
class AxisGroup:
    """An ordered group of axes (e.g. spatial or spectral)."""

    axes: tuple[Axis, ...] = dataclasses.field(default_factory=tuple)

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def units(self) -> list[str]:
        return [ax.unit for ax in self.axes]

    def get_calibration(self, axis: int, key: str | None = None) -> Calibration:
        return self.axes[axis].get_calibration(key)


@dataclasses.dataclass(frozen=True)
class BoundAxisGroup:
    """An :class:`AxisGroup` with sized axes, exposing a concrete shape."""

    bound_axes: tuple[BoundAxis, ...] = dataclasses.field(default_factory=tuple)
    coordinate_system_id: str | None = None
    coordinate_mappings: typing.Mapping[str, tuple[Calibration, ...]] = dataclasses.field(default_factory=dict)
    primary_mapping_key: str | None = None
    axis_group: AxisGroup = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_group", AxisGroup(axes=tuple(ba.axis for ba in self.bound_axes)))
        coordinate_mappings = dict(self.coordinate_mappings)
        normalized_coordinate_mappings: dict[str, tuple[Calibration, ...]] = dict()
        for key, mapping in coordinate_mappings.items():
            normalized_mapping = tuple(mapping)
            if len(normalized_mapping) != self.rank:
                raise ValueError(f"coordinate mapping {key!r} rank {len(normalized_mapping)} does not match axis group rank {self.rank}")
            normalized_coordinate_mappings[key] = normalized_mapping
        object.__setattr__(self, "coordinate_mappings", types.MappingProxyType(normalized_coordinate_mappings))

        if self.primary_mapping_key is not None and self.primary_mapping_key not in normalized_coordinate_mappings:
            raise ValueError(f"primary_mapping_key {self.primary_mapping_key!r} is not present in coordinate_mappings")
        if not normalized_coordinate_mappings and self.primary_mapping_key is not None:
            raise ValueError("primary_mapping_key must be None when coordinate_mappings is empty")

    @staticmethod
    def from_1d_size(size: int, *, label: str = "x", unit: str | None = None) -> BoundAxisGroup:
        """Create a 1D bound axis group with one default coordinate mapping."""
        calibration = CalibrationSet.from_calibration(AffineCalibration(unit=unit or ""))
        return BoundAxisGroup(
            bound_axes=(BoundAxis(Axis(label, calibration), size),),
            coordinate_mappings={
                DEFAULT_COORDINATE_MAPPING_KEY: (calibration.primary,),
            },
            primary_mapping_key=DEFAULT_COORDINATE_MAPPING_KEY,
        )

    @staticmethod
    def from_2d_size(size: tuple[int, int], *, labels: tuple[str, str] = ("x", "y"), unit: str | None = None) -> BoundAxisGroup:
        """Create a 2D bound axis group with one default coordinate mapping."""
        size_x, size_y = size
        x_label, y_label = labels
        calibration = CalibrationSet.from_calibration(AffineCalibration(unit=unit or ""), key=DEFAULT_CALIBRATION_KEY)
        return BoundAxisGroup(
            bound_axes=(
                BoundAxis(Axis(x_label, calibration), size_x),
                BoundAxis(Axis(y_label, calibration), size_y),
            ),
            coordinate_mappings={
                DEFAULT_COORDINATE_MAPPING_KEY: (calibration.primary, calibration.primary),
            },
            primary_mapping_key=DEFAULT_COORDINATE_MAPPING_KEY,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(bound_axis.size for bound_axis in self.bound_axes)

    @property
    def rank(self) -> int:
        return len(self.bound_axes)

    @property
    def units(self) -> list[str]:
        if self.primary_mapping_key is None:
            return []
        return [calibration.unit for calibration in self.get_coordinate_mapping()]

    @property
    def mapping_keys(self) -> tuple[str, ...]:
        return tuple(self.coordinate_mappings.keys())

    def get_coordinate_mapping(self, key: str | None = None) -> tuple[Calibration, ...]:
        target_key = self.primary_mapping_key if key is None else key
        if target_key is None:
            raise KeyError("No primary coordinate mapping is designated")
        mapping = self.coordinate_mappings.get(target_key)
        if mapping is None:
            raise KeyError(f"Unknown coordinate mapping {target_key!r}")
        return mapping

    def get_calibration(self, axis: int, key: str | None = None) -> Calibration:
        return self.get_coordinate_mapping(key)[axis]


@dataclasses.dataclass
class DataDescriptor:
    """Metadata for an array: axis sets, intensity calibrations, and properties."""

    bound_axis_groups: list[BoundAxisGroup] = dataclasses.field(default_factory=list)
    intensity_calibrations: CalibrationSet = dataclasses.field(default_factory=CalibrationSet)
    coordinate_system_transforms: list[CoordinateSystemTransform] = dataclasses.field(default_factory=list)
    properties: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    timestamp: datetime.datetime = dataclasses.field(default_factory=lambda: datetime.datetime.now(tz=tzlocal.get_localzone()))

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

    def get_intensity_calibration(self, key: str | None = None) -> Calibration:
        return self.intensity_calibrations.get(key)

    def with_primary_intensity_calibration(self, key: str) -> DataDescriptor:
        return dataclasses.replace(self, intensity_calibrations=self.intensity_calibrations.with_primary(key))

    def with_intensity_calibration(self, key: str, calibration: Calibration, *, make_primary: bool = False) -> DataDescriptor:
        return dataclasses.replace(
            self,
            intensity_calibrations=self.intensity_calibrations.with_calibration(key, calibration, make_primary=make_primary),
        )

    @property
    def intensity_calibration_keys(self) -> tuple[str, ...]:
        return self.intensity_calibrations.calibration_keys

    @property
    def primary_intensity_calibration_key(self) -> str:
        return self.intensity_calibrations.primary_key

    @property
    def timezone(self) -> str | None:
        return self.timestamp.tzinfo.tzname(self.timestamp) if self.timestamp.tzinfo else "UTC"

    @property
    def timezone_offset(self) -> str | None:
        return self.timestamp.strftime("%z") if self.timestamp.tzinfo else "+0000"


@dataclasses.dataclass
class AnnotatedArray:
    """A numpy array paired with a :class:`DataDescriptor` of its axes."""

    data: numpy.typing.NDArray[typing.Any]
    descriptor: DataDescriptor = dataclasses.field(default_factory=DataDescriptor)

    def __post_init__(self) -> None:
        if not self.descriptor.bound_axis_groups:
            raise ValueError("AnnotatedArray requires at least one axis group (bound_axis_group)")
        if any(bound_axis_group.rank == 0 for bound_axis_group in self.descriptor.bound_axis_groups[:-1]):
            raise ValueError("Only the final axis group may have rank 0")
        total_axes = sum(bound_axis_group.rank for bound_axis_group in self.descriptor.bound_axis_groups)
        if total_axes != self.data.ndim:
            raise ValueError(f"Axis groups account for {total_axes} axes but array has {self.data.ndim}")

    def get_intensity_calibration(self, key: str | None = None) -> Calibration:
        return self.descriptor.get_intensity_calibration(key)

    def get_flat_calibrations(self, key: str | None = None) -> list[Calibration]:
        return [bound_axis_group.get_calibration(axis=i, key=key) for bound_axis_group in self.descriptor.bound_axis_groups for i in range(bound_axis_group.rank)]


def zeros_annotated_array(bound_axis_groups: typing.Sequence[BoundAxisGroup], dtype: numpy.typing.DTypeLike = numpy.float64) -> AnnotatedArray:
    shape = tuple(dim for bound_axis_group in bound_axis_groups for dim in bound_axis_group.shape)
    descriptor = DataDescriptor(bound_axis_groups=list(bound_axis_groups))
    return AnnotatedArray(data=numpy.zeros(shape, dtype=dtype), descriptor=descriptor)
