"""Annotated n-dimensional arrays with calibrated, labeled axes.

This module pairs a numpy array with a structural :class:`ArrayDescriptor` and
contextual :class:`ArrayMetadata`. An :class:`ArrayHeader` packages those
objects with the storage data type when the buffer is passed separately. Axes
are grouped into :class:`Axis`/:class:`AxisGroup` objects with per-axis and
intensity :class:`Calibration` mappings.
"""

from __future__ import annotations

import dataclasses
import datetime
import typing
import types

import numpy
import tzlocal


class ValueType:
    """Supported value types for array data."""
    SCALAR = "scalar"
    COMPLEX = "complex"
    RGB = "rgb"
    RGBA = "rgba"
    VECTOR = "vector"


def infer_value_type(dtype: numpy.dtype) -> str:
    """Infer the value type from a numpy dtype.

    Args:
        dtype: The numpy dtype to infer from.

    Returns:
        A ValueType string (scalar, complex, rgb, rgba, or vector).

    Raises:
        ValueError: If the dtype cannot be mapped to a supported value type.
    """
    dtype = numpy.dtype(typing.cast(typing.Any, dtype))

    # Complex types
    if dtype.kind == 'c':
        return ValueType.COMPLEX

    # Structured/record types for RGB and RGBA
    if dtype.names is not None:
        if len(dtype.names) == 3 and set(dtype.names) >= {'r', 'g', 'b'}:
            return ValueType.RGB
        elif len(dtype.names) == 4 and set(dtype.names) >= {'r', 'g', 'b', 'a'}:
            return ValueType.RGBA
        # Other structured types are treated as vectors
        return ValueType.VECTOR

    # Numeric types (int, uint, float, bool)
    if dtype.kind in ('b', 'i', 'u', 'f'):
        return ValueType.SCALAR

    raise ValueError(f"Unsupported dtype {dtype} for value type inference")




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
    """Keyed calibrations with an optional primary calibration key."""

    calibrations: typing.Mapping[str, Calibration] = dataclasses.field(default_factory=dict)
    primary_key: str | None = None

    def __post_init__(self) -> None:
        calibrations = dict(self.calibrations)
        if self.primary_key is not None and self.primary_key not in calibrations:
            raise ValueError(f"primary_key {self.primary_key!r} is not present in calibrations")
        if not calibrations and self.primary_key is not None:
            raise ValueError("primary_key must be None when calibrations is empty")
        object.__setattr__(self, "calibrations", types.MappingProxyType(calibrations))

    @property
    def calibration_keys(self) -> tuple[str, ...]:
        return tuple(self.calibrations.keys())

    @property
    def primary_calibration(self) -> Calibration:
        if self.primary_key is None:
            raise KeyError("No primary calibration is designated")
        return self.calibrations[self.primary_key]

    def has_calibration(self, key: str) -> bool:
        return key in self.calibrations

    def get_calibration(self, key: str | None = None) -> Calibration:
        target_key = self.primary_key if key is None else key
        if target_key is None:
            raise KeyError("No primary calibration is designated")
        calibration = self.calibrations.get(target_key)
        if calibration is None:
            raise KeyError(f"Unknown calibration {target_key!r}")
        return calibration

    @staticmethod
    def from_calibration(calibration: Calibration, key: str) -> CalibrationSet:
        return CalibrationSet(calibrations={key: calibration}, primary_key=key)

    def with_primary_calibration(self, key: str) -> CalibrationSet:
        if key not in self.calibrations:
            raise KeyError(f"Unknown calibration {key!r}")
        return dataclasses.replace(self, primary_key=key)

    def with_calibration(self, key: str, calibration: Calibration, *, make_primary: bool = False) -> CalibrationSet:
        calibrations = dict(self.calibrations)
        calibrations[key] = calibration
        primary_key = key if make_primary else self.primary_key
        return CalibrationSet(calibrations=calibrations, primary_key=primary_key)


@dataclasses.dataclass(frozen=True)
class Axis:
    """A dimension label, size, and coordinate metadata."""

    label: str = ""
    size: int = 1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Axis size must be positive, got {self.size}")


@dataclasses.dataclass(frozen=True)
class CoordinateCalibration:
    """A labeled per-axis coordinate calibration bundle for an axis group."""

    calibrations: tuple[Calibration, ...] = dataclasses.field(default_factory=tuple)
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibrations", tuple(self.calibrations))


@dataclasses.dataclass(frozen=True)
class AxisGroup:
    """An ordered group of sized axes with optional coordinate calibrations."""

    axes: tuple[Axis, ...] = dataclasses.field(default_factory=tuple)
    coordinate_system_id: str | None = None
    coordinate_calibrations: typing.Mapping[str, CoordinateCalibration] = dataclasses.field(default_factory=dict)
    primary_calibration_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        coordinate_calibrations = dict(self.coordinate_calibrations)
        normalized_coordinate_calibrations: dict[str, CoordinateCalibration] = dict()
        for key, coordinate_calibration in coordinate_calibrations.items():
            if len(coordinate_calibration.calibrations) != self.rank:
                raise ValueError(f"coordinate calibration {key!r} rank {len(coordinate_calibration.calibrations)} does not match axis group rank {self.rank}")
            normalized_coordinate_calibrations[key] = coordinate_calibration
        object.__setattr__(self, "coordinate_calibrations", types.MappingProxyType(normalized_coordinate_calibrations))

        if self.primary_calibration_key is not None and self.primary_calibration_key not in normalized_coordinate_calibrations:
            raise ValueError(f"primary_calibration_key {self.primary_calibration_key!r} is not present in coordinate_calibrations")
        if not normalized_coordinate_calibrations and self.primary_calibration_key is not None:
            raise ValueError("primary_calibration_key must be None when coordinate_calibrations is empty")

    @staticmethod
    def from_1d_size(size: int, *, label: str = "x") -> AxisGroup:
        """Create a 1D axis group with no coordinate calibrations."""
        return AxisGroup(axes=(Axis(label, size),))

    @staticmethod
    def from_2d_size(size: tuple[int, int], *, labels: tuple[str, str] = ("x", "y")) -> AxisGroup:
        """Create a 2D axis group with no coordinate calibrations."""
        size_x, size_y = size
        x_label, y_label = labels
        return AxisGroup(
            axes=(
                Axis(x_label, size_x),
                Axis(y_label, size_y),
            ),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def units(self) -> list[str]:
        if self.primary_calibration_key is None:
            return []
        return [calibration.unit for calibration in self.get_coordinate_calibration().calibrations]

    @property
    def calibration_keys(self) -> tuple[str, ...]:
        return tuple(self.coordinate_calibrations.keys())

    def with_coordinate_calibration(self, key: str, coordinate_calibration: CoordinateCalibration) -> AxisGroup:
        """Replace all coordinate calibrations with one calibration and make it primary."""
        return dataclasses.replace(self, coordinate_calibrations={key: coordinate_calibration}, primary_calibration_key=key)

    def with_added_coordinate_calibration(self, key: str, coordinate_calibration: CoordinateCalibration, *, make_primary: bool = False) -> AxisGroup:
        """Add or replace one keyed coordinate calibration."""
        coordinate_calibrations = dict(self.coordinate_calibrations)
        coordinate_calibrations[key] = coordinate_calibration
        primary_calibration_key = key if make_primary else self.primary_calibration_key
        return dataclasses.replace(self, coordinate_calibrations=coordinate_calibrations, primary_calibration_key=primary_calibration_key)

    def get_coordinate_calibration(self, key: str | None = None) -> CoordinateCalibration:
        target_key = self.primary_calibration_key if key is None else key
        if target_key is None:
            raise KeyError("No primary coordinate calibration is designated")
        coordinate_calibration = self.coordinate_calibrations.get(target_key)
        if coordinate_calibration is None:
            raise KeyError(f"Unknown coordinate calibration {target_key!r}")
        return coordinate_calibration

    def get_calibration(self, axis: int, key: str | None = None) -> Calibration:
        return self.get_coordinate_calibration(key).calibrations[axis]


@dataclasses.dataclass(frozen=True)
class ArrayDescriptor:
    """Intrinsic structure and calibrations used to interpret an array."""

    axis_groups: tuple[AxisGroup, ...]
    intensity_calibrations: CalibrationSet = dataclasses.field(default_factory=CalibrationSet)
    value_type: str = ValueType.SCALAR

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_groups", tuple(self.axis_groups))
        if not self.axis_groups:
            raise ValueError("ArrayDescriptor requires at least one axis group")
        if any(axis_group.rank == 0 for axis_group in self.axis_groups[:-1]):
            raise ValueError("Only the final axis group may have rank 0")
        valid_types = {ValueType.SCALAR, ValueType.COMPLEX, ValueType.RGB, ValueType.RGBA, ValueType.VECTOR}
        if self.value_type not in valid_types:
            raise ValueError(f"value_type must be one of {valid_types}, got {self.value_type!r}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(dimension for axis_group in self.axis_groups for dimension in axis_group.shape)

    @property
    def ndim(self) -> int:
        return sum(axis_group.rank for axis_group in self.axis_groups)

    def get_intensity_calibration(self, key: str | None = None) -> Calibration:
        return self.intensity_calibrations.get_calibration(key)

    def with_primary_intensity_calibration(self, key: str) -> ArrayDescriptor:
        return dataclasses.replace(self, intensity_calibrations=self.intensity_calibrations.with_primary_calibration(key))

    def with_intensity_calibration(self, key: str, calibration: Calibration, *, make_primary: bool = False) -> ArrayDescriptor:
        return dataclasses.replace(
            self,
            intensity_calibrations=self.intensity_calibrations.with_calibration(key, calibration, make_primary=make_primary),
        )

    @property
    def intensity_calibration_keys(self) -> tuple[str, ...]:
        return self.intensity_calibrations.calibration_keys

    @property
    def primary_intensity_calibration_key(self) -> str | None:
        return self.intensity_calibrations.primary_key

    def with_value_type(self, value_type: str) -> ArrayDescriptor:
        return dataclasses.replace(self, value_type=value_type)


@dataclasses.dataclass(frozen=True)
class ExtensionRecord:
    """A versioned, encoded payload whose semantics are defined outside this module."""

    extension_type_id: str
    schema_version: int
    payload: str

    def __post_init__(self) -> None:
        if not self.extension_type_id:
            raise ValueError("ExtensionRecord extension_type_id must not be empty")
        if self.schema_version < 1:
            raise ValueError("ExtensionRecord schema_version must be positive")
        if not isinstance(self.payload, str):
            raise TypeError("ExtensionRecord payload must be str")


@dataclasses.dataclass(frozen=True, init=False)
class ArrayMetadata:
    """Context carried with an array but not required to interpret its data."""

    created: datetime.datetime
    attributes: typing.Mapping[str, typing.Any]
    __extensions: typing.Mapping[str, ExtensionRecord] = dataclasses.field(repr=False)

    def __init__(
        self,
        *,
        created: datetime.datetime | None = None,
        attributes: typing.Mapping[str, typing.Any] | None = None,
        extensions: typing.Iterable[ExtensionRecord] = (),
    ) -> None:
        created = created or datetime.datetime.now(tz=tzlocal.get_localzone())
        if created.tzinfo is None or created.utcoffset() is None:
            raise ValueError("created must be timezone-aware")
        extension_map: dict[str, ExtensionRecord] = dict()
        for extension in extensions:
            if extension.extension_type_id in extension_map:
                raise ValueError(f"Duplicate extension type {extension.extension_type_id!r}")
            extension_map[extension.extension_type_id] = extension
        object.__setattr__(self, "created", created)
        object.__setattr__(self, "attributes", types.MappingProxyType(dict(attributes or {})))
        object.__setattr__(self, "_ArrayMetadata__extensions", types.MappingProxyType(extension_map))

    @property
    def extension_type_ids(self) -> tuple[str, ...]:
        return tuple(self.__extensions.keys())

    def has_extension(self, extension_type_id: str) -> bool:
        return extension_type_id in self.__extensions

    def get_extension(self, extension_type_id: str) -> ExtensionRecord:
        extension = self.__extensions.get(extension_type_id)
        if extension is None:
            raise KeyError(f"Unknown extension {extension_type_id!r}")
        return extension

    def with_extension(self, extension: ExtensionRecord) -> ArrayMetadata:
        extensions = dict(self.__extensions)
        extensions[extension.extension_type_id] = extension
        return ArrayMetadata(created=self.created, attributes=self.attributes, extensions=extensions.values())

    def without_extension(self, extension_type_id: str) -> ArrayMetadata:
        extensions = dict(self.__extensions)
        if extension_type_id not in extensions:
            raise KeyError(f"Unknown extension {extension_type_id!r}")
        del extensions[extension_type_id]
        return ArrayMetadata(created=self.created, attributes=self.attributes, extensions=extensions.values())

    def with_attributes(self, attributes: typing.Mapping[str, typing.Any]) -> ArrayMetadata:
        return ArrayMetadata(created=self.created, attributes=attributes, extensions=self.__extensions.values())



@dataclasses.dataclass(frozen=True)
class ArrayHeader:
    """Everything needed to describe an array without its underlying buffer."""

    descriptor: ArrayDescriptor
    dtype: numpy.typing.DTypeLike
    metadata: ArrayMetadata = dataclasses.field(default_factory=ArrayMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", numpy.dtype(typing.cast(typing.Any, self.dtype)))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.descriptor.shape


class ArrayProtocol(typing.Protocol):
    """Duck-array interface required by AnnotatedArray (satisfied by numpy.ndarray, h5py.Dataset, zarr.Array, etc.)."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> numpy.dtype[typing.Any]: ...

    def __array__(self, dtype: typing.Any = None, /, *, copy: bool | None = None) -> numpy.typing.NDArray[typing.Any]: ...


@dataclasses.dataclass
class AnnotatedArray:
    """A numpy array paired with its descriptor and contextual metadata."""

    data: ArrayProtocol
    descriptor: ArrayDescriptor
    metadata: ArrayMetadata = dataclasses.field(default_factory=ArrayMetadata)

    def __post_init__(self) -> None:
        if self.descriptor.shape != self.data.shape:
            raise ValueError(f"Descriptor shape {self.descriptor.shape} does not match array shape {self.data.shape}")
        inferred_value_type = infer_value_type(self.data.dtype)
        if self.descriptor.value_type != inferred_value_type:
            raise ValueError(f"Descriptor value_type {self.descriptor.value_type!r} does not match inferred value_type {inferred_value_type!r} from array dtype {self.data.dtype}")

    def __array__(self, dtype: typing.Any = None, /, *, copy: bool | None = None) -> numpy.typing.NDArray[typing.Any]:
        """Allow AnnotatedArray to be passed directly to numpy functions."""
        return numpy.asarray(self.data, dtype=dtype, copy=copy)

    @property
    def header(self) -> ArrayHeader:
        return ArrayHeader(self.descriptor, self.data.dtype, self.metadata)

    @classmethod
    def from_header(cls, data: ArrayProtocol, header: ArrayHeader) -> AnnotatedArray:
        if header.dtype != data.dtype:
            raise ValueError(f"Header dtype {header.dtype} does not match array dtype {data.dtype}")
        if header.descriptor.shape != data.shape:
            raise ValueError(f"Header shape {header.descriptor.shape} does not match array shape {data.shape}")
        inferred_value_type = infer_value_type(data.dtype)
        if header.descriptor.value_type != inferred_value_type:
            raise ValueError(f"Header value_type {header.descriptor.value_type!r} does not match inferred value_type {inferred_value_type!r} from array dtype {data.dtype}")
        return cls(data, header.descriptor, header.metadata)

    def get_intensity_calibration(self, key: str | None = None) -> Calibration:
        return self.descriptor.get_intensity_calibration(key)

    def get_flat_axis_calibrations(self, key: str | None = None) -> list[Calibration]:
        return [axis_group.get_calibration(axis=i, key=key) for axis_group in self.descriptor.axis_groups for i in range(axis_group.rank)]


def zeros_annotated_array(axis_groups: typing.Sequence[AxisGroup], dtype: numpy.typing.DTypeLike = numpy.float64) -> AnnotatedArray:
    """Create an AnnotatedArray filled with zeros.

    The value_type is inferred from the provided dtype.

    Args:
        axis_groups: Sequence of AxisGroup objects describing the array structure.
        dtype: The numpy dtype for the array. Defaults to float64 (scalar).

    Returns:
        An AnnotatedArray with the specified structure, filled with zeros.
    """
    dtype_obj = numpy.dtype(typing.cast(typing.Any, dtype))
    value_type = infer_value_type(dtype_obj)
    descriptor = ArrayDescriptor(axis_groups=tuple(axis_groups), value_type=value_type)
    return AnnotatedArray(data=numpy.zeros(descriptor.shape, dtype=dtype_obj), descriptor=descriptor)
