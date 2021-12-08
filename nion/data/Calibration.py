from __future__ import annotations

# standard libraries
import math
import typing

import numpy

from nion.data import Image


integer_types = (int,)
_ImageDataType = Image._ImageDataType


class Calibration:

    """
        Represents a transformation from one coordinate system to another.

        Uses a transformation x' = x * scale + offset
    """

    def __init__(self, offset: typing.Optional[float] = None, scale: typing.Optional[float] = None, units: typing.Optional[str] = None) -> None:
        super(Calibration, self).__init__()
        self.__offset = float(offset) if offset else None
        self.__scale = float(scale) if scale else None
        self.__units = str(units) if units else None

    def __repr__(self) -> str:
        if self.__units:
            return "x {} + {} {}".format(self.__scale, self.__offset, self.__units)
        else:
            return "x {} + {}".format(self.__scale, self.__offset)

    def __eq__(self, other: typing.Any) -> bool:
        if isinstance(other, self.__class__):
            return self.offset == other.offset and self.scale == other.scale and self.units == other.units
        return False

    def __ne__(self, other: typing.Any) -> bool:
        if isinstance(other, self.__class__):
            return self.offset != other.offset or self.scale != other.scale or self.units != other.units
        return True

    def __str__(self) -> str:
        return "{0:s} offset:{1:g} scale:{2:g} units:\'{3:s}\'".format(self.__repr__(), self.offset, self.scale, self.units)

    def __copy__(self) -> Calibration:
        return type(self)(self.__offset, self.__scale, self.__units)

    def read_dict(self, storage_dict: typing.Mapping[str, typing.Any]) -> Calibration:
        self.offset = float(storage_dict["offset"]) if "offset" in storage_dict else 0.0
        self.scale = float(storage_dict["scale"]) if "scale" in storage_dict else 1.0
        self.units = storage_dict["units"] if "units" in storage_dict else str()
        return self  # for convenience

    def write_dict(self) -> typing.Dict[str, typing.Union[float, str]]:
        storage_dict: typing.Dict[str, typing.Union[float, str]] = dict()
        storage_dict["offset"] = self.offset
        storage_dict["scale"] = self.scale
        storage_dict["units"] = self.units
        return storage_dict

    @classmethod
    def from_rpc_dict(cls, d: typing.Mapping[str, typing.Any]) -> typing.Optional[Calibration]:
        if d is None:
            return None
        return Calibration(d.get("offset"), d.get("scale"), d.get("units"))

    @property
    def rpc_dict(self) -> typing.Dict[str, typing.Any]:
        d: typing.Dict[str, typing.Union[float, str]] = dict()
        if self.__offset: d["offset"] = self.__offset
        if self.__scale: d["scale"] = self.__scale
        if self.__units: d["units"] = self.__units
        return d

    @property
    def is_calibrated(self) -> bool:
        return self.__offset is not None or self.__scale is not None or self.__units is not None

    def clear(self) -> None:
        self.__offset = None
        self.__scale = None
        self.__units = None

    @property
    def offset(self) -> float:
        return self.__offset if self.__offset else 0.0

    @offset.setter
    def offset(self, value: float) -> None:
        self.__offset = float(value) if value else None

    @property
    def scale(self) -> float:
        return self.__scale if self.__scale else 1.0

    @scale.setter
    def scale(self, value: float) -> None:
        self.__scale = float(value) if value else None

    @property
    def units(self) -> str:
        return self.__units if self.__units else str()

    @units.setter
    def units(self, value: typing.Optional[str]) -> None:
        self.__units = str(value) if value else None

    @typing.overload
    def convert_to_calibrated_value(self, value: float) -> float: ...

    @typing.overload
    def convert_to_calibrated_value(self, value: complex) -> complex: ...

    def convert_to_calibrated_value(self, value: typing.Union[float, complex]) -> typing.Union[float, complex]:
        return self.offset + value * self.scale

    def convert_array_to_calibrated_value(self, value: _ImageDataType) -> _ImageDataType:
        return typing.cast(_ImageDataType, self.offset + value * self.scale)

    @typing.overload
    def convert_to_calibrated_size(self, size: float) -> float: ...

    @typing.overload
    def convert_to_calibrated_size(self, size: complex) -> complex: ...

    def convert_to_calibrated_size(self, size: typing.Union[float, complex]) -> typing.Union[float, complex]:
        return size * self.scale

    def convert_array_to_calibrated_size(self, size: _ImageDataType) -> _ImageDataType:
        return typing.cast(_ImageDataType, size * self.scale)

    def convert_from_calibrated_value(self, value: float) -> float:
        return (value - self.offset) / self.scale

    def convert_from_calibrated_size(self, size: float) -> float:
        return size / self.scale

    def convert_calibrated_value_to_str(self, calibrated_value: typing.Union[_ImageDataType, int, float, complex],
                                        include_units: bool = True,
                                        calibrated_value_range: typing.Optional[typing.Tuple[float, float]] = None,
                                        samples: typing.Optional[int] = None,
                                        units: typing.Optional[str] = None) -> str:
        units = units if units is not None else self.units
        units_str = (" " + units) if include_units and self.__units else ""
        if hasattr(calibrated_value, 'dtype') and not getattr(calibrated_value, "shape"):  # convert NumPy types to Python scalar types
            calibrated_value = calibrated_value.item()  # type: ignore
        if isinstance(calibrated_value, integer_types) or isinstance(calibrated_value, float):
            if calibrated_value_range and samples:
                calibrated_value0 = calibrated_value_range[0]
                calibrated_value1 = calibrated_value_range[1]
                precision = int(max(-math.floor(math.log10(abs(calibrated_value0 - calibrated_value1)/samples + numpy.nextafter(0,1))), 0)) + 1
                result = ("{0:0." + "{0:d}".format(precision) + "f}{1:s}").format(calibrated_value, units_str)
            else:
                result = "{0:g}{1:s}".format(calibrated_value, units_str)
        elif isinstance(calibrated_value, complex):
            result = "{0:g}+{1:g}j{2:s}".format(calibrated_value.real, calibrated_value.imag, units_str)
        elif isinstance(calibrated_value, numpy.ndarray) and numpy.ndim(calibrated_value) == 1 and calibrated_value.shape[0] in (3, 4) and numpy.dtype(calibrated_value.dtype) == numpy.dtype(numpy.uint8):
            result = ", ".join(["{0:d}".format(v) for v in calibrated_value])
        else:
            result = str()
        return result

    def convert_calibrated_size_to_str(self, calibrated_value: typing.Union[_ImageDataType, int, float, complex],
                                       include_units: bool = True,
                                       calibrated_value_range: typing.Optional[typing.Tuple[float, float]] = None,
                                       samples: typing.Optional[int] = None, units: typing.Optional[str] = None) -> str:
        return self.convert_calibrated_value_to_str(calibrated_value, include_units, calibrated_value_range, samples, units)

    def convert_to_calibrated_value_str(self, value: typing.Union[_ImageDataType, int, float, complex],
                                        include_units: bool = True,
                                        value_range: typing.Optional[typing.Tuple[float, float]] = None,
                                        samples: typing.Optional[int] = None, display_inverted: bool = False) -> str:
        if hasattr(value, 'dtype') and not getattr(value, "shape"):  # convert NumPy types to Python scalar types
            value = typing.cast(_ImageDataType, value).item()
        if isinstance(value, integer_types) or isinstance(value, float):
            calibrated_value = self.convert_to_calibrated_value(value)
            if value_range and samples:
                calibrated_value0 = self.convert_to_calibrated_value(value_range[0])
                calibrated_value1 = self.convert_to_calibrated_value(value_range[1])
                if display_inverted and self.units.startswith("1/") and abs(calibrated_value) > 1e-13:
                    return self.convert_calibrated_value_to_str(1 / calibrated_value, include_units, (1/ calibrated_value1, 1/ calibrated_value0), samples, units=self.units[2:])
                else:
                    return self.convert_calibrated_value_to_str(calibrated_value, include_units, (calibrated_value0, calibrated_value1), samples)
            else:
                if display_inverted and self.units.startswith("1/") and abs(calibrated_value) > 1e-13:
                    return self.convert_calibrated_value_to_str(1 / calibrated_value, include_units, units=self.units[2:])
                else:
                    return self.convert_calibrated_value_to_str(calibrated_value, include_units)
        elif isinstance(value, complex):
            calibrated_value_c = self.convert_to_calibrated_value(value)
            return self.convert_calibrated_value_to_str(calibrated_value_c, include_units)
        elif isinstance(value, numpy.ndarray) and numpy.ndim(value) == 1 and value.shape[0] in (3, 4) and numpy.dtype(value.dtype) == numpy.dtype(numpy.uint8):
            result = ", ".join(["{0:d}".format(v) for v in value])
        else:
            result = str()
        return result

    def convert_to_calibrated_size_str(self, size: typing.Union[_ImageDataType, int, float, complex],
                                       include_units: bool = True,
                                       value_range: typing.Optional[typing.Tuple[float, float]] = None,
                                       samples: typing.Optional[int] = None) -> str:
        units_str = (" " + self.units) if include_units and self.__units else ""
        if hasattr(size, 'dtype') and not getattr(size, "shape"):  # convert NumPy types to Python scalar types
            size = typing.cast(_ImageDataType, size).item()
        if isinstance(size, integer_types) or isinstance(size, float):
            calibrated_value = self.convert_to_calibrated_size(size)
            if value_range and samples:
                calibrated_value0 = self.convert_to_calibrated_value(value_range[0])
                calibrated_value1 = self.convert_to_calibrated_value(value_range[1])
                precision = int(max(-math.floor(math.log10(abs(calibrated_value0 - calibrated_value1)/samples + numpy.nextafter(0,1))), 0)) + 1
                result = ("{0:0." + "{0:d}".format(precision) + "f}{1:s}").format(calibrated_value, units_str)
            else:
                result = "{0:g}{1:s}".format(calibrated_value, units_str)
        elif isinstance(size, complex):
            calibrated_value_c = self.convert_to_calibrated_size(size)
            result = "{0:g}+{1:g}j{2:s}".format(calibrated_value_c.real, calibrated_value_c.imag, units_str)
        elif isinstance(size, numpy.ndarray) and numpy.ndim(size) == 1 and size.shape[0] in (3, 4) and numpy.dtype(size.dtype) == numpy.dtype(numpy.uint8):
            result = ", ".join(["{0:d}".format(v) for v in size])
        else:
            result = str()
        return result
