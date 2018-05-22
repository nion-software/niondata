# standard libraries
import base64
import copy
import datetime
import gettext
import logging
import numbers
import re
import threading
import typing
import warnings

import numpy
from nion.data import Calibration
from nion.data import Image

_ = gettext.gettext


ShapeType = typing.Sequence[int]
Shape2dType = typing.Tuple[int, int]
Shape3dType = typing.Tuple[int, int, int]
PositionType = typing.Sequence[int]
CalibrationListType = typing.Sequence[Calibration.Calibration]


class DataDescriptor:
    "A class describing the layout of data."
    def __init__(self, is_sequence: bool, collection_dimension_count: int, datum_dimension_count: int):
        assert datum_dimension_count in (1, 2)
        assert collection_dimension_count in (0, 1, 2)
        self.is_sequence = is_sequence
        self.collection_dimension_count = collection_dimension_count
        self.datum_dimension_count = datum_dimension_count

    def __repr__(self):
        return ("sequence of " if self.is_sequence else "") + "[" + str(self.collection_dimension_count) + "," + str(self.datum_dimension_count) + "]"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.is_sequence == other.is_sequence and self.collection_dimension_count == other.collection_dimension_count and self.datum_dimension_count == other.datum_dimension_count

    @property
    def expected_dimension_count(self) -> int:
        return (1 if self.is_sequence else 0) + self.collection_dimension_count + self.datum_dimension_count

    @property
    def is_collection(self) -> bool:
        return self.collection_dimension_count > 0

    @property
    def sequence_dimension_index_slice(self) -> slice:
        return slice(0, 1) if self.is_sequence else slice(0, 0)

    @property
    def collection_dimension_index_slice(self) -> slice:
        sequence_dimension_index_slice = self.sequence_dimension_index_slice
        return slice(sequence_dimension_index_slice.stop, sequence_dimension_index_slice.stop + self.collection_dimension_count)

    @property
    def datum_dimension_index_slice(self) -> slice:
        collection_dimension_index_slice = self.collection_dimension_index_slice
        return slice(collection_dimension_index_slice.stop, collection_dimension_index_slice.stop + self.datum_dimension_count)


class DataMetadata:
    """A class describing data metadata, including size, data type, calibrations, the metadata dict, and the creation timestamp.

    Timestamp is UTC string in ISO 8601 format, e.g. 2013-11-17T08:43:21.389391.

    Timezone and timezone are optional. Timezone is the Olson timezone string, e.g. America/Los_Angeles. Timezone offset is
    a string representing hours different from UTC, e.g. +0300 or -0700. Daylight savings can be calculated using the timezone
    string for a given timestamp.

    Values passed to init and set methods are copied before storing. Returned values are return directly and not copied.
    """

    def __init__(self, data_shape_and_dtype, intensity_calibration=None, dimensional_calibrations=None, metadata=None, timestamp=None, data_descriptor=None, timezone=None, timezone_offset=None):
        if data_shape_and_dtype is not None and data_shape_and_dtype[0] is not None and not all([type(data_shape_item) == int for data_shape_item in data_shape_and_dtype[0]]):
            warnings.warn('using a non-integer shape in DataAndMetadata', DeprecationWarning, stacklevel=2)
        self.data_shape_and_dtype = (tuple(data_shape_and_dtype[0]), numpy.dtype(data_shape_and_dtype[1])) if data_shape_and_dtype is not None else None

        dimensional_shape = Image.dimensional_shape_from_shape_and_dtype(data_shape_and_dtype[0], data_shape_and_dtype[1]) if data_shape_and_dtype is not None else ()
        dimension_count = len(dimensional_shape)

        if not data_descriptor:
            is_sequence = False
            collection_dimension_count = 2 if dimension_count in (3, 4) else 0
            datum_dimension_count = dimension_count - collection_dimension_count
            data_descriptor = DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

        assert data_descriptor.expected_dimension_count == dimension_count

        self.data_descriptor = data_descriptor

        self.intensity_calibration = copy.deepcopy(intensity_calibration) if intensity_calibration else Calibration.Calibration()
        if dimensional_calibrations is None:
            dimensional_calibrations = list()
            for _ in dimensional_shape:
                dimensional_calibrations.append(Calibration.Calibration())
        else:
            dimensional_calibrations = copy.deepcopy(dimensional_calibrations)
        self.dimensional_calibrations = dimensional_calibrations
        self.timestamp = timestamp if timestamp else datetime.datetime.utcnow()
        self.timezone = timezone
        self.timezone_offset = timezone_offset
        self.metadata = copy.deepcopy(metadata) if metadata is not None else dict()

        assert isinstance(self.metadata, dict)
        assert len(dimensional_calibrations) == len(dimensional_shape)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.data_shape_and_dtype != other.data_shape_and_dtype:
            return False
        if self.data_descriptor != other.data_descriptor:
            return False
        if self.intensity_calibration != other.intensity_calibration:
            return False
        if self.dimensional_calibrations != other.dimensional_calibrations:
            return False
        if self.timezone != other.timezone:
            return False
        if self.timezone_offset != other.timezone_offset:
            return False
        if self.metadata != other.metadata:
            return False
        return True

    @property
    def data_shape(self) -> ShapeType:
        data_shape_and_dtype = self.data_shape_and_dtype
        return data_shape_and_dtype[0] if data_shape_and_dtype is not None else None

    @property
    def data_dtype(self) -> numpy.dtype:
        data_shape_and_dtype = self.data_shape_and_dtype
        return data_shape_and_dtype[1] if data_shape_and_dtype is not None else None

    @property
    def dimensional_shape(self) -> ShapeType:
        data_shape_and_dtype = self.data_shape_and_dtype
        if data_shape_and_dtype is not None:
            data_shape, data_dtype = self.data_shape_and_dtype
            return Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype)
        return None

    @property
    def is_sequence(self) -> bool:
        return self.data_descriptor.is_sequence

    @property
    def is_collection(self) -> bool:
        return self.data_descriptor.is_collection

    @property
    def collection_dimension_count(self) -> int:
        return self.data_descriptor.collection_dimension_count

    @property
    def datum_dimension_count(self) -> int:
        return self.data_descriptor.datum_dimension_count

    @property
    def max_sequence_index(self) -> int:
        return self.dimensional_shape[0] if self.is_sequence else 0

    @property
    def sequence_dimension_shape(self) -> ShapeType:
        return self.dimensional_shape[self.data_descriptor.sequence_dimension_index_slice]

    @property
    def collection_dimension_shape(self) -> ShapeType:
        return self.dimensional_shape[self.data_descriptor.collection_dimension_index_slice]

    @property
    def datum_dimension_shape(self) -> ShapeType:
        return self.dimensional_shape[self.data_descriptor.datum_dimension_index_slice]

    @property
    def sequence_dimension_index(self) -> int:
        return 0 if self.is_sequence else None

    @property
    def sequence_dimension_slice(self) -> slice:
        return slice(0, 1) if self.is_sequence else None

    @property
    def collection_dimension_indexes(self) -> typing.Sequence[int]:
        return range(1, 1 + self.collection_dimension_count) if self.is_sequence else range(self.collection_dimension_count)

    @property
    def collection_dimension_slice(self) -> slice:
        return slice(1, 1 + self.collection_dimension_count) if self.is_sequence else slice(0, self.collection_dimension_count)

    @property
    def datum_dimension_indexes(self) -> typing.Sequence[int]:
        if self.is_sequence:
            return range(1 + self.collection_dimension_count, 1 + self.collection_dimension_count + self.datum_dimension_count)
        else:
            return range(self.collection_dimension_count, self.collection_dimension_count + self.datum_dimension_count)

    @property
    def datum_dimension_slice(self) -> slice:
        if self.is_sequence:
            return slice(1 + self.collection_dimension_count, 1 + self.collection_dimension_count + self.datum_dimension_count)
        else:
            return slice(self.collection_dimension_count, self.collection_dimension_count + self.datum_dimension_count)

    @property
    def sequence_dimensional_calibration(self) -> Calibration.Calibration:
        return self.dimensional_calibrations[self.data_descriptor.sequence_dimension_index_slice] if self.is_sequence else None

    @property
    def collection_dimensional_calibrations(self) -> CalibrationListType:
        return self.dimensional_calibrations[self.data_descriptor.collection_dimension_index_slice]

    @property
    def datum_dimensional_calibrations(self) -> CalibrationListType:
        return self.dimensional_calibrations[self.data_descriptor.datum_dimension_index_slice]

    def get_intensity_calibration(self) -> Calibration.Calibration:
        return self.intensity_calibration

    def get_dimensional_calibration(self, index) -> Calibration.Calibration:
        return self.dimensional_calibrations[index]

    def _set_intensity_calibration(self, intensity_calibration: Calibration.Calibration) -> None:
        self.intensity_calibration = copy.deepcopy(intensity_calibration)

    def _set_dimensional_calibrations(self, dimensional_calibrations: CalibrationListType) -> None:
        self.dimensional_calibrations = copy.deepcopy(dimensional_calibrations)

    def _set_metadata(self, metadata: dict) -> None:
        self.metadata = copy.deepcopy(metadata)

    def _set_timestamp(self, timestamp: datetime.datetime) -> None:
        self.timestamp = timestamp

    @property
    def is_data_1d(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_1d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_2d(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_2d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_3d(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_3d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_4d(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_4d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgb(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_rgb(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgba(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_rgba(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgb_type(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return (Image.is_shape_and_dtype_rgb(*data_shape_and_dtype) or Image.is_shape_and_dtype_rgba(*data_shape_and_dtype)) if data_shape_and_dtype else False

    @property
    def is_data_scalar_type(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_scalar_type(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_complex_type(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_complex_type(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_bool(self) -> bool:
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_bool(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_datum_1d(self) -> bool:
        if self.datum_dimension_count == 1:
            return True
        if self.datum_dimension_count == 0 and self.collection_dimension_count == 1:
            return True
        if self.datum_dimension_count == 0 and self.collection_dimension_count == 0 and self.is_sequence:
            return True
        return False

    @property
    def is_datum_2d(self) -> bool:
        if self.datum_dimension_count == 2:
            return True
        if self.datum_dimension_count == 0 and self.collection_dimension_count == 2:
            return True
        return False

    @property
    def size_and_data_format_as_string(self) -> str:
        try:
            dimensional_shape = self.dimensional_shape
            data_dtype = self.data_dtype
            if dimensional_shape is not None and data_dtype is not None:
                spatial_shape_str = " x ".join([str(d) for d in dimensional_shape])
                if len(dimensional_shape) == 1:
                    spatial_shape_str += " x 1"
                dtype_names = {
                    numpy.bool: _("Boolean (1-bit)"),
                    numpy.bool_: _("Boolean (1-bit)"),
                    numpy.int8: _("Integer (8-bit)"),
                    numpy.int16: _("Integer (16-bit)"),
                    numpy.int32: _("Integer (32-bit)"),
                    numpy.int64: _("Integer (64-bit)"),
                    numpy.uint8: _("Unsigned Integer (8-bit)"),
                    numpy.uint16: _("Unsigned Integer (16-bit)"),
                    numpy.uint32: _("Unsigned Integer (32-bit)"),
                    numpy.uint64: _("Unsigned Integer (64-bit)"),
                    numpy.float32: _("Real (32-bit)"),
                    numpy.float64: _("Real (64-bit)"),
                    numpy.complex64: _("Complex (2 x 32-bit)"),
                    numpy.complex128: _("Complex (2 x 64-bit)"),
                }
                if self.is_data_rgb_type:
                    data_size_and_data_format_as_string = _("RGB (8-bit)") if self.is_data_rgb else _("RGBA (8-bit)")
                else:
                    if not self.data_dtype.type in dtype_names:
                        logging.debug("Unknown dtype %s", self.data_dtype.type)
                    data_size_and_data_format_as_string = dtype_names[self.data_dtype.type] if self.data_dtype.type in dtype_names else _("Unknown Data Type")
                return "{0}, {1}".format(spatial_shape_str, data_size_and_data_format_as_string)
            return _("No Data")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


class DataAndMetadata:
    """A class encapsulating a data future and metadata about the data.

    Timestamp is UTC string in ISO 8601 format, e.g. 2013-11-17T08:43:21.389391.

    Timezone and timezone are optional. Timezone is the Olson timezone string, e.g. America/Los_Angeles. Timezone offset is
    a string representing hours different from UTC, e.g. +0300 or -0700. Daylight savings can be calculated using the timezone
    string for a given timestamp.

    Value other than data that are passed to init and set methods are copied before storing. Returned values are return
    directly and not copied.
    """

    def __init__(self, data_fn: typing.Callable[[], numpy.ndarray], data_shape_and_dtype: typing.Tuple[ShapeType, numpy.dtype],
                 intensity_calibration: Calibration.Calibration = None, dimensional_calibrations: CalibrationListType = None, metadata: dict = None,
                 timestamp: datetime.datetime = None, data: numpy.ndarray = None, data_descriptor: DataDescriptor=None,
                 timezone: str = None, timezone_offset: str = None):
        self.__data_lock = threading.RLock()
        self.__data_valid = data is not None
        self.__data = data
        self.__data_ref_count = 0
        self.unloadable = False
        self.data_fn = data_fn
        assert isinstance(metadata, dict) if metadata is not None else True
        self.__data_metadata = DataMetadata(data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor=data_descriptor,
                                            timezone=timezone, timezone_offset=timezone_offset)

    def __deepcopy__(self, memo):
        # use numpy.copy so that it handles h5py arrays too (resulting in ndarray).
        deepcopy = DataAndMetadata.from_data(numpy.copy(self.data), self.intensity_calibration, self.dimensional_calibrations, self.metadata, self.timestamp, self.data_descriptor, self.timezone, self.timezone_offset)
        memo[id(self)] = deepcopy
        return deepcopy

    def __array__(self, dtype=None):
        return self.data.__array__(dtype)

    @classmethod
    def from_data(cls, data: numpy.ndarray, intensity_calibration: Calibration.Calibration = None, dimensional_calibrations: CalibrationListType = None,
                  metadata: dict = None, timestamp: datetime.datetime = None, data_descriptor: DataDescriptor=None, timezone: str = None, timezone_offset: str = None):
        """Return a new data and metadata from an ndarray. Takes ownership of data."""
        data_shape_and_dtype = (data.shape, data.dtype) if data is not None else None
        return cls(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data, data_descriptor=data_descriptor, timezone=timezone, timezone_offset=timezone_offset)

    @classmethod
    def from_rpc_dict(cls, d):
        if d is None:
            return None
        data = numpy.loads(base64.b64decode(d["data"].encode('utf-8')))
        dimensional_shape = Image.dimensional_shape_from_data(data)
        data_shape_and_dtype = data.shape, data.dtype
        intensity_calibration = Calibration.Calibration.from_rpc_dict(d.get("intensity_calibration"))
        if "dimensional_calibrations" in d:
            dimensional_calibrations = [Calibration.Calibration.from_rpc_dict(dc) for dc in d.get("dimensional_calibrations")]
        else:
            dimensional_calibrations = None
        metadata = d.get("metadata")
        timestamp = datetime.datetime(*list(map(int, re.split('[^\d]', d.get("timestamp"))))) if "timestamp" in d else None
        timezone = d.get("timezone")
        timezone_offset = d.get("timezone_offset")
        is_sequence = d.get("is_sequence", False)
        collection_dimension_count = d.get("collection_dimension_count")
        datum_dimension_count = d.get("datum_dimension_count")
        if collection_dimension_count is None:
            collection_dimension_count = 2 if len(dimensional_shape) == 3 and not is_sequence else 0
        if datum_dimension_count is None:
            datum_dimension_count = len(dimensional_shape) - collection_dimension_count - (1 if is_sequence else 0)
        data_descriptor = DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
        return DataAndMetadata(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor=data_descriptor, timezone=timezone, timezone_offset=timezone_offset)

    @property
    def rpc_dict(self):
        d = dict()
        data = self.data
        if data is not None:
            d["data"] = base64.b64encode(numpy.ndarray.dumps(data)).decode('utf=8')
        if self.intensity_calibration:
            d["intensity_calibration"] = self.intensity_calibration.rpc_dict
        if self.dimensional_calibrations:
            d["dimensional_calibrations"] = [dimensional_calibration.rpc_dict for dimensional_calibration in self.dimensional_calibrations]
        if self.timestamp:
            d["timestamp"] = self.timestamp.isoformat()
        if self.timezone:
            d["timezone"] = self.timezone
        if self.timezone_offset:
            d["timezone_offset"] = self.timezone_offset
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        d["is_sequence"] = self.is_sequence
        d["collection_dimension_count"] = self.collection_dimension_count
        d["datum_dimension_count"] = self.datum_dimension_count
        return d

    @property
    def is_data_valid(self) -> bool:
        return self.__data_valid

    @property
    def data(self) -> numpy.ndarray:
        self.increment_data_ref_count()
        try:
            return self.__data
        finally:
            self.decrement_data_ref_count()

    @property
    def data_if_loaded(self) -> bool:
        return self.__data is not None

    def increment_data_ref_count(self) -> int:
        with self.__data_lock:
            initial_count = self.__data_ref_count
            self.__data_ref_count += 1
            if initial_count == 0 and not self.__data_valid:
                self.__data = self.data_fn()
                self.__data_valid = True
        return initial_count+1

    def decrement_data_ref_count(self) -> int:
        with self.__data_lock:
            assert self.__data_ref_count > 0
            self.__data_ref_count -= 1
            final_count = self.__data_ref_count
            if final_count == 0 and self.unloadable:
                self.__data = None
                self.__data_valid = False
        return final_count

    def clone_with_data(self, data: numpy.ndarray) -> "DataAndMetadata":
        return new_data_and_metadata(data, intensity_calibration=self.intensity_calibration, dimensional_calibrations=self.dimensional_calibrations, data_descriptor=self.data_descriptor)

    @property
    def data_shape_and_dtype(self) -> typing.Tuple[ShapeType, numpy.dtype]:
        return self.__data_metadata.data_shape_and_dtype

    @property
    def data_metadata(self) -> DataMetadata:
        return self.__data_metadata

    @property
    def data_shape(self) -> ShapeType:
        return self.__data_metadata.data_shape

    @property
    def data_dtype(self) -> numpy.dtype:
        return self.__data_metadata.data_dtype

    @property
    def dimensional_shape(self) -> ShapeType:
        return self.__data_metadata.dimensional_shape

    @property
    def data_descriptor(self) -> DataDescriptor:
        return self.__data_metadata.data_descriptor

    @property
    def is_sequence(self) -> bool:
        return self.__data_metadata.is_sequence

    @property
    def is_collection(self) -> bool:
        return self.__data_metadata.is_collection

    @property
    def collection_dimension_count(self) -> int:
        return self.__data_metadata.collection_dimension_count

    @property
    def datum_dimension_count(self) -> int:
        return self.__data_metadata.datum_dimension_count

    @property
    def max_sequence_index(self) -> int:
        return self.__data_metadata.max_sequence_index

    @property
    def sequence_dimension_shape(self) -> ShapeType:
        return self.__data_metadata.sequence_dimension_shape

    @property
    def collection_dimension_shape(self) -> ShapeType:
        return self.__data_metadata.collection_dimension_shape

    @property
    def datum_dimension_shape(self) -> ShapeType:
        return self.__data_metadata.datum_dimension_shape

    @property
    def sequence_dimension_index(self) -> int:
        return self.__data_metadata.sequence_dimension_index

    @property
    def sequence_dimension_slice(self) -> slice:
        return self.__data_metadata.sequence_dimension_slice

    @property
    def collection_dimension_indexes(self) -> typing.Sequence[int]:
        return self.__data_metadata.collection_dimension_indexes

    @property
    def collection_dimension_slice(self) -> slice:
        return self.__data_metadata.collection_dimension_slice

    @property
    def datum_dimension_indexes(self) -> typing.Sequence[int]:
        return self.__data_metadata.datum_dimension_indexes

    @property
    def datum_dimension_slice(self) -> slice:
        return self.__data_metadata.datum_dimension_slice

    @property
    def sequence_dimensional_calibration(self) -> Calibration.Calibration:
        return self.__data_metadata.sequence_dimensional_calibration

    @property
    def collection_dimensional_calibrations(self) -> CalibrationListType:
        return self.__data_metadata.collection_dimensional_calibrations

    @property
    def datum_dimensional_calibrations(self) -> CalibrationListType:
        return self.__data_metadata.datum_dimensional_calibrations

    @property
    def intensity_calibration(self) -> Calibration.Calibration:
        return self.__data_metadata.intensity_calibration

    @property
    def dimensional_calibrations(self) -> CalibrationListType:
        return self.__data_metadata.dimensional_calibrations

    @property
    def metadata(self) -> dict:
        return self.__data_metadata.metadata

    def _set_data(self, data: numpy.ndarray) -> None:
        self.__data = data
        self.__data_valid = True

    def _add_data_ref_count(self, data_ref_count: int) -> None:
        with self.__data_lock:
            self.__data_ref_count += data_ref_count

    def _set_intensity_calibration(self, intensity_calibration: Calibration.Calibration) -> None:
        self.__data_metadata._set_intensity_calibration(intensity_calibration)

    def _set_dimensional_calibrations(self, dimensional_calibrations: CalibrationListType) -> None:
        self.__data_metadata._set_dimensional_calibrations(dimensional_calibrations)

    def _set_metadata(self, metadata: dict) -> None:
        self.__data_metadata._set_metadata(metadata)

    def _set_timestamp(self, timestamp: datetime.datetime) -> None:
        self.__data_metadata._set_timestamp(timestamp)

    @property
    def timestamp(self) -> datetime.datetime:
        return self.__data_metadata.timestamp

    @property
    def timezone(self) -> str:
        return self.__data_metadata.timezone

    @property
    def timezone_offset(self) -> str:
        return self.__data_metadata.timezone_offset

    @property
    def is_data_1d(self) -> bool:
        return self.__data_metadata.is_data_1d

    @property
    def is_data_2d(self) -> bool:
        return self.__data_metadata.is_data_2d

    @property
    def is_data_3d(self) -> bool:
        return self.__data_metadata.is_data_3d

    @property
    def is_data_4d(self) -> bool:
        return self.__data_metadata.is_data_4d

    @property
    def is_data_rgb(self) -> bool:
        return self.__data_metadata.is_data_rgb

    @property
    def is_data_rgba(self) -> bool:
        return self.__data_metadata.is_data_rgba

    @property
    def is_data_rgb_type(self) -> bool:
        return self.__data_metadata.is_data_rgb_type

    @property
    def is_data_scalar_type(self) -> bool:
        return self.__data_metadata.is_data_scalar_type

    @property
    def is_data_complex_type(self) -> bool:
        return self.__data_metadata.is_data_complex_type

    @property
    def is_data_bool(self) -> bool:
        return self.__data_metadata.is_data_bool

    @property
    def is_datum_1d(self) -> bool:
        return self.__data_metadata.is_datum_1d

    @property
    def is_datum_2d(self) -> bool:
        return self.__data_metadata.is_datum_2d

    @property
    def size_and_data_format_as_string(self) -> str:
        return self.__data_metadata.size_and_data_format_as_string

    def get_intensity_calibration(self) -> Calibration.Calibration:
        return self.intensity_calibration

    def get_dimensional_calibration(self, index) -> Calibration.Calibration:
        return self.dimensional_calibrations[index]

    def get_data_value(self, pos: ShapeType) -> typing.Any:
        data = self.data
        if self.is_data_1d:
            if data is not None:
                return data[int(pos[0])]
        elif self.is_data_2d:
            if data is not None:
                return data[int(pos[0]), int(pos[1])]
        elif self.is_data_3d:
            if data is not None:
                return data[int(pos[0]), int(pos[1]), int(pos[2])]
        elif self.is_data_4d:
            if data is not None:
                return data[int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        return None

    def __unary_op(self, op):
        return new_data_and_metadata(op(self.data), intensity_calibration=self.intensity_calibration, dimensional_calibrations=self.dimensional_calibrations)

    def __binary_op(self, op, other):
        return new_data_and_metadata(op(self.data, extract_data(other)), intensity_calibration=self.intensity_calibration, dimensional_calibrations=self.dimensional_calibrations)

    def __rbinary_op(self, op, other):
        return new_data_and_metadata(op(extract_data(other), self.data), intensity_calibration=self.intensity_calibration, dimensional_calibrations=self.dimensional_calibrations)

    def __abs__(self):
        return self.__unary_op(numpy.abs)

    def __neg__(self):
        return self.__unary_op(numpy.negative)

    def __pos__(self):
        return self.__unary_op(numpy.positive)

    def __add__(self, other):
        return self.__binary_op(numpy.add, other)

    def __radd__(self, other):
        return self.__rbinary_op(numpy.add, other)

    def __sub__(self, other):
        return self.__binary_op(numpy.subtract, other)

    def __rsub__(self, other):
        return self.__rbinary_op(numpy.subtract, other)

    def __mul__(self, other):
        return self.__binary_op(numpy.multiply, other)

    def __rmul__(self, other):
        return self.__rbinary_op(numpy.multiply, other)

    def __div__(self, other):
        return self.__binary_op(numpy.divide, other)

    def __rdiv__(self, other):
        return self.__rbinary_op(numpy.divide, other)

    def __truediv__(self, other):
        return self.__binary_op(numpy.divide, other)

    def __rtruediv__(self, other):
        return self.__rbinary_op(numpy.divide, other)

    def __floordiv__(self, other):
        return self.__binary_op(numpy.floor_divide, other)

    def __rfloordiv__(self, other):
        return self.__rbinary_op(numpy.floor_divide, other)

    def __mod__(self, other):
        return self.__binary_op(numpy.mod, other)

    def __rmod__(self, other):
        return self.__rbinary_op(numpy.mod, other)

    def __pow__(self, other):
        return self.__binary_op(numpy.power, other)

    def __rpow__(self, other):
        return self.__rbinary_op(numpy.power, other)

    def __complex__(self):
        raise Exception("Use astype(data, complex128) instead.")

    def __int__(self):
        raise Exception("Use astype(data, int) instead.")

    def __long__(self):
        raise Exception("Use astype(data, int64) instead.")

    def __float__(self):
        raise Exception("Use astype(data, float64) instead.")

    def __getitem__(self, key):
        return function_data_slice(self, key_to_list(key))


class ScalarAndMetadata:
    """Represent the ability to calculate data and provide immediate calibrations."""

    def __init__(self, value_fn, calibration, metadata=None, timestamp=None):
        self.value_fn = value_fn
        self.calibration = calibration
        self.timestamp = timestamp if not timestamp else datetime.datetime.utcnow()
        self.metadata = copy.deepcopy(metadata) if metadata is not None else dict()

    @classmethod
    def from_value(cls, value):
        calibration = Calibration.Calibration()
        metadata = dict()
        timestamp = datetime.datetime.utcnow()
        return cls(lambda: value, calibration, metadata, timestamp)

    @classmethod
    def from_value_fn(cls, value_fn):
        calibration = Calibration.Calibration()
        metadata = dict()
        timestamp = datetime.datetime.utcnow()
        return cls(value_fn, calibration, metadata, timestamp)

    @property
    def value(self):
        return self.value_fn()


def is_equal(left, right):
    if left is right:
        return True
    if (left is None) != (right is None):
        return False
    if not isinstance(right, left.__class__):
        return False
    if not left.data_metadata == right.data_metadata:
        return False
    return numpy.array_equal(left.data, right.data)


def extract_data(evaluated_input):
    if isinstance(evaluated_input, DataAndMetadata):
        return evaluated_input.data
    if isinstance(evaluated_input, ScalarAndMetadata):
        return evaluated_input.value
    return evaluated_input


def key_to_list(key):
    if not isinstance(key, tuple):
        key = (key, )
    l = list()
    for k in key:
        if isinstance(k, slice):
            d = dict()
            if k.start is not None:
                d["start"] = k.start
            if k.stop is not None:
                d["stop"] = k.stop
            if k.step is not None:
                d["step"] = k.step
            l.append(d)
        elif isinstance(k, numbers.Integral):
            l.append({"index": k})
        elif isinstance(k, type(Ellipsis)):
            l.append({"ellipses": True})
        elif k is None:
            l.append({"newaxis": True})
        else:
            print(type(k))
            assert False
    return l


def list_to_key(l):
    key = list()
    for d in l:
        if isinstance(d, (slice, type(Ellipsis))):
            key.append(d)
        elif d is None:
            key.append(None)
        elif isinstance(d, numbers.Integral):
            key.append(d)
        elif "index" in d:
            key.append(d.get("index"))
        elif d.get("ellipses", False):
            key.append(Ellipsis)
        elif d.get("newaxis", False):
            key.append(None)
        else:
            key.append(slice(d.get("start"), d.get("stop"), d.get("step")))
    if len(key) == 1:
        return (key[0], )
    return tuple(key)


def function_data_slice(data_and_metadata, key):
    """Slice data.

    a[2, :]

    Keeps calibrations.
    """

    # (4, 8, 8)[:, 4, 4]
    # (4, 8, 8)[:, :, 4]
    # (4, 8, 8)[:, 4:4, 4]
    # (4, 8, 8)[:, 4:5, 4]
    # (4, 8, 8)[2, ...]
    # (4, 8, 8)[..., 2]
    # (4, 8, 8)[2, ..., 2]

    if data_and_metadata is None:
        return None

    data_and_metadata = promote_ndarray(data_and_metadata)

    def non_ellipses_count(slices):
        return sum(1 if not isinstance(slice, type(Ellipsis)) else 0 for slice in slices)

    def new_axis_count(slices):
        return sum(1 if slice is None else 0 for slice in slices)

    def ellipses_count(slices):
        return sum(1 if isinstance(slice, type(Ellipsis)) else 0 for slice in slices)

    def normalize_slice(index: int, s: slice, shape: typing.List[int], ellipse_count: int):
        size = shape[index] if index < len(shape) else 1
        is_collapsible = False  # if the index is fixed, it will disappear in final data
        is_new_axis = False
        if isinstance(s, type(Ellipsis)):
            # for the ellipse, return a full slice for each ellipse dimension
            slices = list()
            for ellipse_index in range(ellipse_count):
                slices.append((False, False, slice(0, shape[index + ellipse_index], 1)))
            return slices
        elif isinstance(s, numbers.Integral):
            s = slice(s, s + 1, 1)
            is_collapsible = True
        elif s is None:
            s = slice(0, size, 1)
            is_new_axis = True
        s_start = s.start
        s_stop = s.stop
        s_step = s.step
        s_start = s_start if s_start is not None else 0
        s_start = size + s_start if s_start < 0 else s_start
        s_stop = s_stop if s_stop is not None else size
        s_stop = size + s_stop if s_stop < 0 else s_stop
        s_step = s_step if s_step is not None else 1
        return [(is_collapsible, is_new_axis, slice(s_start, s_stop, s_step))]

    slices = list_to_key(key)

    if ellipses_count(slices) == 0 and len(slices) < len(data_and_metadata.dimensional_shape):
        slices = slices + (Ellipsis, )

    ellipse_count = len(data_and_metadata.data_shape) - non_ellipses_count(slices) + new_axis_count(slices)  # how many slices go into the ellipse
    normalized_slices = list()  # type: typing.List[(bool, bool, slice)]
    slice_index = 0
    for s in slices:
        new_normalized_slices = normalize_slice(slice_index, s, data_and_metadata.data_shape, ellipse_count)
        normalized_slices.extend(new_normalized_slices)
        for normalized_slice in new_normalized_slices:
            if not normalized_slice[1]:
                slice_index += 1

    if any(s.start >= s.stop for c, n, s in normalized_slices):
        return None

    cropped_dimensional_calibrations = list()

    dimensional_calibration_index = 0
    for slice_index, dimensional_calibration in enumerate(normalized_slices):
        normalized_slice = normalized_slices[slice_index]
        if normalized_slice[0]:  # if_collapsible
            dimensional_calibration_index += 1
        else:
            if normalized_slice[1]:  # is_newaxis
                cropped_calibration = Calibration.Calibration()
            else:
                dimensional_calibration = data_and_metadata.dimensional_calibrations[dimensional_calibration_index]
                cropped_calibration = Calibration.Calibration(
                    dimensional_calibration.offset + normalized_slice[2].start * dimensional_calibration.scale,
                    dimensional_calibration.scale / normalized_slice[2].step, dimensional_calibration.units)
                dimensional_calibration_index += 1
            cropped_dimensional_calibrations.append(cropped_calibration)

    is_sequence = data_and_metadata.data_descriptor.is_sequence
    collection_dimension_count = data_and_metadata.data_descriptor.collection_dimension_count
    datum_dimension_count = data_and_metadata.data_descriptor.datum_dimension_count

    # print(f"slices {slices}  {data_and_metadata.data_descriptor}")

    skip = False

    if isinstance(slices[0], type(Ellipsis)):
        skip = True

    if not skip and isinstance(slices[0], numbers.Integral):
        # print("s")
        is_sequence = False

    for collection_dimension_index in data_and_metadata.collection_dimension_indexes:
        # print(f"c {collection_dimension_index}")
        if skip:
            # print("skipping")
            break
        if isinstance(slices[collection_dimension_index], type(Ellipsis)):
            # print("ellipsis")
            skip = True
        elif isinstance(slices[collection_dimension_index], numbers.Integral):
            # print("integral")
            collection_dimension_count -= 1
        elif slices[collection_dimension_index] is None:
            # print("newaxis")
            if collection_dimension_index == 0 and not is_sequence:
                is_sequence = True
            else:
                collection_dimension_count += 1

    for datum_dimension_index in data_and_metadata.datum_dimension_indexes:
        # print(f"d {datum_dimension_index}")
        if skip:
            # print("skipping")
            break
        if isinstance(slices[datum_dimension_index], type(Ellipsis)):
            # print("ellipsis")
            skip = True
        elif isinstance(slices[datum_dimension_index], numbers.Integral):
            # print("integral")
            datum_dimension_count -= 1
        elif slices[datum_dimension_index] is None:
            # print("newaxis")
            if datum_dimension_index == 0 and not is_sequence:
                is_sequence = True
            elif datum_dimension_count >= 2:
                collection_dimension_count += 1
            else:
                datum_dimension_count += 1

    if skip and slices[-1] is None:  # case of adding newaxis after ellipsis
        # print("adding datum, newaxis")
        datum_dimension_count += 1

    if datum_dimension_count == 0:  # case where datum has been sliced
        # print("collection to datum")
        datum_dimension_count = collection_dimension_count
        collection_dimension_count = 0

    data = data_and_metadata.data[slices].copy()
    # print(f"was {new_data_and_metadata(data, data_and_metadata.intensity_calibration, cropped_dimensional_calibrations).data_descriptor}")
    # print(f"now [{is_sequence if is_sequence else ''}{collection_dimension_count},{datum_dimension_count}]")

    data_descriptor = DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
    # print(f"data descriptor {data_descriptor}")

    return new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations, data_descriptor=data_descriptor)


def promote_ndarray(data):
    if isinstance(data, DataAndMetadata):
        return data
    if isinstance(data, numpy.ndarray):
        return new_data_and_metadata(data)
    if hasattr(data, "__array__"):
        return new_data_and_metadata(data)
    return data


def determine_shape(*datas):
    for data in datas:
        if data is not None and hasattr(data, "data_shape"):
            return data.data_shape
    return None


def promote_constant(data, shape):
    if isinstance(data, DataAndMetadata):
        return data
    if isinstance(data, numpy.ndarray):
        return new_data_and_metadata(data)
    if data is not None:
        return new_data_and_metadata(numpy.full(shape, data))
    return None


def new_data_and_metadata(data,
                          intensity_calibration: Calibration.Calibration = None,
                          dimensional_calibrations: CalibrationListType = None,
                          metadata: dict = None,
                          timestamp: datetime.datetime = None,
                          data_descriptor: DataDescriptor = None,
                          timezone: str = None,
                          timezone_offset: str = None) -> DataAndMetadata:
    """Return a new data and metadata from an ndarray. Takes ownership of data."""
    return DataAndMetadata.from_data(data, intensity_calibration, dimensional_calibrations, metadata, timestamp=timestamp, timezone=timezone, timezone_offset=timezone_offset, data_descriptor=data_descriptor)
