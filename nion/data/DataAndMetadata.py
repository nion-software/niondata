# standard libraries
import base64
import copy
import datetime
import gettext
import logging
import numbers
import operator
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

    def __str__(self):
        return ("sequence of " if self.is_sequence else "") + "[" + str(self.collection_dimension_count) + "," + str(self.datum_dimension_count) + "]"

    @property
    def expected_dimension_count(self) -> int:
        return (1 if self.is_sequence else 0) + self.collection_dimension_count + self.datum_dimension_count

    @property
    def is_collection(self) -> bool:
        return self.collection_dimension_count > 0


class DataMetadata:
    """A class describing data metadata, including size, data type, calibrations, the metadata dict, and the creation timestamp."""

    def __init__(self, data_shape_and_dtype, intensity_calibration=None, dimensional_calibrations=None, metadata=None, timestamp=None, data_descriptor=None):
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
        self.dimensional_calibrations = copy.deepcopy(dimensional_calibrations)
        self.timestamp = timestamp if timestamp else datetime.datetime.utcnow()
        self.metadata = copy.deepcopy(metadata) if metadata is not None else dict()

        assert isinstance(self.metadata, dict)
        assert len(dimensional_calibrations) == len(dimensional_shape)

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
    def size_and_data_format_as_string(self) -> str:
        try:
            dimensional_shape = self.dimensional_shape
            data_dtype = self.data_dtype
            if dimensional_shape is not None and data_dtype is not None:
                spatial_shape_str = " x ".join([str(d) for d in dimensional_shape])
                if len(dimensional_shape) == 1:
                    spatial_shape_str += " x 1"
                dtype_names = {
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
    """A class encapsulating a data future and metadata about the data."""

    def __init__(self, data_fn: typing.Callable[[], numpy.ndarray], data_shape_and_dtype: typing.Tuple[ShapeType, numpy.dtype],
                 intensity_calibration: Calibration.Calibration = None, dimensional_calibrations: CalibrationListType = None, metadata: dict = None,
                 timestamp: datetime.datetime = None, data: numpy.ndarray = None, data_descriptor: DataDescriptor=None):
        self.__data_lock = threading.RLock()
        self.__data_valid = data is not None
        self.__data = data
        self.__data_ref_count = 0
        self.unloadable = False
        self.data_fn = data_fn
        assert isinstance(metadata, dict) if metadata is not None else True
        self.__data_metadata = DataMetadata(data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor=data_descriptor)

    @classmethod
    def from_data(cls, data: numpy.ndarray, intensity_calibration: Calibration.Calibration = None, dimensional_calibrations: CalibrationListType = None,
                  metadata: dict = None, timestamp: datetime.datetime = None, data_descriptor: DataDescriptor=None):
        data_shape_and_dtype = (data.shape, data.dtype) if data is not None else None
        return cls(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data, data_descriptor=data_descriptor)

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
        is_sequence = d.get("is_sequence", False)
        collection_dimension_count = d.get("collection_dimension_count")
        datum_dimension_count = d.get("datum_dimension_count")
        if collection_dimension_count is None:
            collection_dimension_count = 2 if len(dimensional_shape) == 3 and not is_sequence else 0
        if datum_dimension_count is None:
            datum_dimension_count = len(dimensional_shape) - collection_dimension_count - (1 if is_sequence else 0)
        data_descriptor = DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
        return DataAndMetadata(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor=data_descriptor)

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
        return self.__data

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
    def intensity_calibration(self) -> Calibration.Calibration:
        return copy.deepcopy(self.__data_metadata.intensity_calibration)

    @property
    def dimensional_calibrations(self) -> CalibrationListType:
        return copy.deepcopy(self.__data_metadata.dimensional_calibrations)

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

    @property
    def timestamp(self) -> datetime.datetime:
        return self.__data_metadata.timestamp

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
        return None

    def __unary_op(self, op):
        def calculate_data():
            return op(self.data)

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               dict(), datetime.datetime.utcnow())

    def __binary_op(self, op, other):
        def calculate_data():
            return op(self.data, extract_data(other))

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               dict(), datetime.datetime.utcnow())

    def __rbinary_op(self, op, other):
        def calculate_data():
            return op(extract_data(other), self.data)

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               dict(), datetime.datetime.utcnow())

    def __abs__(self):
        return self.__unary_op(operator.abs)

    def __neg__(self):
        return self.__unary_op(operator.neg)

    def __pos__(self):
        return self.__unary_op(operator.pos)

    def __add__(self, other):
        return self.__binary_op(operator.add, other)

    def __radd__(self, other):
        return self.__rbinary_op(operator.add, other)

    def __sub__(self, other):
        return self.__binary_op(operator.sub, other)

    def __rsub__(self, other):
        return self.__rbinary_op(operator.sub, other)

    def __mul__(self, other):
        return self.__binary_op(operator.mul, other)

    def __rmul__(self, other):
        return self.__rbinary_op(operator.mul, other)

    def __div__(self, other):
        return self.__binary_op(operator.truediv, other)

    def __rdiv__(self, other):
        return self.__rbinary_op(operator.truediv, other)

    def __truediv__(self, other):
        return self.__binary_op(operator.truediv, other)

    def __rtruediv__(self, other):
        return self.__rbinary_op(operator.truediv, other)

    def __floordiv__(self, other):
        return self.__binary_op(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self.__rbinary_op(operator.floordiv, other)

    def __mod__(self, other):
        return self.__binary_op(operator.mod, other)

    def __rmod__(self, other):
        return self.__rbinary_op(operator.mod, other)

    def __pow__(self, other):
        return self.__binary_op(operator.pow, other)

    def __rpow__(self, other):
        return self.__rbinary_op(operator.pow, other)

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
        return [key[0]]
    return key


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

    def non_ellipses_count(slices):
        return sum(1 if not isinstance(slice, type(Ellipsis)) else 0 for slice in slices)

    def new_axis_count(slices):
        return sum(1 if slice is None else 0 for slice in slices)

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

    data = data_and_metadata.data[slices].copy()

    return new_data_and_metadata(data, data_and_metadata.intensity_calibration, cropped_dimensional_calibrations)


def new_data_and_metadata(data, intensity_calibration: Calibration.Calibration = None, dimensional_calibrations: CalibrationListType = None,
                          metadata: dict = None, timestamp: datetime.datetime = None, data_descriptor: DataDescriptor = None) -> DataAndMetadata:
    return DataAndMetadata.from_data(data, intensity_calibration, dimensional_calibrations, metadata, timestamp, data_descriptor=data_descriptor)
