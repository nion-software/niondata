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
import warnings

# typing
from typing import List

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import Image

_ = gettext.gettext


class DataAndMetadata:
    """Represent the ability to calculate data and provide immediate calibrations."""

    def __init__(self, data_fn, data_shape_and_dtype, intensity_calibration=None, dimensional_calibrations=None, metadata=None, timestamp=None):
        self.__data_lock = threading.RLock()
        self.__data_valid = False
        self.__data = None
        self.data_fn = data_fn
        if data_shape_and_dtype is not None and not all([type(data_shape_item) == int for data_shape_item in data_shape_and_dtype[0]]):
            warnings.warn('using a non-integer shape in DataAndMetadata', DeprecationWarning, stacklevel=2)
        self.data_shape_and_dtype = data_shape_and_dtype
        self.intensity_calibration = copy.deepcopy(intensity_calibration) if intensity_calibration else Calibration.Calibration()
        if dimensional_calibrations is None:
            dimensional_calibrations = list()
            for _ in data_shape_and_dtype[0]:
                dimensional_calibrations.append(Calibration.Calibration())
        else:
            dimensional_calibrations = copy.deepcopy(dimensional_calibrations)
        self.dimensional_calibrations = copy.deepcopy(dimensional_calibrations)
        self.timestamp = timestamp if not timestamp else datetime.datetime.utcnow()
        self.metadata = copy.copy(metadata) if metadata is not None else dict()

    @classmethod
    def from_data(cls, data, intensity_calibration=None, dimensional_calibrations=None, metadata=None, timestamp=None):
        data_shape_and_dtype = Image.spatial_shape_from_data(data), data.dtype
        intensity_calibration = intensity_calibration if intensity_calibration is not None else Calibration.Calibration()
        if dimensional_calibrations is None:
            dimensional_calibrations = list()
            for _ in data_shape_and_dtype[0]:
                dimensional_calibrations.append(Calibration.Calibration())
        assert len(dimensional_calibrations) == len(data_shape_and_dtype[0])
        metadata = copy.copy(metadata) if metadata is not None else dict()
        timestamp = timestamp if not timestamp else datetime.datetime.utcnow()
        return cls(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp)

    @classmethod
    def from_rpc_dict(cls, d):
        if d is None:
            return None
        data = numpy.loads(base64.b64decode(d["data"].encode('utf-8')))
        data_shape_and_dtype = Image.spatial_shape_from_data(data), data.dtype
        intensity_calibration = Calibration.from_rpc_dict(d.get("intensity_calibration"))
        if "dimensional_calibrations" in d:
            dimensional_calibrations = [Calibration.from_rpc_dict(dc) for dc in d.get("dimensional_calibrations")]
        else:
            dimensional_calibrations = None
        metadata = d.get("metadata")
        timestamp = datetime.datetime(*list(map(int, re.split('[^\d]', d.get("timestamp"))))) if "timestamp" in d else None
        return DataAndMetadata(lambda: data, data_shape_and_dtype, intensity_calibration, dimensional_calibrations, metadata, timestamp)

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
            d["metadata"] = copy.copy(self.metadata)
        return d

    @property
    def data(self):
        with self.__data_lock:
            if not self.__data_valid:
                self.__data = self.data_fn()
                self.__data_valid = True
        return self.__data

    @property
    def data_shape(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return data_shape_and_dtype[0] if data_shape_and_dtype is not None else None

    @property
    def data_dtype(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return data_shape_and_dtype[1] if data_shape_and_dtype is not None else None

    @property
    def dimensional_shape(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        if data_shape_and_dtype is not None:
            data_shape, data_dtype = self.data_shape_and_dtype
            return Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype)
        return None

    def get_intensity_calibration(self):
        return self.intensity_calibration

    def get_dimensional_calibration(self, index):
        return self.dimensional_calibrations[index]

    @property
    def is_data_1d(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_1d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_2d(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_2d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_3d(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_3d(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgb(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_rgb(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgba(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_rgba(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_rgb_type(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return (Image.is_shape_and_dtype_rgb(*data_shape_and_dtype) or Image.is_shape_and_dtype_rgba(*data_shape_and_dtype)) if data_shape_and_dtype else False

    @property
    def is_data_scalar_type(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_scalar_type(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_complex_type(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_complex_type(*data_shape_and_dtype) if data_shape_and_dtype else False

    @property
    def is_data_bool(self):
        data_shape_and_dtype = self.data_shape_and_dtype
        return Image.is_shape_and_dtype_bool(*data_shape_and_dtype) if data_shape_and_dtype else False

    def get_data_value(self, pos):
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

    @property
    def size_and_data_format_as_string(self):
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

    def __unary_op(self, op):
        def calculate_data():
            return op(self.data)

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               self.metadata, datetime.datetime.utcnow())

    def __binary_op(self, op, other):
        def calculate_data():
            return op(self.data, extract_data(other))

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               self.metadata, datetime.datetime.utcnow())

    def __rbinary_op(self, op, other):
        def calculate_data():
            return op(extract_data(other), self.data)

        return DataAndMetadata(calculate_data,
                               self.data_shape_and_dtype,
                               self.intensity_calibration,
                               self.dimensional_calibrations,
                               self.metadata, datetime.datetime.utcnow())

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

    def __init__(self, value_fn, calibration, metadata, timestamp):
        self.value_fn = value_fn
        self.calibration = calibration
        self.timestamp = timestamp
        self.metadata = copy.deepcopy(metadata)

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
        if "index" in d:
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

    slices = list_to_key(key)

    def calculate_data():
        data = data_and_metadata.data
        return data[slices].copy()

    if data_and_metadata is None:
        return None

    def non_ellipses_count(slices):
        return sum(1 if not isinstance(slice, type(Ellipsis)) else 0 for slice in slices)

    def normalize_slice(index: int, s: slice, shape: List[int], ellipse_count: int):
        size = shape[index] if index < len(shape) else 1
        collapsible = False
        if isinstance(s, type(Ellipsis)):
            # for the ellipse, return a full slice for each ellipse dimension
            slices = list()
            for ellipse_index in range(ellipse_count):
                slices.append((False, slice(0, shape[index + ellipse_index], 1)))
            return slices
        elif isinstance(s, numbers.Integral):
            s = slice(s, s + 1, 1)
            collapsible = True
        elif s is None:
            s = slice(0, size, 1)
        s_start = s.start
        s_stop = s.stop
        s_step = s.step
        s_start = s_start if s_start is not None else 0
        s_start = size + s_start if s_start < 0 else s_start
        s_stop = s_stop if s_stop is not None else size
        s_stop = size + s_stop if s_stop < 0 else s_stop
        s_step = s_step if s_step is not None else 1
        return [(collapsible, slice(s_start, s_stop, s_step))]

    ellipse_count = len(data_and_metadata.data_shape) - non_ellipses_count(slices)
    normalized_slices = list()  # type: List[(bool, slice)]
    for index, s in enumerate(slices):
        normalized_slices.extend(normalize_slice(index, s, data_and_metadata.data_shape, ellipse_count))

    if any(s.start >= s.stop for c, s in normalized_slices):
        return None

    data_shape = [abs(s.start - s.stop) // s.step for c, s in normalized_slices if not c]

    uncollapsed_data_shape = [abs(s.start - s.stop) // s.step for c, s in normalized_slices]

    cropped_dimensional_calibrations = list()

    for index, dimensional_calibration in enumerate(data_and_metadata.dimensional_calibrations):
        if not normalized_slices[index][0]:  # not collapsible
            cropped_calibration = Calibration.Calibration(
                dimensional_calibration.offset + uncollapsed_data_shape[index] * normalized_slices[index][1].start * dimensional_calibration.scale,
                dimensional_calibration.scale / normalized_slices[index][1].step, dimensional_calibration.units)
            cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata(calculate_data,
                           (data_shape, data_and_metadata.data_dtype),
                           data_and_metadata.intensity_calibration, cropped_dimensional_calibrations,
                           data_and_metadata.metadata, datetime.datetime.utcnow())
