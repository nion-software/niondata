# standard libraries
import datetime
import math

# third party libraries
import numpy
import scipy
import scipy.fftpack
import scipy.ndimage
import scipy.ndimage.filters
import scipy.ndimage.fourier
import scipy.signal

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import Image
from nion.ui import Geometry


def column(data_and_metadata, start, stop):
    def calculate_data():
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        return numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)[0]

    return DataAndMetadata.DataAndMetadata(calculate_data,
                                           data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())

def row(data_and_metadata, start, stop):
    def calculate_data():
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        return numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)[1]

    return DataAndMetadata.DataAndMetadata(calculate_data,
                                           data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())

def radius(data_and_metadata, normalize=True):
    def calculate_data():
        start_0 = -1 if normalize else -data_shape(data_and_metadata)[0] * 0.5
        stop_0 = -start_0
        start_1 = -1 if normalize else -data_shape(data_and_metadata)[1] * 0.5
        stop_1 = -start_1
        icol, irow = numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)
        return numpy.sqrt(icol * icol + irow * irow)

    return DataAndMetadata.DataAndMetadata(calculate_data,
                                           data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())

def full(shape, fill_value, dtype=None):
    """Generate a constant valued image with the given shape.

    full(4, shape(4, 5))
    full(0, data_shape(b))
    """
    dtype = dtype if dtype else numpy.float64

    def calculate_data():
        return numpy.full(shape, DataAndMetadata.extract_data(fill_value), dtype)

    return DataAndMetadata.DataAndMetadata(calculate_data, (shape, dtype))

def arange(start, stop=None, step=None):
    if stop is None:
        start = 0
        stop = start
    if step is None:
        step = 1
    def calculate_data():
        return numpy.linspace(int(start), int(stop), int(step))
    return DataAndMetadata.DataAndMetadata(calculate_data, ((math.ceil((stop - start) / step),), numpy.int))

def linspace(start, stop, num, endpoint=True):
    def calculate_data():
        return numpy.linspace(start, stop, num, endpoint)
    return DataAndMetadata.DataAndMetadata(calculate_data, ((num, ), numpy.float))

def logspace(start, stop, num, endpoint=True, base=10.0):
    def calculate_data():
        return numpy.logspace(start, stop, num, endpoint, base)
    return DataAndMetadata.DataAndMetadata(calculate_data, ((num, ), numpy.float))

def apply_dist(data_and_metadata, mean, stddev, dist, fn):
    def calculate_data():
        return getattr(dist(loc=mean, scale=stddev), fn)(data_and_metadata.data)
    return DataAndMetadata.DataAndMetadata(calculate_data, (data_and_metadata.data_shape, numpy.float))

def take_item(data, key):
    return data[key]

def data_shape(data_and_metadata):
    return data_and_metadata.data_shape

def astype(data, dtype):
    return data.astype(dtype)

dtype_map = {int: "int", float: "float", complex: "complex", numpy.int16: "int16", numpy.int32: "int32",
    numpy.int64: "int64", numpy.uint8: "uint8", numpy.uint16: "uint16", numpy.uint32: "uint32", numpy.uint64: "uint64",
    numpy.float32: "float32", numpy.float64: "float64", numpy.complex64: "complex64", numpy.complex128: "complex128"}

dtype_inverse_map = {dtype_map[k]: k for k in dtype_map}

def str_to_dtype(str: str) -> numpy.dtype:
    return dtype_inverse_map.get(str, float)

def dtype_to_str(dtype: numpy.dtype) -> str:
    return dtype_map.get(dtype, "float")


def function_fft(data_and_metadata):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if data is None or not Image.is_data_valid(data):
            return None
        # scaling: numpy.sqrt(numpy.mean(numpy.absolute(data_copy)**2)) == numpy.sqrt(numpy.mean(numpy.absolute(data_copy_fft)**2))
        # see https://gist.github.com/endolith/1257010
        if Image.is_data_1d(data):
            scaling = 1.0 / numpy.sqrt(data_shape[0])
            return scipy.fftpack.fftshift(scipy.fftpack.fft(data) * scaling)
        elif Image.is_data_2d(data):
            data_copy = data.copy()  # let other threads use data while we're processing
            scaling = 1.0 / numpy.sqrt(data_shape[1] * data_shape[0])
            return scipy.fftpack.fftshift(scipy.fftpack.fft2(data_copy) * scaling)
        else:
            raise NotImplementedError()

    src_dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or src_dimensional_calibrations is None:
        return None

    assert len(src_dimensional_calibrations) == len(
        Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype))

    data_shape_and_dtype = data_shape, numpy.dtype(numpy.complex128)

    dimensional_calibrations = [Calibration.Calibration(-0.5 / dimensional_calibration.scale, 1.0 / (dimensional_calibration.scale * data_shape_n),
                                                        "1/" + dimensional_calibration.units) for
        dimensional_calibration, data_shape_n in zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype, Calibration.Calibration(),
                                           dimensional_calibrations, dict(), datetime.datetime.utcnow())


def function_ifft(data_and_metadata):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if data is None or not Image.is_data_valid(data):
            return None
        # scaling: numpy.sqrt(numpy.mean(numpy.absolute(data_copy)**2)) == numpy.sqrt(numpy.mean(numpy.absolute(data_copy_fft)**2))
        # see https://gist.github.com/endolith/1257010
        if Image.is_data_1d(data):
            scaling = numpy.sqrt(data_shape[0])
            return scipy.fftpack.fftshift(scipy.fftpack.ifft(data) * scaling)
        elif Image.is_data_2d(data):
            data_copy = data.copy()  # let other threads use data while we're processing
            scaling = numpy.sqrt(data_shape[1] * data_shape[0])
            return scipy.fftpack.ifft2(scipy.fftpack.ifftshift(data_copy) * scaling)
        else:
            raise NotImplementedError()

    src_dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or src_dimensional_calibrations is None:
        return None

    assert len(src_dimensional_calibrations) == len(
        Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype))

    data_shape_and_dtype = data_shape, data_dtype

    dimensional_calibrations = [Calibration.Calibration(0.0, 1.0 / (dimensional_calibration.scale * data_shape_n),
                                                        "1/" + dimensional_calibration.units) for
        dimensional_calibration, data_shape_n in zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype, Calibration.Calibration(),
                                           dimensional_calibrations, dict(), datetime.datetime.utcnow())


def function_autocorrelate(data_and_metadata):
    def calculate_data():
        data = data_and_metadata.data
        if data is None or not Image.is_data_valid(data):
            return None
        if Image.is_data_2d(data):
            data_copy = data.copy()  # let other threads use data while we're processing
            data_std = data_copy.std(dtype=numpy.float64)
            if data_std != 0.0:
                data_norm = (data_copy - data_copy.mean(dtype=numpy.float64)) / data_std
            else:
                data_norm = data_copy
            scaling = 1.0 / (data_norm.shape[0] * data_norm.shape[1])
            data_norm = numpy.fft.rfft2(data_norm)
            return numpy.fft.fftshift(numpy.fft.irfft2(data_norm * numpy.conj(data_norm))) * scaling
            # this gives different results. why? because for some reason scipy pads out to 1023 and does calculation.
            # see https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
            # return scipy.signal.fftconvolve(data_copy, numpy.conj(data_copy), mode='same')
        return None

    if data_and_metadata is None:
        return None

    dimensional_calibrations = [Calibration.Calibration() for _ in data_and_metadata.data_shape]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           Calibration.Calibration(), dimensional_calibrations, dict(),
                                           datetime.datetime.utcnow())


def function_crosscorrelate(*args):
    if len(args) != 2:
        return None

    data_and_metadata1, data_and_metadata2 = args[0], args[1]

    def calculate_data():
        data1 = data_and_metadata1.data
        data2 = data_and_metadata2.data
        if data1 is None or data2 is None:
            return None
        if Image.is_data_2d(data1) and Image.is_data_2d(data2):
            data_std1 = data1.std(dtype=numpy.float64)
            if data_std1 != 0.0:
                norm1 = (data1 - data1.mean(dtype=numpy.float64)) / data_std1
            else:
                norm1 = data1
            data_std2 = data2.std(dtype=numpy.float64)
            if data_std2 != 0.0:
                norm2 = (data2 - data2.mean(dtype=numpy.float64)) / data_std2
            else:
                norm2 = data2
            scaling = 1.0 / (norm1.shape[0] * norm1.shape[1])
            return numpy.fft.fftshift(numpy.fft.irfft2(numpy.fft.rfft2(norm1) * numpy.conj(numpy.fft.rfft2(norm2)))) * scaling
            # this gives different results. why? because for some reason scipy pads out to 1023 and does calculation.
            # see https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
            # return scipy.signal.fftconvolve(data1.copy(), numpy.conj(data2.copy()), mode='same')
        return None

    if data_and_metadata1 is None or data_and_metadata2 is None:
        return None

    dimensional_calibrations = [Calibration.Calibration() for _ in data_and_metadata1.data_shape]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata1.data_shape_and_dtype,
                                           Calibration.Calibration(), dimensional_calibrations, dict(),
                                           datetime.datetime.utcnow())


def function_sobel(data_and_metadata):
    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.sobel(data[..., 0])
            rgb[..., 1] = scipy.ndimage.sobel(data[..., 1])
            rgb[..., 2] = scipy.ndimage.sobel(data[..., 2])
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.sobel(data[..., 0])
            rgba[..., 1] = scipy.ndimage.sobel(data[..., 1])
            rgba[..., 2] = scipy.ndimage.sobel(data[..., 2])
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.sobel(data)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations, data_and_metadata.metadata,
                                           datetime.datetime.utcnow())


def function_laplace(data_and_metadata):
    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.laplace(data[..., 0])
            rgb[..., 1] = scipy.ndimage.laplace(data[..., 1])
            rgb[..., 2] = scipy.ndimage.laplace(data[..., 2])
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.laplace(data[..., 0])
            rgba[..., 1] = scipy.ndimage.laplace(data[..., 1])
            rgba[..., 2] = scipy.ndimage.laplace(data[..., 2])
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.laplace(data)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations, data_and_metadata.metadata,
                                           datetime.datetime.utcnow())


def function_gaussian_blur(data_and_metadata, sigma):
    sigma = float(sigma)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        return scipy.ndimage.gaussian_filter(data, sigma=sigma)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations, data_and_metadata.metadata,
                                           datetime.datetime.utcnow())


def function_median_filter(data_and_metadata, size):
    size = max(min(int(size), 999), 1)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.median_filter(data[..., 0], size=size)
            rgb[..., 1] = scipy.ndimage.median_filter(data[..., 1], size=size)
            rgb[..., 2] = scipy.ndimage.median_filter(data[..., 2], size=size)
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.median_filter(data[..., 0], size=size)
            rgba[..., 1] = scipy.ndimage.median_filter(data[..., 1], size=size)
            rgba[..., 2] = scipy.ndimage.median_filter(data[..., 2], size=size)
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.median_filter(data, size=size)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations, data_and_metadata.metadata,
                                           datetime.datetime.utcnow())


def function_uniform_filter(data_and_metadata, size):
    size = max(min(int(size), 999), 1)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.uniform_filter(data[..., 0], size=size)
            rgb[..., 1] = scipy.ndimage.uniform_filter(data[..., 1], size=size)
            rgb[..., 2] = scipy.ndimage.uniform_filter(data[..., 2], size=size)
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.uniform_filter(data[..., 0], size=size)
            rgba[..., 1] = scipy.ndimage.uniform_filter(data[..., 1], size=size)
            rgba[..., 2] = scipy.ndimage.uniform_filter(data[..., 2], size=size)
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.uniform_filter(data, size=size)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations, data_and_metadata.metadata,
                                           datetime.datetime.utcnow())


def function_transpose_flip(data_and_metadata, transpose=False, flip_v=False, flip_h=False):
    def calculate_data():
        data = data_and_metadata.data
        data_id = id(data)
        if not Image.is_data_valid(data):
            return None
        if transpose:
            if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
                data = numpy.transpose(data, [1, 0, 2])
            else:
                data = numpy.transpose(data, [1, 0])
        if flip_h:
            data = numpy.fliplr(data)
        if flip_v:
            data = numpy.flipud(data)
        if id(data) == data_id:  # ensure real data, not a view
            data = data.copy()
        return data

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype):
        return None

    if transpose:
        dimensional_calibrations = list(reversed(data_and_metadata.dimensional_calibrations))
    else:
        dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if transpose:
        if Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
            data_shape = list(reversed(data_shape[0:2])) + [data_shape[-1], ]
        else:
            data_shape = list(reversed(data_shape))

    return DataAndMetadata.DataAndMetadata(calculate_data, (data_shape, data_dtype),
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_invert(data_and_metadata):
    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_data_rgba(data):
                inverted = 255 - data[:]
                inverted[...,3] = data[...,3]
                return inverted
            else:
                return 255 - data[:]
        else:
            return -data[:]

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype):
        return None

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    return DataAndMetadata.DataAndMetadata(calculate_data, (data_shape, data_dtype),
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_crop(data_and_metadata, bounds):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        data_shape = data_and_metadata.data_shape
        bounds_int = ((int(data_shape[0] * bounds[0][0]), int(data_shape[1] * bounds[0][1])),
            (int(data_shape[0] * bounds[1][0]), int(data_shape[1] * bounds[1][1])))
        return data[bounds_int[0][0]:bounds_int[0][0] + bounds_int[1][0],
            bounds_int[0][1]:bounds_int[0][1] + bounds_int[1][1]].copy()

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    bounds_int = ((int(data_shape[0] * bounds[0][0]), int(data_shape[1] * bounds[0][1])),
        (int(data_shape[0] * bounds[1][0]), int(data_shape[1] * bounds[1][1])))

    if Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
        data_shape_and_dtype = bounds_int[1] + (data_shape[-1], ), data_dtype
    else:
        data_shape_and_dtype = bounds_int[1], data_dtype

    cropped_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + data_shape[index] * bounds[0][index] * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, cropped_dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_crop_interval(data_and_metadata, interval):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        data_shape = data_and_metadata.data_shape
        interval_int = int(data_shape[0] * interval[0]), int(data_shape[0] * interval[1])
        return data[interval_int[0]:interval_int[1]].copy()

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    interval_int = int(data_shape[0] * interval[0]), int(data_shape[0] * interval[1])

    if Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
        data_shape_and_dtype = (interval_int[1] - interval_int[0], ) + (data_shape[-1], ), data_dtype
    else:
        data_shape_and_dtype = (interval_int[1] - interval_int[0], ), data_dtype

    cropped_dimensional_calibrations = list()
    dimensional_calibration = dimensional_calibrations[0]
    cropped_calibration = Calibration.Calibration(
        dimensional_calibration.offset + data_shape[0] * interval_int[0] * dimensional_calibration.scale,
        dimensional_calibration.scale, dimensional_calibration.units)
    cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, cropped_dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_slice_sum(data_and_metadata, slice_center, slice_width):
    slice_center = int(slice_center)
    slice_width = int(slice_width)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        shape = data.shape
        slice_start = int(slice_center - slice_width * 0.5 + 0.5)
        slice_start = max(slice_start, 0)
        slice_end = slice_start + slice_width
        slice_end = min(shape[0], slice_end)
        return numpy.sum(data[slice_start:slice_end,:], 0)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    data_shape_and_dtype = data_shape[1:], data_dtype

    dimensional_calibrations = dimensional_calibrations[1:]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_pick(data_and_metadata, position):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        data_shape = data_and_metadata.data_shape
        position_f = Geometry.FloatPoint.make(position)
        position_i = Geometry.IntPoint(y=position_f.y * data_shape[1], x=position_f.x * data_shape[2])
        if position_i.y >= 0 and position_i.y < data_shape[1] and position_i.x >= 0 and position_i.x < data_shape[2]:
            return data[:, position_i[0], position_i[1]].copy()
        else:
            return numpy.zeros((data_shape[:-2], ), dtype=data.dtype)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    data_shape_and_dtype = data_shape[:-2], data_dtype

    dimensional_calibrations = dimensional_calibrations[0:-2]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_concatenate(data_and_metadata_list, axis=0):
    """Concatenate multiple data_and_metadatas.

    concatenate((a, b, c), 1)

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    partial_shape = data_and_metadata_list[0].data_shape

    def calculate_data():
        if any([data_and_metadata.data is None for data_and_metadata in data_and_metadata_list]):
            return None
        if all([data_and_metadata.data_shape[1:] == partial_shape[1:] for data_and_metadata in data_and_metadata_list]):
            data_list = list(data_and_metadata.data for data_and_metadata in data_and_metadata_list)
            return numpy.concatenate(data_list, axis)
        return None

    if any([data_and_metadata.data is None for data_and_metadata in data_and_metadata_list]):
        return None

    if any([data_and_metadata.data_shape != partial_shape[1:] is None for data_and_metadata in data_and_metadata_list]):
        return None

    dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(data_and_metadata_list[0].dimensional_calibrations):
        if index != axis:
            dimensional_calibrations.append(Calibration.Calibration())
        else:
            dimensional_calibrations.append(dimensional_calibration)

    new_shape = [sum([data_and_metadata.data_shape[0] for data_and_metadata in data_and_metadata_list]), ] + list(partial_shape[1:])

    intensity_calibration = data_and_metadata_list[0].intensity_calibration

    return DataAndMetadata.DataAndMetadata(calculate_data, (new_shape, data_and_metadata_list[0].data_dtype), intensity_calibration, dimensional_calibrations, dict(),
                                           datetime.datetime.utcnow())


def function_hstack(data_and_metadata_list):
    """Stack multiple data_and_metadatas along axis 1.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    partial_shape = data_and_metadata_list[0].data_shape

    if len(partial_shape) >= 2:
        return function_concatenate(data_and_metadata_list, 1)
    else:
        return function_concatenate(data_and_metadata_list, 0)


def function_vstack(data_and_metadata_list):
    """Stack multiple data_and_metadatas along axis 0.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    partial_shape = data_and_metadata_list[0].data_shape

    if len(partial_shape) >= 2:
        return function_concatenate(data_and_metadata_list, 0)

    def calculate_data():
        if any([data_and_metadata.data is None for data_and_metadata in data_and_metadata_list]):
            return None
        if all([data_and_metadata.data_shape[0] == partial_shape[0] for data_and_metadata in data_and_metadata_list]):
            data_list = list(data_and_metadata.data for data_and_metadata in data_and_metadata_list)
            return numpy.vstack(data_list)
        return None

    if any([data_and_metadata.data is None for data_and_metadata in data_and_metadata_list]):
        return None

    if any([data_and_metadata.data_shape[0] != partial_shape[0] is None for data_and_metadata in data_and_metadata_list]):
        return None

    dimensional_calibrations = list()
    dimensional_calibrations.append(Calibration.Calibration())
    dimensional_calibrations.append(data_and_metadata_list[0].dimensional_calibrations[0])

    new_shape = [len(data_and_metadata_list), ] + list(partial_shape)

    intensity_calibration = data_and_metadata_list[0].intensity_calibration

    return DataAndMetadata.DataAndMetadata(calculate_data, (new_shape, data_and_metadata_list[0].data_dtype), intensity_calibration, dimensional_calibrations, dict(),
                                           datetime.datetime.utcnow())


def function_sum(data_and_metadata, axis=None):
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
                rgb_image = numpy.empty(data.shape[1:], numpy.uint8)
                rgb_image[:,0] = numpy.average(data[...,0], 0)
                rgb_image[:,1] = numpy.average(data[...,1], 0)
                rgb_image[:,2] = numpy.average(data[...,2], 0)
                return rgb_image
            else:
                rgba_image = numpy.empty(data.shape[1:], numpy.uint8)
                rgba_image[:,0] = numpy.average(data[...,0], 0)
                rgba_image[:,1] = numpy.average(data[...,1], 0)
                rgba_image[:,2] = numpy.average(data[...,2], 0)
                rgba_image[:,3] = numpy.average(data[...,3], 0)
                return rgba_image
        else:
            return numpy.sum(data, axis)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    data_shape_and_dtype = data_shape[1:], data_dtype

    dimensional_calibrations = dimensional_calibrations[1:]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_reshape(data_and_metadata, shape):
    """Reshape a data and metadata to shape.

    reshape(a, shape(4, 5))
    reshape(a, data_shape(b))

    Handles special cases when going to one extra dimension and when going to one fewer
    dimension -- namely to keep the calibrations intact.

    When increasing dimension, a -1 can be passed for the new dimension and this function
    will calculate the missing value.
    """
    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        return numpy.reshape(data, shape)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    total_old_pixels = 1
    for dimension in data_shape:
        total_old_pixels *= dimension
    total_new_pixels = 1
    for dimension in shape:
        total_new_pixels *= dimension if dimension > 0 else 1
    new_dimensional_calibrations = list()
    new_dimensions = list()
    if len(data_shape) + 1 == len(shape) and -1 in shape:
        # special case going to one more dimension
        index = 0
        for dimension in shape:
            if dimension == -1:
                new_dimensional_calibrations.append(Calibration.Calibration())
                new_dimensions.append(total_old_pixels / total_new_pixels)
            else:
                new_dimensional_calibrations.append(dimensional_calibrations[index])
                new_dimensions.append(dimension)
                index += 1
    elif len(data_shape) - 1 == len(shape) and 1 in data_shape:
        # special case going to one fewer dimension
        for dimension, dimensional_calibration in zip(data_shape, dimensional_calibrations):
            if dimension == 1:
                continue
            else:
                new_dimensional_calibrations.append(dimensional_calibration)
                new_dimensions.append(dimension)
    else:
        for _ in range(len(shape)):
            new_dimensional_calibrations.append(Calibration.Calibration())

    data_shape_and_dtype = new_dimensions, data_dtype

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, new_dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_resample_2d(data_and_metadata, shape):
    height = int(shape[0])
    width = int(shape[1])

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if not Image.is_data_2d(data):
            return None
        if data.shape[0] == height and data.shape[1] == width:
            return data.copy()
        return Image.scaled(data, (height, width))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    if not Image.is_shape_and_dtype_2d(data_shape, data_dtype):
        return None

    if Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
        data_shape_and_dtype = (height, width, data_shape[-1]), data_dtype
    else:
        data_shape_and_dtype = (height, width), data_dtype

    dimensions = height, width
    resampled_dimensional_calibrations = [Calibration.Calibration(dimensional_calibrations[i].offset, dimensional_calibrations[i].scale * data_shape[i] / dimensions[i], dimensional_calibrations[i].units) for i in range(len(dimensional_calibrations))]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, resampled_dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_histogram(data_and_metadata, bins):
    bins = int(bins)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        histogram_data = numpy.histogram(data, bins=bins)
        return histogram_data[0].astype(numpy.int)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    data_shape_and_dtype = (bins, ), numpy.dtype(numpy.int)

    dimensional_calibrations = [Calibration.Calibration()]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())


def function_line_profile(data_and_metadata, vector, integration_width):
    integration_width = int(integration_width)
    assert integration_width > 0  # leave this here for test_evaluation_error_recovers_gracefully

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    # calculate grid of coordinates. returns n coordinate arrays for each row.
    # start and end are in data coordinates.
    # n is a positive integer, not zero
    def get_coordinates(start, end, n):
        assert n > 0 and int(n) == n
        # n=1 => 0
        # n=2 => -0.5, 0.5
        # n=3 => -1, 0, 1
        # n=4 => -1.5, -0.5, 0.5, 1.5
        length = math.sqrt(math.pow(end[0] - start[0], 2) + math.pow(end[1] - start[1], 2))
        l = math.floor(length)
        a = numpy.linspace(0, length, l)  # along
        t = numpy.linspace(-(n-1)*0.5, (n-1)*0.5, n)  # transverse
        dy = (end[0] - start[0]) / length
        dx = (end[1] - start[1]) / length
        ix, iy = numpy.meshgrid(a, t)
        yy = start[0] + dy * ix + dx * iy
        xx = start[1] + dx * ix - dy * iy
        return xx, yy

    # xx, yy = __coordinates(None, (4,4), (8,4), 3)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_data_rgb_type(data):
            data = Image.convert_to_grayscale(data, numpy.double)
        start, end = vector
        shape = data.shape
        actual_integration_width = min(max(shape[0], shape[1]), integration_width)  # limit integration width to sensible value
        start_data = (int(shape[0]*start[0]), int(shape[1]*start[1]))
        end_data = (int(shape[0]*end[0]), int(shape[1]*end[1]))
        length = math.sqrt(math.pow(end_data[1] - start_data[1], 2) + math.pow(end_data[0] - start_data[0], 2))
        if length > 1.0:
            spline_order_lookup = { "nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3 }
            method = "nearest"
            spline_order = spline_order_lookup[method]
            xx, yy = get_coordinates(start_data, end_data, actual_integration_width)
            samples = scipy.ndimage.map_coordinates(data, (yy, xx), order=spline_order)
            if len(samples.shape) > 1:
                return numpy.sum(samples, 0) / actual_integration_width
            else:
                return samples
        return numpy.zeros((1))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    if dimensional_calibrations is None or len(dimensional_calibrations) != 2:
        return None

    start, end = vector
    shape = data_shape
    start_int = (int(shape[0]*start[0]), int(shape[1]*start[1]))
    end_int = (int(shape[0]*end[0]), int(shape[1]*end[1]))
    length = int(math.sqrt((end_int[1] - start_int[1])**2 + (end_int[0] - start_int[0])**2))
    length = max(length, 1)
    data_shape_and_dtype = (length, ), numpy.dtype(numpy.double)

    dimensional_calibrations = [Calibration.Calibration(0.0, dimensional_calibrations[1].scale, dimensional_calibrations[1].units)]

    return DataAndMetadata.DataAndMetadata(calculate_data, data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration, dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())

def function_make_point(y, x):
    return y, x

def function_make_size(height, width):
    return height, width

def function_make_vector(start, end):
    return start, end

def function_make_rectangle_origin_size(origin, size):
    return tuple(Geometry.FloatRect(origin, size))

def function_make_rectangle_center_size(center, size):
    return tuple(Geometry.FloatRect.from_center_and_size(center, size))

def function_make_interval(start, end):
    return start, end

def function_make_shape(*args):
    return tuple(args)

# generic functions

def function_array(array_fn, data_and_metadata, *args, **kwargs):
    def calculate_data():
        return array_fn(data_and_metadata.data, *args, **kwargs)

    return DataAndMetadata.DataAndMetadata(calculate_data,
                                           data_and_metadata.data_shape_and_dtype,
                                           data_and_metadata.intensity_calibration,
                                           data_and_metadata.dimensional_calibrations,
                                           data_and_metadata.metadata, datetime.datetime.utcnow())

def function_scalar(op, data_and_metadata):
    def calculate_value():
        return op(data_and_metadata.data)

    return DataAndMetadata.ScalarAndMetadata(calculate_value, data_and_metadata.intensity_calibration, data_and_metadata.metadata, datetime.datetime.utcnow())
