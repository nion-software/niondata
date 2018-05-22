# standard libraries
import collections
import copy
import functools
import math
import numbers
import operator
import typing

# third party libraries
import numpy
import numpy.fft
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
from nion.data import ImageRegistration
from nion.utils import Geometry


DataRangeType = typing.Tuple[float, float]
NormIntervalType = typing.Tuple[float, float]
NormChannelType = float
NormRectangleType = typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]
NormPointType = typing.Tuple[float, float]
NormSizeType = typing.Tuple[float, float]
NormVectorType = typing.Tuple[NormPointType, NormPointType]


def column(data_and_metadata: DataAndMetadata.DataAndMetadata, start: int, stop: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    def calculate_data():
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        return numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)[0]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def row(data_and_metadata: DataAndMetadata.DataAndMetadata, start: int, stop: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    def calculate_data():
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        return numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)[1]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def radius(data_and_metadata: DataAndMetadata.DataAndMetadata, normalize: bool=True) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    def calculate_data():
        start_0 = -1 if normalize else -data_shape(data_and_metadata)[0] * 0.5
        stop_0 = -start_0
        start_1 = -1 if normalize else -data_shape(data_and_metadata)[1] * 0.5
        stop_1 = -start_1
        icol, irow = numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)
        return numpy.sqrt(icol * icol + irow * irow)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def full(shape: DataAndMetadata.ShapeType, fill_value, dtype: numpy.dtype=None) -> DataAndMetadata.DataAndMetadata:
    """Generate a constant valued image with the given shape.

    full(4, shape(4, 5))
    full(0, data_shape(b))
    """
    dtype = dtype if dtype else numpy.float64

    return DataAndMetadata.new_data_and_metadata(numpy.full(shape, DataAndMetadata.extract_data(fill_value), dtype))


def arange(start: int, stop: int=None, step: int=None) -> DataAndMetadata.DataAndMetadata:
    if stop is None:
        start = 0
        stop = start
    if step is None:
        step = 1
    return DataAndMetadata.new_data_and_metadata(numpy.linspace(int(start), int(stop), int(step)))


def linspace(start: float, stop: float, num: int, endpoint: bool=True) -> DataAndMetadata.DataAndMetadata:
    return DataAndMetadata.new_data_and_metadata(numpy.linspace(start, stop, num, endpoint))


def logspace(start: float, stop: float, num: int, endpoint: bool=True, base: float=10.0) -> DataAndMetadata.DataAndMetadata:
    return DataAndMetadata.new_data_and_metadata(numpy.logspace(start, stop, num, endpoint, base))


def apply_dist(data_and_metadata: DataAndMetadata.DataAndMetadata, mean: float, stddev: float, dist, fn) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    return DataAndMetadata.new_data_and_metadata(getattr(dist(loc=mean, scale=stddev), fn)(data_and_metadata.data))


def take_item(data, key):
    return data[key]


def data_shape(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.ShapeType:
    return data_and_metadata.data_shape


def astype(data: numpy.ndarray, dtype: numpy.dtype) -> numpy.ndarray:
    return data.astype(dtype)


dtype_map = {int: "int", float: "float", complex: "complex", numpy.int16: "int16", numpy.int32: "int32",
    numpy.int64: "int64", numpy.uint8: "uint8", numpy.uint16: "uint16", numpy.uint32: "uint32", numpy.uint64: "uint64",
    numpy.float32: "float32", numpy.float64: "float64", numpy.complex64: "complex64", numpy.complex128: "complex128"}

dtype_inverse_map = {dtype_map[k]: k for k in dtype_map}


def str_to_dtype(str: str) -> numpy.dtype:
    return dtype_inverse_map.get(str, float)

def dtype_to_str(dtype: numpy.dtype) -> str:
    return dtype_map.get(dtype, "float")


def function_fft(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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
            return scipy.fftpack.fftshift(numpy.multiply(scipy.fftpack.fft(data), scaling))
        elif Image.is_data_2d(data):
            if Image.is_data_rgb_type(data):
                if Image.is_data_rgb(data):
                    data_copy = numpy.sum(data[..., :] * (0.2126, 0.7152, 0.0722), 2)
                else:
                    data_copy = numpy.sum(data[..., :] * (0.2126, 0.7152, 0.0722, 0.0), 2)
            else:
                data_copy = data.copy()  # let other threads use data while we're processing
            scaling = 1.0 / numpy.sqrt(data_shape[1] * data_shape[0])
            # note: the numpy.fft.fft2 is faster than scipy.fftpack.fft2, probably either because
            # our conda distribution compiles numpy for multiprocessing, the numpy version releases
            # the GIL, or both.
            return scipy.fftpack.fftshift(numpy.multiply(numpy.fft.fft2(data_copy), scaling))
        else:
            raise NotImplementedError()

    src_dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or src_dimensional_calibrations is None:
        return None

    assert len(src_dimensional_calibrations) == len(
        Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype))

    dimensional_calibrations = [Calibration.Calibration(-0.5 / dimensional_calibration.scale, 1.0 / (dimensional_calibration.scale * data_shape_n),
                                                        "1/" + dimensional_calibration.units) for
        dimensional_calibration, data_shape_n in zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=dimensional_calibrations)


def function_ifft(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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
            return scipy.fftpack.ifft(scipy.fftpack.ifftshift(data) * scaling)
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

    def remove_one_slash(s):
        if s.startswith("1/"):
            return s[2:]
        else:
            return "1/" + s

    dimensional_calibrations = [Calibration.Calibration(0.0, 1.0 / (dimensional_calibration.scale * data_shape_n),
                                                        remove_one_slash(dimensional_calibration.units)) for
        dimensional_calibration, data_shape_n in zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=dimensional_calibrations)


def function_autocorrelate(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_crosscorrelate(*args) -> DataAndMetadata.DataAndMetadata:
    if len(args) != 2:
        return None

    data_and_metadata1, data_and_metadata2 = args[0], args[1]

    data_and_metadata1 = DataAndMetadata.promote_ndarray(data_and_metadata1)
    data_and_metadata2 = DataAndMetadata.promote_ndarray(data_and_metadata2)

    shape = DataAndMetadata.determine_shape(data_and_metadata1, data_and_metadata2)

    data_and_metadata1 = DataAndMetadata.promote_constant(data_and_metadata1, shape)
    data_and_metadata2 = DataAndMetadata.promote_constant(data_and_metadata2, shape)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=data_and_metadata1.dimensional_calibrations)


def function_register(xdata1: DataAndMetadata.DataAndMetadata, xdata2: DataAndMetadata.DataAndMetadata, upsample_factor: int, subtract_means: bool) -> typing.Tuple[float, ...]:
    xdata1 = DataAndMetadata.promote_ndarray(xdata1)
    xdata2 = DataAndMetadata.promote_ndarray(xdata2)

    # FUTURE: use scikit.image register_translation
    data1 = xdata1.data
    data2 = xdata2.data
    if subtract_means:
        data1 = data1 - numpy.average(data1)
        data2 = data2 - numpy.average(data2)
    data_shape = len(data1.shape)
    if len(data1.shape) == 1:
        data1 = data1[..., numpy.newaxis]
    if len(data2.shape) == 1:
        data2 = data2[..., numpy.newaxis]
    return tuple(ImageRegistration.dftregistration(data1, data2, upsample_factor)[0:data_shape])


def function_shift(src: DataAndMetadata.DataAndMetadata, shift: typing.Tuple[float, ...]) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    src_data = numpy.fft.fftn(src.data)
    if len(src_data.shape) == 1:
        src_data = src_data[..., numpy.newaxis]
        shift = tuple(shift) + (1,)
    # NOTE: fourier_shift assumes non-fft-shifted data.
    shifted = numpy.fft.ifftn(scipy.ndimage.fourier_shift(src_data, shift)).real
    return DataAndMetadata.new_data_and_metadata(numpy.squeeze(shifted))


def function_align(src: DataAndMetadata.DataAndMetadata, target: DataAndMetadata.DataAndMetadata, upsample_factor: int) -> DataAndMetadata.DataAndMetadata:
    """Aligns target to src and returns align target, using Fourier space."""
    src = DataAndMetadata.promote_ndarray(src)
    target = DataAndMetadata.promote_ndarray(target)
    return function_shift(target, function_register(src, target, upsample_factor, True))


def function_sequence_register_translation(src: DataAndMetadata.DataAndMetadata, upsample_factor: int, subtract_means: bool) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    if not src.is_sequence:
        return None
    c = src.sequence_dimension_shape[0]
    dim = src.data_shape[1:]
    if len(dim) < 1 or len(dim) > 2:
        return None
    result = numpy.empty((c - 1, len(dim)))
    previous_data = None
    for i in range(c):
        if previous_data is None:
            previous_data = src.data[i, ...]
        else:
            current_data = src.data[i, ...]
            result[i - 1, ...] = function_register(previous_data, current_data, upsample_factor, subtract_means)
            previous_data = current_data
    intensity_calibration = src.dimensional_calibrations[1]  # not the sequence dimension
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration)


def function_sequence_align(src: DataAndMetadata.DataAndMetadata, upsample_factor: int) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    translations = function_sequence_register_translation(src, upsample_factor, True)
    if not translations:
        return None
    result_data = numpy.copy(src.data)
    for i in range(translations.data_shape[0]):
        current_xdata = DataAndMetadata.new_data_and_metadata(numpy.copy(result_data[i + 1]))
        cumulative_shift = numpy.sum(translations.data[0:i + 1], axis=0)
        result_data[i + 1, ...] = function_shift(current_xdata, cumulative_shift).data
    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=src.intensity_calibration, dimensional_calibrations=src.dimensional_calibrations, data_descriptor=src.data_descriptor)


def function_sequence_integrate(src: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    if not src.is_sequence:
        return None
    dim = src.data_shape[1:]
    if len(dim) < 1 or len(dim) > 2:
        return None
    result = numpy.sum(src.data, axis=0)
    intensity_calibration = src.intensity_calibration
    dimensional_calibrations = src.dimensional_calibrations[1:]
    data_descriptor = DataAndMetadata.DataDescriptor(False, src.data_descriptor.collection_dimension_count, src.data_descriptor.datum_dimension_count)
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)

def function_sequence_trim(src: DataAndMetadata.DataAndMetadata, trim_start: int, trim_end: int) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    if not src.is_sequence:
        return None
    c = src.sequence_dimension_shape[0]
    dim = src.data_shape[1:]
    if len(dim) < 1 or len(dim) > 2:
        return None
    cs = max(0, int(trim_start))
    ce = min(c, max(cs + 1, int(trim_end)))
    return src[cs:ce]


def function_sequence_insert(src1: DataAndMetadata.DataAndMetadata, src2: DataAndMetadata.DataAndMetadata, position: int) -> DataAndMetadata.DataAndMetadata:
    src1 = DataAndMetadata.promote_ndarray(src1)
    src2 = DataAndMetadata.promote_ndarray(src2)
    if not src1.is_sequence or not src2.is_sequence:
        return None
    if src1.data_shape[1:] != src2.data_shape[1:]:
        return None
    c = src1.sequence_dimension_shape[0]
    dim = src1.data_shape[1:]
    if len(dim) < 1 or len(dim) > 2:
        return None
    channel = max(0, min(c, int(position)))
    result = numpy.vstack(src1.data[:channel], src2.data, src1.data[channel:])
    intensity_calibration = src1.intensity_calibration
    dimensional_calibrations = src1.dimensional_calibrations
    data_descriptor = src1.data_descriptor
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_sequence_concatenate(src1: DataAndMetadata.DataAndMetadata, src2: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    src1 = DataAndMetadata.promote_ndarray(src1)
    src2 = DataAndMetadata.promote_ndarray(src2)
    return function_sequence_insert(src1, src2, src1.data_shape[0])


def function_sequence_extract(src: DataAndMetadata.DataAndMetadata, position: int) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src)
    if not src.is_sequence:
        return None
    c = src.sequence_dimension_shape[0]
    dim = src.data_shape[1:]
    if len(dim) < 1 or len(dim) > 2:
        return None
    channel = max(0, min(c, int(position)))
    return src[channel]


def function_fourier_mask(data_and_metadata: DataAndMetadata.DataAndMetadata, mask_data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    mask_data_and_metadata = DataAndMetadata.promote_ndarray(mask_data_and_metadata)

    shape = DataAndMetadata.determine_shape(data_and_metadata, mask_data_and_metadata)

    data_and_metadata = DataAndMetadata.promote_constant(data_and_metadata, shape)
    mask_data_and_metadata = DataAndMetadata.promote_constant(mask_data_and_metadata, shape)

    def calculate_data():
        data = data_and_metadata.data
        mask_data = mask_data_and_metadata.data
        if data is None or mask_data is None:
            return None
        if Image.is_data_2d(data) and Image.is_data_2d(mask_data):
            try:
                y_half = data.shape[0] // 2
                y_half_p1 = y_half + 1
                y_half_m1 = y_half - 1
                y_low = 0 if data.shape[0] % 2 == 0 else None
                x_half = data.shape[1] // 2
                x_half_p1 = x_half + 1
                x_half_m1 = x_half - 1
                x_low = 0 if data.shape[1] % 2 == 0 else None
                fourier_mask_data = numpy.empty_like(mask_data)
                fourier_mask_data[y_half_p1:, x_half_p1:] = mask_data[y_half_p1:, x_half_p1:]
                fourier_mask_data[y_half_p1:, x_half_m1:x_low:-1] = mask_data[y_half_p1:, x_half_m1:x_low:-1]
                fourier_mask_data[y_half_m1:y_low:-1, x_half_m1:x_low:-1] = mask_data[y_half_p1:, x_half_p1:]
                fourier_mask_data[y_half_m1:y_low:-1, x_half_p1:] = mask_data[y_half_p1:, x_half_m1:x_low:-1]
                fourier_mask_data[0, :] = mask_data[0, :]
                fourier_mask_data[:, 0] = mask_data[:, 0]
                fourier_mask_data[y_half, :] = mask_data[y_half, :]
                fourier_mask_data[:, x_half] = mask_data[:, x_half]
                return data * fourier_mask_data
            except Exception as e:
                print(e)
                raise
        return None

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_sobel(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_laplace(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_gaussian_blur(data_and_metadata: DataAndMetadata.DataAndMetadata, sigma: float) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    sigma = float(sigma)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        return scipy.ndimage.gaussian_filter(data, sigma=sigma)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_median_filter(data_and_metadata: DataAndMetadata.DataAndMetadata, size: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_uniform_filter(data_and_metadata: DataAndMetadata.DataAndMetadata, size: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_transpose_flip(data_and_metadata: DataAndMetadata.DataAndMetadata, transpose: bool=False, flip_v: bool=False, flip_h: bool=False) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    def calculate_data():
        data = data_and_metadata.data
        data_id = id(data)
        if not Image.is_data_valid(data):
            return None
        if transpose:
            if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
                data = numpy.transpose(data, [1, 0, 2])
            elif len(data_and_metadata.data_shape) == 2:
                data = numpy.transpose(data, [1, 0])
        if flip_h and len(data_and_metadata.data_shape) == 2:
            data = numpy.fliplr(data)
        if flip_v and len(data_and_metadata.data_shape) == 2:
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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_invert(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_crop(data_and_metadata: DataAndMetadata.DataAndMetadata, bounds: NormRectangleType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    cropped_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + data_shape[index] * bounds[0][index] * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations)


def function_crop_interval(data_and_metadata: DataAndMetadata.DataAndMetadata, interval: NormIntervalType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    cropped_dimensional_calibrations = list()
    dimensional_calibration = dimensional_calibrations[0]
    cropped_calibration = Calibration.Calibration(
        dimensional_calibration.offset + data_shape[0] * interval_int[0] * dimensional_calibration.scale,
        dimensional_calibration.scale, dimensional_calibration.units)
    cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations)


def function_slice_sum(data_and_metadata: DataAndMetadata.DataAndMetadata, slice_center: int, slice_width: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    signal_index = -1

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
        slice_end = min(shape[signal_index], slice_end)
        return numpy.sum(data[..., slice_start:slice_end], signal_index)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    dimensional_calibrations = dimensional_calibrations[0:signal_index]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_pick(data_and_metadata: DataAndMetadata.DataAndMetadata, position: DataAndMetadata.PositionType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        collection_dimensions = data_and_metadata.dimensional_shape[data_and_metadata.collection_dimension_slice]
        datum_dimensions = data_and_metadata.dimensional_shape[data_and_metadata.datum_dimension_slice]
        assert len(collection_dimensions) == len(position)
        position_i = list()
        for collection_dimension, pos in zip(collection_dimensions, position):
            pos_i = int(pos * collection_dimension)
            if not (0 <= pos_i < collection_dimension):
                return numpy.zeros(datum_dimensions, dtype=data.dtype)
            position_i.append(pos_i)
        return data[tuple(position_i + [...])].copy()

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    if len(position) != data_and_metadata.collection_dimension_count:
        return None

    if data_and_metadata.datum_dimension_count == 0:
        return None

    dimensional_calibrations = dimensional_calibrations[data_and_metadata.datum_dimension_slice]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_concatenate(data_and_metadata_list: typing.Sequence[DataAndMetadata.DataAndMetadata], axis: int=0) -> DataAndMetadata.DataAndMetadata:
    """Concatenate multiple data_and_metadatas.

    concatenate((a, b, c), 1)

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_list]

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

    intensity_calibration = data_and_metadata_list[0].intensity_calibration

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_hstack(data_and_metadata_list: typing.Sequence[DataAndMetadata.DataAndMetadata]) -> DataAndMetadata.DataAndMetadata:
    """Stack multiple data_and_metadatas along axis 1.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_list]

    partial_shape = data_and_metadata_list[0].data_shape

    if len(partial_shape) >= 2:
        return function_concatenate(data_and_metadata_list, 1)
    else:
        return function_concatenate(data_and_metadata_list, 0)


def function_vstack(data_and_metadata_list: typing.Sequence[DataAndMetadata.DataAndMetadata]) -> DataAndMetadata.DataAndMetadata:
    """Stack multiple data_and_metadatas along axis 0.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    if len(data_and_metadata_list) < 1:
        return None

    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_list]

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

    intensity_calibration = data_and_metadata_list[0].intensity_calibration

    data_descriptor = data_and_metadata_list[0].data_descriptor

    data_descriptor = DataAndMetadata.DataDescriptor(data_descriptor.is_sequence, data_descriptor.collection_dimension_count + 1, data_descriptor.datum_dimension_count)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_moveaxis(data_and_metadata: DataAndMetadata.DataAndMetadata, src_axis: int, dst_axis: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data = numpy.moveaxis(data_and_metadata.data, src_axis, dst_axis)

    dimensional_calibrations = copy.deepcopy(data_and_metadata.dimensional_calibrations)

    dimensional_calibrations.insert(dst_axis, dimensional_calibrations.pop(src_axis))

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_sum(data_and_metadata: DataAndMetadata.DataAndMetadata, axis: typing.Union[int, typing.Sequence[int]]=None) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
                rgb_image = numpy.empty(data.shape[1:], numpy.uint8)
                rgb_image[:,0] = numpy.average(data[...,0], axis)
                rgb_image[:,1] = numpy.average(data[...,1], axis)
                rgb_image[:,2] = numpy.average(data[...,2], axis)
                return rgb_image
            else:
                rgba_image = numpy.empty(data.shape[1:], numpy.uint8)
                rgba_image[:,0] = numpy.average(data[...,0], axis)
                rgba_image[:,1] = numpy.average(data[...,1], axis)
                rgba_image[:,2] = numpy.average(data[...,2], axis)
                rgba_image[:,3] = numpy.average(data[...,3], axis)
                return rgba_image
        else:
            return numpy.sum(data, axis)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    new_dimensional_calibrations = list()

    if isinstance(axis, numbers.Integral):
        for index, dimensional_calibration in enumerate(dimensional_calibrations):
            if index != axis:
                new_dimensional_calibrations.append(dimensional_calibration)
    elif isinstance(axis, collections.Sequence):
        axes = tuple(axis)
        for index, dimensional_calibration in enumerate(dimensional_calibrations):
            if not index in axes:
                new_dimensional_calibrations.append(dimensional_calibration)

    dimensional_calibrations = new_dimensional_calibrations

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_sum_region(data_and_metadata: DataAndMetadata.DataAndMetadata, mask_data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    mask_data_and_metadata = DataAndMetadata.promote_ndarray(mask_data_and_metadata)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    assert len(data_and_metadata.dimensional_shape) == 3
    assert len(mask_data_and_metadata.dimensional_shape) == 2

    data = data_and_metadata.data
    mask_data = mask_data_and_metadata.data

    assert data is not None
    assert mask_data is not None

    result_data = numpy.sum(data * mask_data[..., numpy.newaxis], tuple(range(0, len(data_and_metadata.dimensional_shape) - 1)))

    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=[data_and_metadata.dimensional_calibrations[-1]])


def function_average_region(data_and_metadata: DataAndMetadata.DataAndMetadata, mask_data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    mask_data_and_metadata = DataAndMetadata.promote_ndarray(mask_data_and_metadata)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    assert len(data_and_metadata.dimensional_shape) == 3
    assert len(mask_data_and_metadata.dimensional_shape) == 2

    data = data_and_metadata.data
    mask_data = mask_data_and_metadata.data

    assert data is not None
    assert mask_data is not None

    mask_sum = numpy.sum(mask_data)

    if mask_sum == 0:
        mask_sum = 1

    result_data = numpy.sum(data * mask_data[..., numpy.newaxis], tuple(range(0, len(data_and_metadata.dimensional_shape) - 1))) / mask_sum

    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=[data_and_metadata.dimensional_calibrations[-1]])


def function_reshape(data_and_metadata: DataAndMetadata.DataAndMetadata, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
    """Reshape a data and metadata to shape.

    reshape(a, shape(4, 5))
    reshape(a, data_shape(b))

    Handles special cases when going to one extra dimension and when going to one fewer
    dimension -- namely to keep the calibrations intact.

    When increasing dimension, a -1 can be passed for the new dimension and this function
    will calculate the missing value.
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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
    if len(data_shape) + 1 == len(shape) and -1 in shape:
        # special case going to one more dimension
        index = 0
        for dimension in shape:
            if dimension == -1:
                new_dimensional_calibrations.append(Calibration.Calibration())
            else:
                new_dimensional_calibrations.append(dimensional_calibrations[index])
                index += 1
    elif len(data_shape) - 1 == len(shape) and 1 in data_shape:
        # special case going to one fewer dimension
        for dimension, dimensional_calibration in zip(data_shape, dimensional_calibrations):
            if dimension == 1:
                continue
            else:
                new_dimensional_calibrations.append(dimensional_calibration)
    else:
        for _ in range(len(shape)):
            new_dimensional_calibrations.append(Calibration.Calibration())

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=new_dimensional_calibrations)


def function_squeeze(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    """Remove dimensions with lengths of one."""
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data_shape = data_and_metadata.data_shape

    dimensional_calibrations = data_and_metadata.dimensional_calibrations
    is_sequence = data_and_metadata.is_sequence
    collection_dimension_count = data_and_metadata.collection_dimension_count
    datum_dimension_count = data_and_metadata.datum_dimension_count
    new_dimensional_calibrations = list()
    dimensional_index = 0

    # fix the data descriptor and the dimensions
    if is_sequence:
        if data_shape[dimensional_index] <= 1:
            is_sequence = 0
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1
    for collection_dimension_index in range(collection_dimension_count):
        if data_shape[dimensional_index] <= 1:
            collection_dimension_count -= 1
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1
    for datum_dimension_index in range(datum_dimension_count):
        if data_shape[dimensional_index] <= 1:
            datum_dimension_count -= 1
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1

    data = data_and_metadata.data
    if not Image.is_data_valid(data):
        return None
    data = numpy.squeeze(data)

    data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=new_dimensional_calibrations, data_descriptor=data_descriptor)


def function_resize(data_and_metadata: DataAndMetadata.DataAndMetadata, shape: DataAndMetadata.ShapeType, mode: str=None) -> DataAndMetadata.DataAndMetadata:
    """Resize a data and metadata to shape, padding if larger, cropping if smaller.

    resize(a, shape(4, 5))
    resize(a, data_shape(b))

    Shape must have same number of dimensions as original.
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        c = numpy.mean(data)
        data_shape = data_and_metadata.data_shape
        slices = list()
        for data_size, new_size in zip(data_shape, shape):
            if new_size <= data_size:
                left = data_size // 2 - new_size // 2
                slices.append(slice(left, left + new_size))
            else:
                slices.append(slice(None))
        data = data[slices]
        data_shape = data_and_metadata.data_shape
        pads = list()
        for data_size, new_size in zip(data_shape, shape):
            if new_size > data_size:
                left = new_size // 2 - data_size // 2
                pads.append((left, new_size - left - data_size))
            else:
                pads.append((0, 0))
        return numpy.pad(data, pads, 'constant', constant_values=c)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    resized_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        offset = data_shape[index] // 2 - shape[index] // 2
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + offset * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        resized_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=resized_dimensional_calibrations)


def function_rescale(data_and_metadata: DataAndMetadata.DataAndMetadata, data_range: DataRangeType=None) -> DataAndMetadata.DataAndMetadata:
    """Rescale data and update intensity calibration.

    rescale(a, (0.0, 1.0))
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    data_range = data_range if data_range is not None else (0.0, 1.0)

    def calculate_data():
        data = data_and_metadata.data
        if not Image.is_data_valid(data):
            return None
        data_ptp = numpy.ptp(data)
        data_ptp_i = 1.0 / data_ptp if data_ptp != 0.0 else 1.0
        data_min = numpy.amin(data)
        data_span = data_range[1] - data_range[0]
        if data_span == 1.0 and data_range[0] == 0.0:
            return (data - data_min) * data_ptp_i
        else:
            m = data_span * data_ptp_i
            return (data - data_min) * m + data_range[0]

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype):
        return None

    intensity_calibration = Calibration.Calibration()

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_resample_2d(data_and_metadata: DataAndMetadata.DataAndMetadata, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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

    dimensions = height, width
    resampled_dimensional_calibrations = [Calibration.Calibration(dimensional_calibrations[i].offset, dimensional_calibrations[i].scale * data_shape[i] / dimensions[i], dimensional_calibrations[i].units) for i in range(len(dimensional_calibrations))]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=resampled_dimensional_calibrations)


def function_warp(data_and_metadata: DataAndMetadata.DataAndMetadata, coordinates: typing.Sequence[DataAndMetadata.DataAndMetadata]) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    coords = numpy.moveaxis(numpy.dstack([coordinate.data for coordinate in coordinates]), -1, 0)
    if data_and_metadata.is_data_rgb:
        rgb = numpy.zeros(tuple(data_and_metadata.dimensional_shape) + (3,), numpy.uint8)
        rgb[..., 0] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 0], coords)
        rgb[..., 1] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 1], coords)
        rgb[..., 2] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 2], coords)
        return DataAndMetadata.new_data_and_metadata(rgb)
    elif data_and_metadata.is_data_rgba:
        rgba = numpy.zeros(tuple(data_and_metadata.dimensional_shape) + (4,), numpy.uint8)
        rgba[..., 0] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 0], coords)
        rgba[..., 1] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 1], coords)
        rgba[..., 2] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 2], coords)
        rgba[..., 3] = scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data[..., 3], coords)
        return DataAndMetadata.new_data_and_metadata(rgba)
    else:
        return DataAndMetadata.new_data_and_metadata(scipy.ndimage.interpolation.map_coordinates(data_and_metadata.data, coords))



def function_histogram(data_and_metadata: DataAndMetadata.DataAndMetadata, bins: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

    bins = int(bins)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    data = data_and_metadata.data
    if not Image.is_data_valid(data):
        return None

    histogram_data = numpy.histogram(data, bins=bins)
    min_x = data_and_metadata.intensity_calibration.convert_to_calibrated_value(histogram_data[1][0])
    max_x = data_and_metadata.intensity_calibration.convert_to_calibrated_value(histogram_data[1][-1])
    result_data = histogram_data[0].astype(numpy.int)

    x_calibration = Calibration.Calibration(min_x, (max_x - min_x) / bins, data_and_metadata.intensity_calibration.units)

    return DataAndMetadata.new_data_and_metadata(result_data, dimensional_calibrations=[x_calibration])


def function_line_profile(data_and_metadata: DataAndMetadata.DataAndMetadata, vector: NormVectorType,
                          integration_width: float) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)

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
        length_f = math.sqrt(math.pow(end[0] - start[0], 2) + math.pow(end[1] - start[1], 2))
        samples = int(math.floor(length_f))
        a = numpy.linspace(0, samples - 1, samples)  # along
        t = numpy.linspace(-(n-1)*0.5, (n-1)*0.5, n)  # transverse
        dy = (end[0] - start[0]) / samples
        dx = (end[1] - start[1]) / samples
        ix, iy = numpy.meshgrid(a, t)
        yy = start[0] + dy * ix + dx * iy
        xx = start[1] + dx * ix - dy * iy
        return yy, xx

    # xx, yy = __coordinates(None, (4,4), (8,4), 3)

    data = data_and_metadata.data
    shape = data.shape
    actual_integration_width = min(max(shape[0], shape[1]), integration_width)  # limit integration width to sensible value

    def calculate_data(data):
        if not Image.is_data_valid(data):
            return None
        if Image.is_data_rgb_type(data):
            data = Image.convert_to_grayscale(data, numpy.double)
        start, end = vector
        start_data = (int(shape[0]*start[0]), int(shape[1]*start[1]))
        end_data = (int(shape[0]*end[0]), int(shape[1]*end[1]))
        length = math.sqrt(math.pow(end_data[1] - start_data[1], 2) + math.pow(end_data[0] - start_data[0], 2))
        if length > 1.0:
            spline_order_lookup = { "nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3 }
            method = "nearest"
            spline_order = spline_order_lookup[method]
            yy, xx = get_coordinates(start_data, end_data, actual_integration_width)
            samples = scipy.ndimage.map_coordinates(data, (yy, xx), order=spline_order)
            if len(samples.shape) > 1:
                return numpy.sum(samples, 0)
            else:
                return samples
        else:
            return numpy.zeros((1))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_shape_and_dtype_valid(data_shape, data_dtype) or dimensional_calibrations is None:
        return None

    if dimensional_calibrations is None or len(dimensional_calibrations) != 2:
        return None

    dimensional_calibrations = [Calibration.Calibration(0.0, dimensional_calibrations[1].scale, dimensional_calibrations[1].units)]

    intensity_calibration = copy.deepcopy(data_and_metadata.intensity_calibration)
    intensity_calibration.scale /= actual_integration_width

    return DataAndMetadata.new_data_and_metadata(calculate_data(data), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

def function_make_point(y: float, x: float) -> NormPointType:
    return y, x

def function_make_size(height, width) -> NormSizeType:
    return height, width

def function_make_vector(start, end) -> NormVectorType:
    return start, end

def function_make_rectangle_origin_size(origin, size) -> NormRectangleType:
    return tuple(Geometry.FloatRect(origin, size))

def function_make_rectangle_center_size(center, size) -> NormRectangleType:
    return tuple(Geometry.FloatRect.from_center_and_size(center, size))

def function_make_interval(start, end) -> NormIntervalType:
    return start, end

def function_make_shape(*args) -> DataAndMetadata.ShapeType:
    return tuple([int(arg) for arg in args])

# generic functions

def function_array(array_fn, data_and_metadata: DataAndMetadata.DataAndMetadata, *args, **kwargs) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata)
    data = array_fn(data_and_metadata.data, *args, **kwargs)
    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)

def function_scalar(op, data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.ScalarAndMetadata:
    def calculate_value():
        return op(data_and_metadata.data)

    return DataAndMetadata.ScalarAndMetadata(lambda: calculate_value(), data_and_metadata.intensity_calibration)

def function_display_data_no_copy(data_and_metadata: DataAndMetadata.DataAndMetadata, sequence_index: int=0, collection_index: DataAndMetadata.PositionType=None, slice_center: int=0, slice_width: int=1, complex_display_type: str=None) -> DataAndMetadata.DataAndMetadata:
    dimensional_shape = data_and_metadata.dimensional_shape
    modified = False
    next_dimension = 0
    if data_and_metadata.is_sequence:
        # next dimension is treated as a sequence index, which may be time or just a sequence index
        sequence_index = min(max(sequence_index, 0), dimensional_shape[next_dimension])
        data_and_metadata = DataAndMetadata.function_data_slice(data_and_metadata, [sequence_index, Ellipsis])
        modified = True
        next_dimension += 1
    if data_and_metadata and data_and_metadata.is_collection:
        collection_dimension_count = data_and_metadata.collection_dimension_count
        datum_dimension_count = data_and_metadata.datum_dimension_count
        # next dimensions are treated as collection indexes.
        if collection_dimension_count == 1 and datum_dimension_count == 1 and data_and_metadata.collection_dimension_shape[0] <= 16:
            pass  # this is a special case to display a few rows all at once. once true multi-data displays are available, remove this
        elif collection_dimension_count == 2 and datum_dimension_count == 1:
            data_and_metadata = function_slice_sum(data_and_metadata, slice_center, slice_width)
            modified = True
        else:  # default, "pick"
            collection_slice = [collection_index for collection_index in collection_index][0:collection_dimension_count] + [Ellipsis, ]
            data_and_metadata = DataAndMetadata.function_data_slice(data_and_metadata, collection_slice)
            modified = True
        next_dimension += collection_dimension_count + datum_dimension_count
    if data_and_metadata and data_and_metadata.is_data_complex_type:
        if complex_display_type == "real":
            data_and_metadata = function_array(numpy.real, data_and_metadata)
        elif complex_display_type == "imaginary":
            data_and_metadata = function_array(numpy.imag, data_and_metadata)
        elif complex_display_type == "absolute":
            data_and_metadata = function_array(numpy.absolute, data_and_metadata)
        else:  # default, log-absolute
            def log_absolute(d):
                return numpy.log(numpy.abs(d).astype(numpy.float64) + numpy.nextafter(0,1))
            data_and_metadata = function_array(log_absolute, data_and_metadata)
        modified = True
    if data_and_metadata and functools.reduce(operator.mul, data_and_metadata.dimensional_shape) == 0:
        data_and_metadata = None
    return data_and_metadata, modified

def function_display_data(data_and_metadata: DataAndMetadata.DataAndMetadata, sequence_index: int=0, collection_index: DataAndMetadata.PositionType=None, slice_center: int=0, slice_width: int=1, complex_display_type: str=None) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata, modified = function_display_data_no_copy(data_and_metadata, sequence_index, collection_index, slice_center, slice_width, complex_display_type)
    return copy.deepcopy(data_and_metadata) if data_and_metadata and not modified else data_and_metadata

def function_display_rgba(data_and_metadata: DataAndMetadata.DataAndMetadata, display_range: typing.Tuple[float, float]=None, color_table: numpy.ndarray=None) -> DataAndMetadata.DataAndMetadata:
    data_2d = data_and_metadata.data
    if Image.is_data_1d(data_2d):
        data_2d = data_2d.reshape(1, *data_2d.shape)
    if not Image.is_data_rgb_type(data_2d):
        assert display_range is not None
    assert len(Image.dimensional_shape_from_data(data_2d)) == 2
    rgba_data = Image.create_rgba_image_from_array(data_2d, display_limits=display_range, lookup=color_table)
    return DataAndMetadata.new_data_and_metadata(rgba_data)
