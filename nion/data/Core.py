# standard libraries
import copy
import functools
import math
import operator
import typing

# third party libraries
import numpy
import numpy.typing
import scipy
import scipy.fft
import scipy.ndimage
import scipy.ndimage.filters
import scipy.ndimage.fourier
import scipy.signal

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import Image
from nion.data import TemplateMatching
from nion.utils import Geometry


DataRangeType = typing.Tuple[float, float]
NormIntervalType = typing.Tuple[float, float]
NormChannelType = float
NormRectangleType = typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]
NormPointType = typing.Tuple[float, float]
NormSizeType = typing.Tuple[float, float]
NormVectorType = typing.Tuple[NormPointType, NormPointType]
PickPositionType = typing.Tuple[float, float]

_DataAndMetadataLike = DataAndMetadata._DataAndMetadataLike
_DataAndMetadataIndeterminateSizeLike = DataAndMetadata._DataAndMetadataIndeterminateSizeLike
_DataAndMetadataOrConstant = DataAndMetadata._DataAndMetadataOrConstant
_SliceKeyElementType = DataAndMetadata._SliceKeyElementType

_ImageDataType = Image._ImageDataType


def column(data_and_metadata_in: _DataAndMetadataLike, start: int, stop: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        meshgrid = numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]),
                                  numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]),
                                  sparse=True)  # type: ignore
        return meshgrid[0]  # type: ignore

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def row(data_and_metadata_in: _DataAndMetadataLike, start: int, stop: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        start_0 = start if start is not None else 0
        stop_0 = stop if stop is not None else data_shape(data_and_metadata)[0]
        start_1 = start if start is not None else 0
        stop_1 = stop if stop is not None else data_shape(data_and_metadata)[1]
        meshgrid = numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]),
                                  numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)  # type: ignore
        return meshgrid[1]  # type: ignore

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def radius(data_and_metadata_in: _DataAndMetadataLike, normalize: bool = True) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        start_0 = -1 if normalize else -data_shape(data_and_metadata)[0] * 0.5
        stop_0 = -start_0
        start_1 = -1 if normalize else -data_shape(data_and_metadata)[1] * 0.5
        stop_1 = -start_1
        icol, irow = numpy.meshgrid(numpy.linspace(start_1, stop_1, data_shape(data_and_metadata)[1]), numpy.linspace(start_0, stop_0, data_shape(data_and_metadata)[0]), sparse=True)  # type: ignore
        return numpy.sqrt(icol * icol + irow * irow)  # type: ignore

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def full(shape: DataAndMetadata.ShapeType, fill_value: typing.Any, dtype: typing.Optional[numpy.typing.DTypeLike] = None) -> DataAndMetadata.DataAndMetadata:
    """Generate a constant valued image with the given shape.

    full(4, shape(4, 5))
    full(0, data_shape(b))
    """
    dtype = dtype if dtype else numpy.dtype(numpy.float64)

    return DataAndMetadata.new_data_and_metadata(numpy.full(shape, DataAndMetadata.extract_data(fill_value), dtype))


def arange(start: int, stop: typing.Optional[int] = None, step: typing.Optional[int] = None) -> DataAndMetadata.DataAndMetadata:
    if stop is None:
        start = 0
        stop = start
    if step is None:
        step = 1
    return DataAndMetadata.new_data_and_metadata(numpy.linspace(int(start), int(stop), int(step)))


def linspace(start: float, stop: float, num: int, endpoint: bool = True) -> DataAndMetadata.DataAndMetadata:
    return DataAndMetadata.new_data_and_metadata(numpy.linspace(start, stop, num, endpoint))


def logspace(start: float, stop: float, num: int, endpoint: bool = True, base: float = 10.0) -> DataAndMetadata.DataAndMetadata:
    return DataAndMetadata.new_data_and_metadata(numpy.logspace(start, stop, num, endpoint, base))


def apply_dist(data_and_metadata_in: _DataAndMetadataLike, mean: float, stddev: float, dist: typing.Callable[..., typing.Any], fn: str) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    return DataAndMetadata.new_data_and_metadata(getattr(dist(loc=mean, scale=stddev), fn)(data_and_metadata.data))


def data_shape(data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.ShapeType:
    return data_and_metadata.data_shape


def astype(data: _ImageDataType, dtype: numpy.typing.DTypeLike) -> _ImageDataType:
    return data.astype(dtype)


dtype_map: typing.Mapping[typing.Any, str] = {int: "int", float: "float", complex: "complex", numpy.int16: "int16",
                                              numpy.int32: "int32", numpy.int64: "int64", numpy.uint8: "uint8",
                                              numpy.uint16: "uint16", numpy.uint32: "uint32", numpy.uint64: "uint64",
                                              numpy.float32: "float32", numpy.float64: "float64",
                                              numpy.complex64: "complex64", numpy.complex128: "complex128"}

dtype_inverse_map = {dtype_map[k]: k for k in dtype_map}


def str_to_dtype(str: str) -> numpy.typing.DTypeLike:
    return dtype_inverse_map.get(str, float)


def dtype_to_str(dtype: numpy.typing.DTypeLike) -> str:
    return dtype_map.get(dtype, "float")


def function_fft(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        # scaling: numpy.sqrt(numpy.mean(numpy.absolute(data_copy)**2)) == numpy.sqrt(numpy.mean(numpy.absolute(data_copy_fft)**2))
        # see https://gist.github.com/endolith/1257010
        if Image.is_data_1d(data):
            scaling = 1.0 / numpy.sqrt(data_shape[0])
            return scipy.fft.fftshift(numpy.multiply(scipy.fft.fft(data), scaling))  # type: ignore
        elif Image.is_data_2d(data):
            if Image.is_data_rgb_type(data):
                if Image.is_data_rgb(data):
                    data_copy = numpy.sum(data[..., :] * (0.2126, 0.7152, 0.0722), 2)
                else:
                    data_copy = numpy.sum(data[..., :] * (0.2126, 0.7152, 0.0722, 0.0), 2)
            else:
                data_copy = numpy.copy(data)  # type: ignore  # let other threads use data while we're processing
            scaling = 1.0 / numpy.sqrt(data_shape[1] * data_shape[0])
            # see https://gist.github.com/cmeyer/d2c9a7636df21d07d91cd73ee06d0ef9
            return scipy.fft.fftshift(numpy.multiply(scipy.fft.fft2(data_copy), scaling))  # type: ignore
        else:
            raise NotImplementedError()

    src_dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("FFT: invalid data")

    assert len(src_dimensional_calibrations) == len(Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype) or ())

    dimensional_calibrations = [
        Calibration.Calibration((-0.5 - 0.5 * data_shape_n) / (dimensional_calibration.scale * data_shape_n),
                                1.0 / (dimensional_calibration.scale * data_shape_n),
                                "1/" + dimensional_calibration.units) for dimensional_calibration, data_shape_n in
        zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=dimensional_calibrations)


def function_ifft(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        # scaling: numpy.sqrt(numpy.mean(numpy.absolute(data_copy)**2)) == numpy.sqrt(numpy.mean(numpy.absolute(data_copy_fft)**2))
        # see https://gist.github.com/endolith/1257010
        if Image.is_data_1d(data):
            scaling = numpy.sqrt(data_shape[0])
            return scipy.fft.ifft(scipy.fft.ifftshift(data) * scaling)  # type: ignore
        elif Image.is_data_2d(data):
            data_copy = numpy.copy(data)  # type: ignore  # let other threads use data while we're processing
            scaling = numpy.sqrt(data_shape[1] * data_shape[0])
            return scipy.fft.ifft2(scipy.fft.ifftshift(data_copy) * scaling)  # type: ignore
        else:
            raise NotImplementedError()

    src_dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Inverse FFT: invalid data")

    assert len(src_dimensional_calibrations) == len(Image.dimensional_shape_from_shape_and_dtype(data_shape, data_dtype) or ())

    def remove_one_slash(s: str) -> str:
        if s.startswith("1/"):
            return s[2:]
        else:
            return "1/" + s

    dimensional_calibrations = [Calibration.Calibration(0.0, 1.0 / (dimensional_calibration.scale * data_shape_n),
                                                        remove_one_slash(dimensional_calibration.units)) for
        dimensional_calibration, data_shape_n in zip(src_dimensional_calibrations, data_shape)]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=dimensional_calibrations)


def function_autocorrelate(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_data_2d(data):
            data_copy = numpy.copy(data)  # type: ignore  # let other threads use data while we're processing
            data_std = data_copy.std(dtype=numpy.float64)
            if data_std != 0.0:
                data_norm = (data_copy - data_copy.mean(dtype=numpy.float64)) / data_std
            else:
                data_norm = data_copy
            scaling = 1.0 / (data_norm.shape[0] * data_norm.shape[1])
            data_norm = scipy.fft.rfft2(data_norm)  # type: ignore
            return scipy.fft.fftshift(scipy.fft.irfft2(data_norm * numpy.conj(data_norm))) * scaling  # type: ignore
            # this gives different results. why? because for some reason scipy pads out to 1023 and does calculation.
            # see https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
            # return scipy.signal.fftconvolve(data_copy, numpy.conj(data_copy), mode='same')
        raise NotImplementedError()

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Auto-correlate: invalid data")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_crosscorrelate(*args: _DataAndMetadataIndeterminateSizeLike) -> DataAndMetadata.DataAndMetadata:
    if len(args) != 2:
        raise ValueError("Cross correlate: expects two inputs")

    input1, input2 = args[0], args[1]

    data_and_metadata_or_constant1 = DataAndMetadata.promote_indeterminate_array(input1)
    data_and_metadata_or_constant2 = DataAndMetadata.promote_indeterminate_array(input2)

    shape = DataAndMetadata.determine_shape(data_and_metadata_or_constant1, data_and_metadata_or_constant2)

    if shape is None:
        raise ValueError("Cross correlate: data shapes do not match or are indeterminate")

    data_and_metadata1 = DataAndMetadata.promote_constant(data_and_metadata_or_constant1, shape)
    data_and_metadata2 = DataAndMetadata.promote_constant(data_and_metadata_or_constant2, shape)

    def calculate_data() -> _ImageDataType:
        data1 = data_and_metadata1._data_ex
        data2 = data_and_metadata2._data_ex
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
            return scipy.fft.fftshift(scipy.fft.irfft2(scipy.fft.rfft2(norm1) * numpy.conj(scipy.fft.rfft2(norm2)))) * scaling  # type: ignore
            # this gives different results. why? because for some reason scipy pads out to 1023 and does calculation.
            # see https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
            # return scipy.signal.fftconvolve(data1.copy(), numpy.conj(data2.copy()), mode='same')
        raise NotImplementedError()

    if not Image.is_data_valid(data_and_metadata1.data):
        raise ValueError("Cross correlate: invalid data 1")

    if not Image.is_data_valid(data_and_metadata2.data):
        raise ValueError("Cross correlate: invalid data 2")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), dimensional_calibrations=data_and_metadata1.dimensional_calibrations)


def function_register(data1_in: _DataAndMetadataLike, data2_in: _DataAndMetadataLike, subtract_means: bool,
                      bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> typing.Tuple[float, ...]:
    xdata1 = DataAndMetadata.promote_ndarray(data1_in)
    xdata2 = DataAndMetadata.promote_ndarray(data2_in)
    # data dimensionality and descriptors should match
    assert len(xdata1.data_shape) == len(xdata2.data_shape)
    assert xdata1.data_descriptor == xdata2.data_descriptor
    # get the raw data
    data1 = xdata1.data
    data2 = xdata2.data
    if data1 is None:
        return tuple()
    if data2 is None:
        return tuple()
    assert data1 is not None
    assert data2 is not None
    # take the slice if there is one
    if bounds is not None:
        d_rank = xdata1.datum_dimension_count
        shape = data1.shape
        bounds_pixels = numpy.rint(numpy.array(bounds) * numpy.array(shape)).astype(numpy.int_)
        bounds_slice: typing.Optional[typing.Union[slice, typing.Tuple[slice, ...]]]
        if d_rank == 1:
            bounds_slice = slice(max(0, bounds_pixels[0]), min(shape[0], bounds_pixels[1]))
        elif d_rank == 2:
            bounds_slice = (slice(max(0, bounds_pixels[0][0]), min(shape[0], bounds_pixels[0][0]+bounds_pixels[1][0])),
                            slice(max(0, bounds_pixels[0][1]), min(shape[1], bounds_pixels[0][1]+bounds_pixels[1][1])))
        else:
            bounds_slice = None
        data1 = data1[bounds_slice]
        data2 = data2[bounds_slice]
        assert data1 is not None
        assert data2 is not None
    # subtract the means if desired
    if subtract_means:
        data1 = data1 - typing.cast(float, numpy.average(data1))  # type: ignore
        data2 = data2 - typing.cast(float, numpy.average(data2))  # type: ignore
    # carry out the registration
    ccorr = scipy.signal.correlate(data1, data2, mode="same")
    max_pos = TemplateMatching.find_ccorr_max(ccorr)[2]
    assert max_pos is not None
    return tuple(max_pos[i] - data1.shape[i] * 0.5 for i in range(len(data1.shape)))


def function_match_template(image_xdata_in: _DataAndMetadataLike, template_xdata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    """
    Calculates the normalized cross-correlation for a template with an image. The returned xdata will have the same
    shape as `image_xdata`.
    Inputs can be 1D or 2D and the template must be smaller than or the same size as the image.
    """
    image_xdata = DataAndMetadata.promote_ndarray(image_xdata_in)
    template_xdata = DataAndMetadata.promote_ndarray(template_xdata_in)
    assert image_xdata.is_data_2d or image_xdata.is_data_1d
    assert template_xdata.is_data_2d or template_xdata.is_data_1d
    assert image_xdata.data_descriptor == template_xdata.data_descriptor
    # The template needs to be the smaller of the two if they have different shape
    assert numpy.less_equal(template_xdata.data_shape, image_xdata.data_shape).all()
    image = image_xdata.data
    template = template_xdata.data
    assert image is not None
    assert template is not None
    squeeze = False
    if image_xdata.is_data_1d:
        image = image[..., numpy.newaxis]
        template = template[..., numpy.newaxis]
        assert image is not None
        assert template is not None
        squeeze = True
    ccorr = TemplateMatching.match_template(image, template)
    if squeeze:
        ccorr = numpy.squeeze(ccorr)
    return DataAndMetadata.new_data_and_metadata(ccorr, dimensional_calibrations=image_xdata.dimensional_calibrations)


def function_register_template(image_xdata_in: _DataAndMetadataLike, template_xdata_in: _DataAndMetadataLike, ccorr_mask: typing.Optional[_DataAndMetadataLike] = None) -> typing.Tuple[float, typing.Tuple[float, ...]]:
    """
    Calculates and returns the position of a template on an image. The returned values are the intensity if the
    normalized cross-correlation peak (between -1 and 1) and the sub-pixel position of the template on the image.
    The sub-pixel position is calculated by fitting a parabola to the tip of the cross-correlation peak.
    Inputs can be 1D or 2D and the template must be smaller than or the same size as the image.
    If "ccorr_mask" is not "None", it should be a boolean array with the same shape as "image_xdata_in". It is then
    used to mask the cross-correlation array before finding the maximum.
    """
    image_xdata = DataAndMetadata.promote_ndarray(image_xdata_in)
    template_xdata = DataAndMetadata.promote_ndarray(template_xdata_in)
    ccorr_mask_promoted = None
    if ccorr_mask is not None:
        ccorr_mask_promoted = DataAndMetadata.promote_ndarray(ccorr_mask)
    ccorr_xdata = function_match_template(image_xdata, template_xdata)

    if ccorr_xdata:
        ccorr_data = ccorr_xdata.data
        if ccorr_data is not None:
            if ccorr_mask_promoted is not None:
                ccorr_data *= ccorr_mask_promoted.data
            error, ccoeff, max_pos = TemplateMatching.find_ccorr_max(ccorr_data)
            if not error and ccoeff is not None and max_pos is not None:
                return ccoeff, tuple(max_pos[i] - image_xdata.data_shape[i] // 2 for i in range(len(image_xdata.data_shape)))
    return 0.0, (0.0, ) * len(image_xdata.data_shape)


def function_shift(src_in: _DataAndMetadataLike, shift: typing.Tuple[float, ...], *, order: int = 1) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not Image.is_data_valid(src.data):
        raise ValueError("Shift: invalid data")
    src_data = src._data_ex
    shifted = scipy.ndimage.shift(src_data, shift, order=order, cval=numpy.mean(src_data))
    return DataAndMetadata.new_data_and_metadata(numpy.squeeze(shifted))


def function_fourier_shift(src_in: _DataAndMetadataLike, shift: typing.Tuple[float, ...]) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not Image.is_data_valid(src.data):
        raise ValueError("Shift: invalid data")
    src_data = scipy.fft.fftn(src._data_ex)  # type: ignore
    do_squeeze = False
    if len(src_data.shape) == 1:
        src_data = src_data[..., numpy.newaxis]
        shift = tuple(shift) + (1,)
        do_squeeze = True
    # NOTE: fourier_shift assumes non-fft-shifted data.
    shifted = scipy.fft.ifftn(scipy.ndimage.fourier_shift(src_data, shift)).real  # type: ignore
    shifted = numpy.squeeze(shifted) if do_squeeze else shifted
    return DataAndMetadata.new_data_and_metadata(shifted)


def function_align(src_in: _DataAndMetadataLike, target_in: _DataAndMetadataLike, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    """Aligns target to src and returns align target, using Fourier space."""
    src = DataAndMetadata.promote_ndarray(src_in)
    target = DataAndMetadata.promote_ndarray(target_in)
    return function_shift(target, function_register(src, target, True, bounds=bounds))


def function_fourier_align(src_in: _DataAndMetadataLike, target_in: _DataAndMetadataLike, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    """Aligns target to src and returns align target, using Fourier space."""
    src = DataAndMetadata.promote_ndarray(src_in)
    target = DataAndMetadata.promote_ndarray(target_in)
    return function_fourier_shift(target, function_register(src, target, True, bounds=bounds))


def function_sequence_register_translation(src_in: _DataAndMetadataLike, subtract_means: bool, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    # measures shift relative to last position in sequence
    # only works on sequences
    src = DataAndMetadata.promote_ndarray(src_in)
    if not src.is_navigable:
        raise ValueError("Sequence register translation: source must be a collection or sequence.")
    d_rank = src.datum_dimension_count
    if d_rank not in (1, 2):
        raise ValueError("Sequence register translation: source must be have 1 or 2 dimension data.")
    src_shape = tuple(src.data_shape)
    s_shape = src_shape[0:-d_rank]
    c = int(numpy.product(s_shape, dtype=numpy.uint64))  # type: ignore
    result = numpy.empty(s_shape + (d_rank, ))
    previous_data = None
    src_data = src._data_ex
    for i in range(c):
        ii = numpy.unravel_index(i, s_shape) + (..., )
        if previous_data is None:
            previous_data = src_data[ii]
            result[0, ...] = 0
        else:
            current_data = src_data[ii]
            result[ii] = function_register(previous_data, current_data, subtract_means, bounds=bounds)
            previous_data = current_data
    intensity_calibration = src.dimensional_calibrations[1]  # not the sequence dimension
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 1))


def function_sequence_measure_relative_translation(src_in: _DataAndMetadataLike, ref_in: _DataAndMetadataLike, subtract_means: bool, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    # measures shift at each point in sequence/collection relative to reference
    src = DataAndMetadata.promote_ndarray(src_in)
    if not src.is_navigable:
        raise ValueError("Sequence register translation: source must be a collection or sequence.")
    d_rank = src.datum_dimension_count
    if d_rank not in (1, 2):
        raise ValueError("Sequence register translation: source must be have 1 or 2 dimension data.")
    src_shape = tuple(src.data_shape)
    s_shape = src_shape[0:-d_rank]
    c = int(numpy.product(s_shape, dtype=numpy.uint64))  # type: ignore
    result = numpy.empty(s_shape + (d_rank, ))
    src_data = src._data_ex
    for i in range(c):
        ii = numpy.unravel_index(i, s_shape)
        current_data = src_data[ii]
        result[ii] = function_register(ref_in, current_data, subtract_means, bounds=bounds)
    intensity_calibration = src.dimensional_calibrations[1]  # not the sequence dimension
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration, data_descriptor=DataAndMetadata.DataDescriptor(src.is_sequence, src.collection_dimension_count, 1))


def function_squeeze_measurement(src_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    # squeezes a measurement of a sequence or collection so that it can be sensibly displayed
    src = DataAndMetadata.promote_ndarray(src_in)
    data = src._data_ex
    descriptor = src.data_descriptor
    calibrations = list(src.dimensional_calibrations)
    if descriptor.is_sequence and data.shape[0] == 1:
        data = numpy.squeeze(data, axis=0)
        descriptor = DataAndMetadata.DataDescriptor(False, descriptor.collection_dimension_count, descriptor.datum_dimension_count)
        calibrations.pop(0)
    for index in reversed(descriptor.collection_dimension_indexes):
        if data.shape[index] == 1:
            data = numpy.squeeze(data, axis=index)
            descriptor = DataAndMetadata.DataDescriptor(descriptor.is_sequence, descriptor.collection_dimension_count - 1, descriptor.datum_dimension_count)
            calibrations.pop(index)
    for index in reversed(descriptor.datum_dimension_indexes):
        if data.shape[index] == 1:
            if descriptor.datum_dimension_count > 1:
                data = numpy.squeeze(data, axis=index)
                descriptor = DataAndMetadata.DataDescriptor(descriptor.is_sequence, descriptor.collection_dimension_count, descriptor.datum_dimension_count - 1)
                calibrations.pop(index)
            elif descriptor.collection_dimension_count > 0:
                data = numpy.squeeze(data, axis=index)
                descriptor = DataAndMetadata.DataDescriptor(descriptor.is_sequence, 0, descriptor.collection_dimension_count)
                calibrations.pop(index)
            elif descriptor.is_sequence:
                data = numpy.squeeze(data, axis=index)
                descriptor = DataAndMetadata.DataDescriptor(False, 0, 1)
                calibrations.pop(index)
    intensity_calibration = src.intensity_calibration
    intensity_calibration.offset = 0.0
    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=calibrations, data_descriptor=descriptor)


def function_sequence_align(src_in: _DataAndMetadataLike, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not src.is_navigable:
        raise ValueError("Sequence register translation: source must be a collection or sequence.")
    d_rank = src.datum_dimension_count
    if d_rank not in (1, 2):
        raise ValueError("Sequence register translation: source must be have 1 or 2 dimension data.")
    src_shape = list(src.data_shape)
    s_shape = src_shape[0:-d_rank]
    c = int(numpy.product(s_shape, dtype=numpy.uint64))  # type: ignore
    ref = src[numpy.unravel_index(0, s_shape) + (..., )]
    translations = function_sequence_measure_relative_translation(src, ref, True, bounds=bounds)
    result_data = numpy.copy(src.data)  # type: ignore
    for i in range(1, c):
        ii = numpy.unravel_index(i, s_shape) + (..., )
        new_data = numpy.copy(result_data[ii])  # type: ignore
        current_xdata = DataAndMetadata.new_data_and_metadata(new_data)
        translation = translations._data_ex[numpy.unravel_index(i, s_shape)]
        shift_xdata = function_shift(current_xdata, tuple(translation))
        if shift_xdata:
            result_data[ii] = shift_xdata.data
    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=src.intensity_calibration, dimensional_calibrations=src.dimensional_calibrations, data_descriptor=src.data_descriptor)


def function_sequence_fourier_align(src_in: _DataAndMetadataLike, bounds: typing.Optional[typing.Union[NormRectangleType, NormIntervalType]] = None) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not src.is_navigable:
        raise ValueError("Sequence register translation: source must be a collection or sequence.")
    d_rank = src.datum_dimension_count
    if d_rank not in (1, 2):
        raise ValueError("Sequence register translation: source must be have 1 or 2 dimension data.")
    src_shape = list(src.data_shape)
    s_shape = src_shape[0:-d_rank]
    c = int(numpy.product(s_shape, dtype=numpy.uint64))  # type: ignore
    ref = src[numpy.unravel_index(0, s_shape) + (..., )]
    translations = function_sequence_measure_relative_translation(src, ref, True, bounds=bounds)
    result_data = numpy.copy(src.data)  # type: ignore
    for i in range(1, c):
        ii = numpy.unravel_index(i, s_shape) + (..., )
        new_data = numpy.copy(result_data[ii])  # type: ignore
        current_xdata = DataAndMetadata.new_data_and_metadata(new_data)
        translation = translations._data_ex[numpy.unravel_index(i, s_shape)]
        shift_xdata = function_fourier_shift(current_xdata, tuple(translation))
        if shift_xdata:
            result_data[ii] = shift_xdata.data
    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=src.intensity_calibration, dimensional_calibrations=src.dimensional_calibrations, data_descriptor=src.data_descriptor)


def function_sequence_integrate(src_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not (src.is_sequence or src.collection_dimension_count == 1):
        raise ValueError("Sequence integrate: source must be a 1D collection or a sequence.")
    result = numpy.sum(src._data_ex, axis=0)
    intensity_calibration = src.intensity_calibration
    dimensional_calibrations = src.dimensional_calibrations[1:]
    data_descriptor = DataAndMetadata.DataDescriptor(False, src.data_descriptor.collection_dimension_count, src.data_descriptor.datum_dimension_count)
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_sequence_trim(src_in: _DataAndMetadataLike, trim_start: int, trim_end: int) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not (src.is_sequence or src.collection_dimension_count == 1):
        raise ValueError("Sequence trim: source must be a 1D collection or a sequence.")
    c = src.sequence_dimension_shape[0]
    cs = max(0, int(trim_start))
    ce = min(c, max(cs + 1, int(trim_end)))
    return src[cs:ce]


def function_sequence_insert(src1_in: _DataAndMetadataLike, src2_in: _DataAndMetadataLike, position: int) -> DataAndMetadata.DataAndMetadata:
    src1 = DataAndMetadata.promote_ndarray(src1_in)
    src2 = DataAndMetadata.promote_ndarray(src2_in)
    if not src1.navigation_dimension_count != 1 or not src2.navigation_dimension_count != 1:
        raise ValueError("Sequence insert: both sources must be 1D collections or sequences.")
    if src1.datum_dimension_shape != src2.datum_dimension_shape:
        raise ValueError("Sequence insert: both sources must have same datum shape.")
    c = src1.sequence_dimension_shape[0]
    channel = max(0, min(c, int(position)))
    result: numpy.typing.NDArray[typing.Any] = numpy.vstack([src1._data_ex[:channel], src2._data_ex, src1._data_ex[channel:]])
    intensity_calibration = src1.intensity_calibration
    dimensional_calibrations = src1.dimensional_calibrations
    data_descriptor = src1.data_descriptor
    return DataAndMetadata.new_data_and_metadata(result, intensity_calibration=intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations,
                                                 data_descriptor=data_descriptor)


def function_sequence_concatenate(src1_in: _DataAndMetadataLike, src2_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    src1 = DataAndMetadata.promote_ndarray(src1_in)
    src2 = DataAndMetadata.promote_ndarray(src2_in)
    return function_sequence_insert(src1, src2, src1.data_shape[0])


def function_sequence_join(data_and_metadata_like_list: typing.Sequence[_DataAndMetadataLike]) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_like_list]

    if not data_and_metadata_list:
        raise ValueError("Sequence join: must have at least one item to join.")

    def ensure_sequence(xdata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        data = xdata._data_ex
        if xdata.is_sequence:
            return xdata
        sequence_data = numpy.reshape(data, (1,) + data.shape)
        dimensional_calibrations = [Calibration.Calibration()] + list(xdata.dimensional_calibrations)
        data_descriptor = DataAndMetadata.DataDescriptor(True, xdata.collection_dimension_count, xdata.datum_dimension_count)
        return DataAndMetadata.new_data_and_metadata(sequence_data, dimensional_calibrations=dimensional_calibrations,
                                                     intensity_calibration=xdata.intensity_calibration,
                                                     data_descriptor=data_descriptor)

    sequence_xdata_list = [ensure_sequence(xdata) for xdata in data_and_metadata_list]
    xdata_0 = sequence_xdata_list[0]
    non_sequence_shape_0 = xdata_0.data_shape[1:]
    for xdata in sequence_xdata_list[1:]:
        if xdata.data_shape[1:] != non_sequence_shape_0:
            raise ValueError("Sequence join: all sources must have same datum shape.")
    return function_concatenate(sequence_xdata_list)


def function_sequence_extract(src_in: _DataAndMetadataLike, position: int) -> DataAndMetadata.DataAndMetadata:
    src = DataAndMetadata.promote_ndarray(src_in)
    if src.navigation_dimension_count != 1:
        raise ValueError("Sequence extract: source must be a 1D collection or a sequence.")
    c = src.sequence_dimension_shape[0]
    channel = max(0, min(c, int(position)))
    return src[channel]


def function_sequence_split(src_in: _DataAndMetadataLike) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
    src = DataAndMetadata.promote_ndarray(src_in)
    if not src.is_sequence:
        raise ValueError("Sequence split: source must be a a sequence.")
    dimensional_calibrations = copy.deepcopy(src.dimensional_calibrations[1:])
    data_descriptor = DataAndMetadata.DataDescriptor(False, src.collection_dimension_count, src.datum_dimension_count)
    return [
        DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=copy.deepcopy(dimensional_calibrations),
                                              intensity_calibration=copy.deepcopy(src.intensity_calibration),
                                              data_descriptor=copy.copy(data_descriptor)) for data in src._data_ex]


def function_make_elliptical_mask(data_shape: DataAndMetadata.ShapeType, center: NormPointType, size: NormSizeType, rotation: float) -> DataAndMetadata.DataAndMetadata:
    data_size = Geometry.IntSize.make(typing.cast(Geometry.SizeIntTuple, data_shape))
    data_rect = Geometry.FloatRect(origin=Geometry.FloatPoint(), size=Geometry.FloatSize.make(typing.cast(Geometry.SizeFloatTuple, data_size)))
    center_point = Geometry.map_point(Geometry.FloatPoint.make(center), Geometry.FloatRect.unit_rect(), data_rect)
    size_size = Geometry.map_size(Geometry.FloatSize.make(size), Geometry.FloatRect.unit_rect(), data_rect)
    mask = numpy.zeros((data_size.height, data_size.width))
    bounds = Geometry.FloatRect.from_center_and_size(center_point, size_size)
    if bounds.height <= 0 or bounds.width <= 0:
        return DataAndMetadata.new_data_and_metadata(mask)
    a, b = bounds.center.y, bounds.center.x
    # work around incomplete numpy typing
    y: _ImageDataType
    x: _ImageDataType
    # does ogrid take float args? not sure. typing says "no" so ignore typing for now.
    y, x = numpy.ogrid[-a:data_size.height - a, -b:data_size.width - b]  # type: ignore
    if rotation:
        angle_sin = math.sin(rotation)
        angle_cos = math.cos(rotation)
        # this ugly casting is necessary to work around incomplete numpy typing
        mask_eq = ((typing.cast(_ImageDataType, x * angle_cos) - typing.cast(_ImageDataType, y * angle_sin)) ** 2) / (
                    (bounds.width / 2) * (bounds.width / 2)) + ((typing.cast(_ImageDataType,
                                                                             y * angle_cos) + typing.cast(
            _ImageDataType, x * angle_sin)) ** 2) / ((bounds.height / 2) * (bounds.height / 2)) <= 1
    else:
        mask_eq = x * x / ((bounds.width / 2) * (bounds.width / 2)) + y * y / ((bounds.height / 2) * (bounds.height / 2)) <= 1
    mask[mask_eq] = 1
    return DataAndMetadata.new_data_and_metadata(mask)


def function_fourier_mask(data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike, mask_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(data_and_metadata_in)
    mask_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(mask_data_and_metadata_in)

    shape = DataAndMetadata.determine_shape(data_and_metadata_c, mask_data_and_metadata_c)

    if shape is None:
        raise ValueError("Fourier mask: data and ask shapes do not match or are indeterminate")

    data_and_metadata = DataAndMetadata.promote_constant(data_and_metadata_c, shape)
    mask_data_and_metadata = DataAndMetadata.promote_constant(mask_data_and_metadata_c, shape)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Fourier mask: invalid data")

    if not Image.is_data_valid(mask_data_and_metadata.data):
        raise ValueError("Fourier mask: invalid mask data")

    if not Image.is_data_2d(data_and_metadata.data):
        raise ValueError("Fourier mask: data must be 2D")

    if not Image.is_data_2d(mask_data_and_metadata.data):
        raise ValueError("Fourier mask: data must be 2D")

    data = data_and_metadata._data_ex
    mask_data = mask_data_and_metadata._data_ex

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
    masked_data: _ImageDataType = data * fourier_mask_data

    return DataAndMetadata.new_data_and_metadata(masked_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_sobel(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.sobel(data[..., 0])
            rgb[..., 1] = scipy.ndimage.sobel(data[..., 1])
            rgb[..., 2] = scipy.ndimage.sobel(data[..., 2])
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.sobel(data[..., 0])
            rgba[..., 1] = scipy.ndimage.sobel(data[..., 1])
            rgba[..., 2] = scipy.ndimage.sobel(data[..., 2])
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.sobel(data)  # type: ignore

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Sobel: invalid data")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_laplace(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.laplace(data[..., 0])
            rgb[..., 1] = scipy.ndimage.laplace(data[..., 1])
            rgb[..., 2] = scipy.ndimage.laplace(data[..., 2])
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.laplace(data[..., 0])
            rgba[..., 1] = scipy.ndimage.laplace(data[..., 1])
            rgba[..., 2] = scipy.ndimage.laplace(data[..., 2])
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.laplace(data)  # type: ignore

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Laplace: invalid data")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_gaussian_blur(data_and_metadata_in: _DataAndMetadataLike, sigma: float) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Gaussian blur: invalid data")

    new_data: _ImageDataType
    data = data_and_metadata._data_ex
    if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
        rgb: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
        rgb[..., 0] = scipy.ndimage.gaussian_filter(data[..., 0], sigma=sigma)
        rgb[..., 1] = scipy.ndimage.gaussian_filter(data[..., 1], sigma=sigma)
        rgb[..., 2] = scipy.ndimage.gaussian_filter(data[..., 2], sigma=sigma)
        new_data = rgb
    elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
        rgba: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
        rgba[..., 0] = scipy.ndimage.gaussian_filter(data[..., 0], sigma=sigma)
        rgba[..., 1] = scipy.ndimage.gaussian_filter(data[..., 1], sigma=sigma)
        rgba[..., 2] = scipy.ndimage.gaussian_filter(data[..., 2], sigma=sigma)
        rgba[..., 3] = data[..., 3]
        new_data = rgba
    else:
        new_data = scipy.ndimage.gaussian_filter(data, sigma=sigma)  # type: ignore

    return DataAndMetadata.new_data_and_metadata(new_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_median_filter(data_and_metadata_in: _DataAndMetadataLike, size: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    size = max(min(int(size), 999), 1)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.median_filter(data[..., 0], size=size)
            rgb[..., 1] = scipy.ndimage.median_filter(data[..., 1], size=size)
            rgb[..., 2] = scipy.ndimage.median_filter(data[..., 2], size=size)
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.median_filter(data[..., 0], size=size)
            rgba[..., 1] = scipy.ndimage.median_filter(data[..., 1], size=size)
            rgba[..., 2] = scipy.ndimage.median_filter(data[..., 2], size=size)
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.median_filter(data, size=size)  # type: ignore

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Median filter: invalid data")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_uniform_filter(data_and_metadata_in: _DataAndMetadataLike, size: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    size = max(min(int(size), 999), 1)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
            rgb: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (3,), numpy.uint8)
            rgb[..., 0] = scipy.ndimage.uniform_filter(data[..., 0], size=size)
            rgb[..., 1] = scipy.ndimage.uniform_filter(data[..., 1], size=size)
            rgb[..., 2] = scipy.ndimage.uniform_filter(data[..., 2], size=size)
            return rgb
        elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
            rgba: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[:-1] + (4,), numpy.uint8)
            rgba[..., 0] = scipy.ndimage.uniform_filter(data[..., 0], size=size)
            rgba[..., 1] = scipy.ndimage.uniform_filter(data[..., 1], size=size)
            rgba[..., 2] = scipy.ndimage.uniform_filter(data[..., 2], size=size)
            rgba[..., 3] = data[..., 3]
            return rgba
        else:
            return scipy.ndimage.uniform_filter(data, size=size)  # type: ignore

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Uniform filter: invalid data")

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_transpose_flip(data_and_metadata_in: _DataAndMetadataLike, transpose: bool = False, flip_v: bool = False, flip_h: bool = False) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata._data_ex
        data_id = id(data)
        if transpose:
            if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
                data = numpy.transpose(data, [1, 0, 2])
            elif len(data_and_metadata.data_shape) == 2:
                data = numpy.transpose(data, [1, 0])
        if flip_h and len(data_and_metadata.data_shape) == 2:
            data = numpy.fliplr(data)  # type: ignore
        if flip_v and len(data_and_metadata.data_shape) == 2:
            data = numpy.flipud(data)  # type: ignore
        assert data is not None  # this is required, seems to be a bug in mypy about reassignment
        if id(data) == data_id:  # ensure real data, not a view
            return numpy.copy(data)  # type: ignore
        else:
            return data

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Transpose flip: invalid data")

    if transpose:
        dimensional_calibrations = list(reversed(data_and_metadata.dimensional_calibrations))
    else:
        dimensional_calibrations = list(data_and_metadata.dimensional_calibrations)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_invert(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_data_rgba(data):
                inverted = 255 - data[:]
                inverted[..., 3] = data[..., 3]
                return typing.cast(_ImageDataType, inverted)
            else:
                return typing.cast(_ImageDataType, 255 - data[:])
        else:
            return typing.cast(_ImageDataType, -data[:])

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Invert: invalid data")

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_crop(data_and_metadata_in: _DataAndMetadataLike, bounds: NormRectangleType) -> DataAndMetadata.DataAndMetadata:
    bounds_rect = Geometry.FloatRect.make(bounds)

    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = Geometry.IntSize.make(typing.cast(Geometry.SizeIntTuple, data_and_metadata.data_shape))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    data = data_and_metadata._data_ex

    if not Image.is_data_valid(data):
        raise ValueError("Crop: invalid data")

    oheight = int(data_shape.height * bounds_rect.height)
    owidth = int(data_shape.width * bounds_rect.width)

    top = int(data_shape.height * bounds_rect.top)
    left = int(data_shape.width * bounds_rect.left)
    height = int(data_shape.height * bounds_rect.height)
    width = int(data_shape.width * bounds_rect.width)

    dtop = 0
    dleft = 0
    dheight = height
    dwidth = width

    if top < 0:
        dheight += top
        dtop -= top
        height += top
        top = 0
    if top + height > data_shape.height:
        dheight -= (top + height - data_shape.height)
        height = data_shape.height - top
    if left < 0:
        dwidth += left
        dleft -= left
        width += left
        left = 0
    if left + width > data_shape.width:
        dwidth -= (left + width - data_shape.width)
        width = data_shape.width - left

    data_dtype = data.dtype
    assert data_dtype is not None

    if data_and_metadata.is_data_rgb:
        new_data: numpy.typing.NDArray[typing.Any] = numpy.zeros((oheight, owidth, 3), dtype=data_dtype)
        if height > 0 and width > 0:
            new_data[dtop:dtop + dheight, dleft:dleft + dwidth] = data[top:top + height, left:left + width]
    elif data_and_metadata.is_data_rgba:
        new_data = numpy.zeros((oheight, owidth, 4), dtype=data_dtype)
        if height > 0 and width > 0:
            new_data[dtop:dtop + dheight, dleft:dleft + dwidth] = data[top:top + height, left:left + width]
    else:
        new_data = numpy.zeros((oheight, owidth), dtype=data_dtype)
        if height > 0 and width > 0:
            new_data[dtop:dtop + dheight, dleft:dleft + dwidth] = data[top:top + height, left:left + width]

    cropped_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + data_shape[index] * bounds_rect.origin[index] * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(new_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations)


def function_crop_rotated(data_and_metadata_in: _DataAndMetadataLike, bounds: NormRectangleType, angle: float) -> DataAndMetadata.DataAndMetadata:
    bounds_rect = Geometry.FloatRect.make(bounds)

    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = Geometry.IntSize.make(typing.cast(Geometry.SizeIntTuple, data_and_metadata.data_shape))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    data = data_and_metadata._data_ex

    if not Image.is_data_valid(data):
        raise ValueError("Crop rotated: invalid data")

    top = round(data_shape.height * bounds_rect.top)
    left = round(data_shape.width * bounds_rect.left)
    height = round(data_shape.height * bounds_rect.height)
    width = round(data_shape.width * bounds_rect.width)

    y: _ImageDataType
    x: _ImageDataType
    y, x = numpy.meshgrid(numpy.arange(-(width // 2), width - width // 2), numpy.arange(-(height // 2), height - height // 2))  # type: ignore

    angle_sin = math.sin(angle)
    angle_cos = math.cos(angle)

    # this ugly casting is necessary to work around incomplete numpy typing
    coords = [
        typing.cast(_ImageDataType, top + height // 2 + (typing.cast(_ImageDataType, x * angle_cos) - typing.cast(_ImageDataType, y * angle_sin))),
        typing.cast(_ImageDataType, left + width // 2 + (typing.cast(_ImageDataType, y * angle_cos) + typing.cast(_ImageDataType, x * angle_sin)))]

    new_data: numpy.typing.NDArray[numpy.uint8]
    if data_and_metadata.is_data_rgb:
        new_data = numpy.zeros(coords[0].shape + (3,), numpy.uint8)
        new_data[..., 0] = scipy.ndimage.map_coordinates(data[..., 0], coords)
        new_data[..., 1] = scipy.ndimage.map_coordinates(data[..., 1], coords)
        new_data[..., 2] = scipy.ndimage.map_coordinates(data[..., 2], coords)
    elif data_and_metadata.is_data_rgba:
        new_data = numpy.zeros(coords[0].shape + (4,), numpy.uint8)
        new_data[..., 0] = scipy.ndimage.map_coordinates(data[..., 0], coords)
        new_data[..., 1] = scipy.ndimage.map_coordinates(data[..., 1], coords)
        new_data[..., 2] = scipy.ndimage.map_coordinates(data[..., 2], coords)
        new_data[..., 3] = scipy.ndimage.map_coordinates(data[..., 3], coords)
    else:
        new_data = scipy.ndimage.map_coordinates(data, coords)

    cropped_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + data_shape[index] * bounds_rect[0][index] * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(new_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations)


def function_crop_interval(data_and_metadata_in: _DataAndMetadataLike, interval: NormIntervalType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata._data_ex
        data_shape = data_and_metadata.data_shape
        interval_int = int(data_shape[0] * interval[0]), int(data_shape[0] * interval[1])
        return typing.cast(_ImageDataType, data[interval_int[0]:interval_int[1]].copy())

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Crop interval: invalid data")

    interval_int = int(data_shape[0] * interval[0]), int(data_shape[0] * interval[1])

    cropped_dimensional_calibrations = list()
    dimensional_calibration = dimensional_calibrations[0]
    cropped_calibration = Calibration.Calibration(
        dimensional_calibration.offset + data_shape[0] * interval_int[0] * dimensional_calibration.scale,
        dimensional_calibration.scale, dimensional_calibration.units)
    cropped_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=cropped_dimensional_calibrations)


def function_slice_sum(data_and_metadata_in: _DataAndMetadataLike, slice_center: int, slice_width: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    signal_index = -1

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata._data_ex
        shape = data.shape
        slice_start = int(slice_center - slice_width * 0.5 + 0.5)
        slice_start = max(slice_start, 0)
        slice_end = slice_start + slice_width
        slice_end = min(shape[signal_index], slice_end)
        return typing.cast(_ImageDataType, numpy.sum(data[..., slice_start:slice_end], signal_index))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Slice sum: invalid data")

    dimensional_calibrations = dimensional_calibrations[0:signal_index]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_pick(data_and_metadata_in: _DataAndMetadataLike, position: PickPositionType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Pick: invalid data")

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata._data_ex
        collection_dimensions = data_and_metadata.dimensional_shape[data_and_metadata.collection_dimension_slice]
        datum_dimensions = data_and_metadata.dimensional_shape[data_and_metadata.datum_dimension_slice]
        position_i: typing.List[typing.Union[slice, int, ellipsis]] = list()
        for collection_dimension, pos in zip(collection_dimensions, position):
            pos_i = int(pos * collection_dimension)
            if not (0 <= pos_i < collection_dimension):
                return numpy.zeros(datum_dimensions, dtype=data.dtype)
            position_i.append(pos_i)
        if data_and_metadata.is_sequence:
            position_i.insert(0, slice(None))
        position_i.append(...)
        return typing.cast(_ImageDataType, data[tuple(position_i)].copy())

    dimensional_calibrations = data_and_metadata.dimensional_calibrations
    data_descriptor = DataAndMetadata.DataDescriptor(data_and_metadata.is_sequence, 0, data_and_metadata.datum_dimension_count)

    if len(position) != data_and_metadata.collection_dimension_count:
        raise ValueError("Pick: position length much master navigation dimension count.")

    if data_and_metadata.is_sequence:
        dimensional_calibrations = [dimensional_calibrations[0]] + list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])
    else:
        dimensional_calibrations = list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])

    return DataAndMetadata.new_data_and_metadata(calculate_data(),
                                                 intensity_calibration=data_and_metadata.intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations,
                                                 data_descriptor=data_descriptor)


def function_concatenate(data_and_metadata_like_list: typing.Sequence[_DataAndMetadataLike], axis: int = 0) -> DataAndMetadata.DataAndMetadata:
    """Concatenate multiple data_and_metadatas.

    concatenate((a, b, c), 1)

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.
    Keeps data descriptor of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_like_list]

    if not data_and_metadata_list:
        raise ValueError("Concatenate: must have at least one item to join.")

    if any([not Image.is_data_valid(data_and_metadata.data) for data_and_metadata in data_and_metadata_list]):
        raise ValueError("Concatenate: invalid data")

    partial_shape = data_and_metadata_list[0].data_shape

    if any([data_and_metadata.data_shape != partial_shape[1:] is None for data_and_metadata in data_and_metadata_list]):
        raise ValueError("Concatenate: all data must have same shape.")

    dimensional_calibrations: typing.List[Calibration.Calibration] = [typing.cast(Calibration.Calibration, None)] * len(data_and_metadata_list[0].dimensional_calibrations)
    for data_and_metadata in data_and_metadata_list:
        for index, calibration in enumerate(data_and_metadata.dimensional_calibrations):
            if dimensional_calibrations[index] is None:
                dimensional_calibrations[index] = calibration
            elif dimensional_calibrations[index] != calibration:
                dimensional_calibrations[index] = Calibration.Calibration()

    intensity_calibration = data_and_metadata_list[0].intensity_calibration
    data_descriptor = data_and_metadata_list[0].data_descriptor

    data_list = list(data_and_metadata.data for data_and_metadata in data_and_metadata_list)
    data = numpy.concatenate(data_list, axis)  # type: ignore

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_hstack(data_and_metadata_like_list: typing.Sequence[_DataAndMetadataLike]) -> DataAndMetadata.DataAndMetadata:
    """Stack multiple data_and_metadatas along axis 1.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_like_list]

    if not data_and_metadata_list:
        raise ValueError("H Stack: must have at least one item to join.")

    if any([not Image.is_data_valid(data_and_metadata.data) for data_and_metadata in data_and_metadata_list]):
        raise ValueError("H Stack: invalid data")

    partial_shape = data_and_metadata_list[0].data_shape

    if len(partial_shape) >= 2:
        return function_concatenate(data_and_metadata_list, 1)
    else:
        return function_concatenate(data_and_metadata_list, 0)


def function_vstack(data_and_metadata_like_list: typing.Sequence[_DataAndMetadataLike]) -> DataAndMetadata.DataAndMetadata:
    """Stack multiple data_and_metadatas along axis 0.

    hstack((a, b, c))

    Function is called by passing a tuple of the list of source items, which matches the
    form of the numpy function of the same name.

    Keeps intensity calibration of first source item.

    Keeps dimensional calibration in axis dimension.
    """
    data_and_metadata_list = [DataAndMetadata.promote_ndarray(data_and_metadata) for data_and_metadata in data_and_metadata_like_list]

    if not data_and_metadata_list:
        raise ValueError("V Stack: must have at least one item to join.")

    if any([not Image.is_data_valid(data_and_metadata.data) for data_and_metadata in data_and_metadata_list]):
        raise ValueError("V Stack: invalid data")

    partial_shape = data_and_metadata_list[0].data_shape

    if len(partial_shape) >= 2:
        return function_concatenate(data_and_metadata_list, 0)

    dimensional_calibrations = list()
    dimensional_calibrations.append(Calibration.Calibration())
    dimensional_calibrations.append(data_and_metadata_list[0].dimensional_calibrations[0])

    intensity_calibration = data_and_metadata_list[0].intensity_calibration

    data_descriptor = data_and_metadata_list[0].data_descriptor

    data_descriptor = DataAndMetadata.DataDescriptor(data_descriptor.is_sequence, data_descriptor.collection_dimension_count + 1, data_descriptor.datum_dimension_count)

    data_list = list(data_and_metadata.data for data_and_metadata in data_and_metadata_list)
    data: numpy.typing.NDArray[typing.Any] = numpy.vstack(typing.cast(typing.Sequence[DataAndMetadata.DataAndMetadata], data_list))

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_moveaxis(data_and_metadata_in: _DataAndMetadataLike, src_axis: int, dst_axis: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data = numpy.moveaxis(data_and_metadata._data_ex, src_axis, dst_axis)

    dimensional_calibrations = list(copy.deepcopy(data_and_metadata.dimensional_calibrations))

    dimensional_calibrations.insert(dst_axis, dimensional_calibrations.pop(src_axis))

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_sum(data_and_metadata_in: _DataAndMetadataLike, axis: typing.Optional[typing.Union[int, typing.Sequence[int]]] = None, keepdims: bool = False) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
                rgb_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[1:], numpy.uint8)
                rgb_image[:, 0] = numpy.average(data[..., 0], axis)  # type: ignore
                rgb_image[:, 1] = numpy.average(data[..., 1], axis)  # type: ignore
                rgb_image[:, 2] = numpy.average(data[..., 2], axis)  # type: ignore
                return rgb_image
            else:
                rgba_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[1:], numpy.uint8)
                rgba_image[:, 0] = numpy.average(data[..., 0], axis)  # type: ignore
                rgba_image[:, 1] = numpy.average(data[..., 1], axis)  # type: ignore
                rgba_image[:, 2] = numpy.average(data[..., 2], axis)  # type: ignore
                rgba_image[:, 3] = numpy.average(data[..., 3], axis)  # type: ignore
                return rgba_image
        else:
            return typing.cast(_ImageDataType, numpy.sum(data, typing.cast(typing.Any, axis), keepdims=keepdims))

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Sum: invalid data")

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    new_dimensional_calibrations = list()

    if not keepdims or Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
        assert axis is not None
        axes: numpy.typing.NDArray[typing.Any] = numpy.atleast_1d(axis)
        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] += len(dimensional_calibrations)
        for i in range(len(dimensional_calibrations)):
            if i not in axes:
                new_dimensional_calibrations.append(dimensional_calibrations[i])

    dimensional_calibrations = new_dimensional_calibrations

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_mean(data_and_metadata_in: _DataAndMetadataLike, axis: typing.Optional[typing.Union[int, typing.Sequence[int]]] = None, keepdims: bool = False) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape
    data_dtype = data_and_metadata.data_dtype

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if Image.is_shape_and_dtype_rgb_type(data.shape, data.dtype):
            if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
                rgb_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[1:], numpy.uint8)
                rgb_image[:, 0] = numpy.average(data[..., 0], axis)  # type: ignore
                rgb_image[:, 1] = numpy.average(data[..., 1], axis)  # type: ignore
                rgb_image[:, 2] = numpy.average(data[..., 2], axis)  # type: ignore
                return rgb_image
            else:
                rgba_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(data.shape[1:], numpy.uint8)
                rgba_image[:, 0] = numpy.average(data[..., 0], axis)  # type: ignore
                rgba_image[:, 1] = numpy.average(data[..., 1], axis)  # type: ignore
                rgba_image[:, 2] = numpy.average(data[..., 2], axis)  # type: ignore
                rgba_image[:, 3] = numpy.average(data[..., 3], axis)  # type: ignore
                return rgba_image
        else:
            return typing.cast(_ImageDataType, numpy.mean(data, axis, keepdims=keepdims))

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Mean: invalid data")

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    new_dimensional_calibrations = list()

    if not keepdims or Image.is_shape_and_dtype_rgb_type(data_shape, data_dtype):
        assert axis is not None
        axes: numpy.typing.NDArray[typing.Any] = numpy.atleast_1d(axis)
        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] += len(dimensional_calibrations)
        for i in range(len(dimensional_calibrations)):
            if i not in axes:
                new_dimensional_calibrations.append(dimensional_calibrations[i])

    dimensional_calibrations = new_dimensional_calibrations

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations)


def function_sum_region(data_and_metadata_in: _DataAndMetadataLike, mask_data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    mask_data_and_metadata = DataAndMetadata.promote_ndarray(mask_data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Sum region: invalid data")

    if not Image.is_data_valid(mask_data_and_metadata.data):
        raise ValueError("Sum region: invalid mask data")

    if data_and_metadata.is_sequence:
        assert len(data_and_metadata.dimensional_shape) == 4
    else:
        assert len(data_and_metadata.dimensional_shape) == 3
    assert len(mask_data_and_metadata.dimensional_shape) == 2

    data = data_and_metadata._data_ex
    mask_data = mask_data_and_metadata._data_ex.astype(bool)  # type: ignore

    start_index = 1 if data_and_metadata.is_sequence else 0
    result_data = numpy.sum(data, axis=tuple(range(start_index, len(data_and_metadata.dimensional_shape) - 1)), where=mask_data[..., numpy.newaxis])

    data_descriptor = DataAndMetadata.DataDescriptor(data_and_metadata.is_sequence, 0, data_and_metadata.datum_dimension_count)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if data_and_metadata.is_sequence:
        dimensional_calibrations = [dimensional_calibrations[0]] + list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])
    else:
        dimensional_calibrations = list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])

    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_average_region(data_and_metadata_in: _DataAndMetadataLike, mask_data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    mask_data_and_metadata = DataAndMetadata.promote_ndarray(mask_data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Sum region: invalid data")

    if not Image.is_data_valid(mask_data_and_metadata.data):
        raise ValueError("Sum region: invalid mask data")

    if data_and_metadata.is_sequence:
        assert len(data_and_metadata.dimensional_shape) == 4
    else:
        assert len(data_and_metadata.dimensional_shape) == 3
    assert len(mask_data_and_metadata.dimensional_shape) == 2

    data = data_and_metadata._data_ex
    mask_data = mask_data_and_metadata._data_ex.astype(bool)  # type: ignore

    assert data is not None

    mask_sum = max(1.0, typing.cast(float, numpy.sum(mask_data)))

    start_index = 1 if data_and_metadata.is_sequence else 0
    result_data = numpy.sum(data, axis=tuple(range(start_index, len(data_and_metadata.dimensional_shape) - 1)), where=mask_data[..., numpy.newaxis]) / mask_sum

    data_descriptor = DataAndMetadata.DataDescriptor(data_and_metadata.is_sequence, 0, data_and_metadata.datum_dimension_count)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if data_and_metadata.is_sequence:
        dimensional_calibrations = [dimensional_calibrations[0]] + list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])
    else:
        dimensional_calibrations = list(dimensional_calibrations[data_and_metadata.datum_dimension_slice])

    return DataAndMetadata.new_data_and_metadata(result_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)


def function_reshape(data_and_metadata_in: _DataAndMetadataLike, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
    """Reshape a data and metadata to shape.

    reshape(a, shape(4, 5))
    reshape(a, data_shape(b))

    Handles special cases when going to one extra dimension and when going to one fewer
    dimension -- namely to keep the calibrations intact.

    When increasing dimension, a -1 can be passed for the new dimension and this function
    will calculate the missing value.
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data_shape = data_and_metadata.data_shape

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        return numpy.reshape(data, shape)

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Reshape: invalid data")

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


def function_squeeze(data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.DataAndMetadata:
    """Remove dimensions with lengths of one."""
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Squeeze: invalid data")

    data_shape = data_and_metadata.data_shape

    dimensional_calibrations = data_and_metadata.dimensional_calibrations
    is_sequence = data_and_metadata.is_sequence
    collection_dimension_count = data_and_metadata.collection_dimension_count
    datum_dimension_count = data_and_metadata.datum_dimension_count
    new_dimensional_calibrations = list()
    dimensional_index = 0

    # fix the data descriptor and the dimensions
    indexes = list()
    if is_sequence:
        if data_shape[dimensional_index] <= 1:
            is_sequence = False
            indexes.append(dimensional_index)
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1
    for collection_dimension_index in range(collection_dimension_count):
        if data_shape[dimensional_index] <= 1:
            collection_dimension_count -= 1
            indexes.append(dimensional_index)
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1
    for datum_dimension_index in range(datum_dimension_count):
        if data_shape[dimensional_index] <= 1 and datum_dimension_count > 1:
            datum_dimension_count -= 1
            indexes.append(dimensional_index)
        else:
            new_dimensional_calibrations.append(dimensional_calibrations[dimensional_index])
        dimensional_index += 1

    data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

    data = numpy.squeeze(data_and_metadata._data_ex, axis=tuple(indexes))

    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=new_dimensional_calibrations, data_descriptor=data_descriptor)


def function_redimension(data_and_metadata_in: _DataAndMetadataLike, data_descriptor: DataAndMetadata.DataDescriptor) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if data_and_metadata.data_descriptor.expected_dimension_count != data_descriptor.expected_dimension_count:
        raise ValueError("Redimension: overall data array rank must be unchanged")

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Redimension: invalid data")

    return DataAndMetadata.new_data_and_metadata(data_and_metadata._data_ex, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations, data_descriptor=data_descriptor)


def function_resize(data_and_metadata_in: _DataAndMetadataLike, shape: DataAndMetadata.ShapeType, mode: typing.Optional[str] = None) -> DataAndMetadata.DataAndMetadata:
    """Resize a data and metadata to shape, padding if larger, cropping if smaller.

    resize(a, shape(4, 5))
    resize(a, data_shape(b))

    Shape must have same number of dimensions as original.
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Resize: invalid data")

    data_shape = data_and_metadata.data_shape

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata._data_ex
        c = numpy.mean(data)
        data_shape = data_and_metadata.data_shape
        slices = list()
        for data_size, new_size in zip(data_shape, shape):
            if new_size <= data_size:
                left = data_size // 2 - new_size // 2
                slices.append(slice(left, left + new_size))
            else:
                slices.append(slice(None))
        data = data[tuple(slices)]
        data_shape = data_and_metadata.data_shape
        pads = list()
        for data_size, new_size in zip(data_shape, shape):
            if new_size > data_size:
                left = new_size // 2 - data_size // 2
                pads.append((left, new_size - left - data_size))
            else:
                pads.append((0, 0))
        return typing.cast(_ImageDataType, numpy.pad(data, pads, 'constant', constant_values=c))  # type: ignore

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    resized_dimensional_calibrations = list()
    for index, dimensional_calibration in enumerate(dimensional_calibrations):
        offset = data_shape[index] // 2 - shape[index] // 2
        cropped_calibration = Calibration.Calibration(
            dimensional_calibration.offset + offset * dimensional_calibration.scale,
            dimensional_calibration.scale, dimensional_calibration.units)
        resized_dimensional_calibrations.append(cropped_calibration)

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=resized_dimensional_calibrations)


def function_rescale(data_and_metadata_in: _DataAndMetadataLike,
                     data_range: typing.Optional[DataRangeType] = None,
                     in_range: typing.Optional[DataRangeType] = None) -> DataAndMetadata.DataAndMetadata:
    """Rescale data and update intensity calibration.

    rescale(a, (0.0, 1.0))
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Rescale: invalid data")

    used_data_range = data_range if data_range is not None else (0.0, 1.0)

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        data_ptp = numpy.ptp(data) if in_range is None else in_range[1] - in_range[0]
        data_ptp_i = 1.0 / data_ptp if data_ptp != 0.0 else 1.0
        if in_range is not None:
            data_min = in_range[0]
        else:
            data_min = numpy.amin(data)
        data_span = used_data_range[1] - used_data_range[0]
        if data_span == 1.0 and used_data_range[0] == 0.0:
            return typing.cast(_ImageDataType, (data - data_min) * data_ptp_i)
        else:
            m = data_span * data_ptp_i
            return typing.cast(_ImageDataType, (data - data_min) * m + used_data_range[0])

    intensity_calibration = Calibration.Calibration()

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_rebin_2d(data_and_metadata_in: _DataAndMetadataLike, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Rebin 2D: invalid data")

    if not Image.is_data_2d(data_and_metadata.data):
        raise ValueError("Re-bin by 2: data must be 2D")

    height = int(shape[0])
    width = int(shape[1])

    data_shape = data_and_metadata.data_shape

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    height = min(height, data_shape[0])
    width = min(width, data_shape[1])

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if data.shape[0] == height and data.shape[1] == width:
            return numpy.copy(data)  # type: ignore
        shape = height, data.shape[0] // height, width, data.shape[1] // width
        return typing.cast(_ImageDataType, numpy.reshape(data, shape).mean(-1).mean(1))

    dimensions = height, width
    rebinned_dimensional_calibrations = [Calibration.Calibration(dimensional_calibrations[i].offset, dimensional_calibrations[i].scale * data_shape[i] / dimensions[i], dimensional_calibrations[i].units) for i in range(len(dimensional_calibrations))]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=rebinned_dimensional_calibrations)


def function_resample_2d(data_and_metadata_in: _DataAndMetadataLike, shape: DataAndMetadata.ShapeType) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Resample: invalid data")

    height = int(shape[0])
    width = int(shape[1])

    data_shape = data_and_metadata.data_shape

    def calculate_data() -> _ImageDataType:
        data = data_and_metadata.data
        assert data is not None
        if data.shape[0] == height and data.shape[1] == width:
            return numpy.copy(data)  # type: ignore
        return Image.scaled(data, (height, width))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    dimensions = height, width
    resampled_dimensional_calibrations = [Calibration.Calibration(dimensional_calibrations[i].offset, dimensional_calibrations[i].scale * data_shape[i] / dimensions[i], dimensional_calibrations[i].units) for i in range(len(dimensional_calibrations))]

    return DataAndMetadata.new_data_and_metadata(calculate_data(), intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=resampled_dimensional_calibrations)


def function_warp(data_and_metadata_in: _DataAndMetadataLike, coordinates_in: typing.Sequence[_DataAndMetadataLike], order: int = 1) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    coordinates = [DataAndMetadata.promote_ndarray(c) for c in coordinates_in]
    coords = numpy.moveaxis(numpy.dstack([coordinate.data for coordinate in coordinates]), -1, 0)  # type: ignore
    data = data_and_metadata._data_ex
    if data_and_metadata.is_data_rgb:
        rgb: numpy.typing.NDArray[numpy.uint8] = numpy.zeros(tuple(data_and_metadata.dimensional_shape) + (3,), numpy.uint8)
        rgb[..., 0] = scipy.ndimage.map_coordinates(data[..., 0], coords, order=order)
        rgb[..., 1] = scipy.ndimage.map_coordinates(data[..., 1], coords, order=order)
        rgb[..., 2] = scipy.ndimage.map_coordinates(data[..., 2], coords, order=order)
        return DataAndMetadata.new_data_and_metadata(rgb,
                                                     dimensional_calibrations=data_and_metadata.dimensional_calibrations,
                                                     intensity_calibration=data_and_metadata.intensity_calibration)
    elif data_and_metadata.is_data_rgba:
        rgba: numpy.typing.NDArray[numpy.uint8] = numpy.zeros(tuple(data_and_metadata.dimensional_shape) + (4,), numpy.uint8)
        rgba[..., 0] = scipy.ndimage.map_coordinates(data[..., 0], coords, order=order)
        rgba[..., 1] = scipy.ndimage.map_coordinates(data[..., 1], coords, order=order)
        rgba[..., 2] = scipy.ndimage.map_coordinates(data[..., 2], coords, order=order)
        rgba[..., 3] = scipy.ndimage.map_coordinates(data[..., 3], coords, order=order)
        return DataAndMetadata.new_data_and_metadata(rgba,
                                                     dimensional_calibrations=data_and_metadata.dimensional_calibrations,
                                                     intensity_calibration=data_and_metadata.intensity_calibration)
    else:
        return DataAndMetadata.new_data_and_metadata(
            scipy.ndimage.map_coordinates(data, coords, order=order),
            dimensional_calibrations=data_and_metadata.dimensional_calibrations,
            intensity_calibration=data_and_metadata.intensity_calibration)


def calculate_coordinates_for_affine_transform(data_and_metadata_in: _DataAndMetadataLike, transformation_matrix: _ImageDataType) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    if data_and_metadata.is_data_rgb_type:
        coords_shape = data_and_metadata.data_shape[:-1]
    else:
        coords_shape = data_and_metadata.data_shape
    assert transformation_matrix.ndim == 2
    assert transformation_matrix.shape[0] == transformation_matrix.shape[1]
    assert transformation_matrix.shape[0] in {len(coords_shape), len(coords_shape) + 1}
    half_shape = (coords_shape[0] * 0.5, coords_shape[1] * 0.5)
    coords = numpy.mgrid[0:coords_shape[0], 0:coords_shape[1]].astype(float)  # type: ignore
    coords[0] -= half_shape[0] - 0.5
    coords[1] -= half_shape[1] - 0.5
    if transformation_matrix.shape[0] == len(coords_shape) + 1:
        coords = numpy.concatenate([numpy.ones((1,) + coords.shape[1:]), coords])  # type: ignore
    coords = coords[::-1, ...]
    transformed = numpy.einsum('ij,ikm', transformation_matrix, coords)
    transformed = transformed[::-1, ...]
    if transformation_matrix.shape[0] == len(coords_shape) + 1:
        transformed = transformed[1:, ...]
    transformed[0] += half_shape[0] - 0.5
    transformed[1] += half_shape[1] - 0.5
    result = [DataAndMetadata.new_data_and_metadata(transformed[0]),
              DataAndMetadata.new_data_and_metadata(transformed[1])]
    return result


def function_affine_transform(data_and_metadata_in: _DataAndMetadataLike, transformation_matrix: _ImageDataType, order: int = 1) -> DataAndMetadata.DataAndMetadata:
    coordinates = calculate_coordinates_for_affine_transform(data_and_metadata_in, transformation_matrix)
    return function_warp(data_and_metadata_in, coordinates, order=order)


def function_histogram(data_and_metadata_in: _DataAndMetadataLike, bins: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Resample: invalid data")

    histogram_data = numpy.histogram(data_and_metadata._data_ex, bins=bins)  # type: ignore
    min_x = data_and_metadata.intensity_calibration.convert_to_calibrated_value(histogram_data[1][0])
    max_x = data_and_metadata.intensity_calibration.convert_to_calibrated_value(histogram_data[1][-1])
    result_data: numpy.typing.NDArray[numpy.int32] = histogram_data[0].astype(numpy.int32)

    x_calibration = Calibration.Calibration(min_x, (max_x - min_x) / bins, data_and_metadata.intensity_calibration.units)

    return DataAndMetadata.new_data_and_metadata(result_data, dimensional_calibrations=[x_calibration])


def function_line_profile(data_and_metadata_in: _DataAndMetadataLike, vector: NormVectorType,
                          integration_width: float) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Line profile: invalid data")

    if not Image.is_data_2d(data_and_metadata.data):
        raise ValueError("Line profile: data must be 2D")

    assert round(integration_width) > 0  # leave this here for test_evaluation_error_recovers_gracefully

    # calculate grid of coordinates. returns n coordinate arrays for each row.
    # start and end are in data coordinates.
    # n is a positive integer, not zero
    def get_coordinates(start: _ImageDataType, end: _ImageDataType, n: int) -> typing.Tuple[_ImageDataType, _ImageDataType]:
        assert n > 0
        # n=1 => 0
        # n=2 => -0.5, 0.5
        # n=3 => -1, 0, 1
        # n=4 => -1.5, -0.5, 0.5, 1.5
        length_f = math.sqrt(math.pow(end[0] - start[0], 2) + math.pow(end[1] - start[1], 2))
        samples = int(math.floor(length_f))
        a = numpy.linspace(0, samples - 1, samples)  # along
        t = numpy.linspace(-(n - 1) * 0.5, (n - 1) * 0.5, round(n))  # transverse
        dy = (end[0] - start[0]) / samples
        dx = (end[1] - start[1]) / samples
        ix, iy = numpy.meshgrid(a, t)  # type: ignore
        yy = start[0] + dy * ix + dx * iy
        xx = start[1] + dx * ix - dy * iy
        return yy, xx

    # xx, yy = __coordinates(None, (4,4), (8,4), 3)

    data = data_and_metadata._data_ex
    shape = data.shape
    actual_integration_width = min(max(shape[0], shape[1]), round(integration_width))  # limit integration width to sensible value

    def calculate_data(data: _ImageDataType) -> _ImageDataType:
        if Image.is_data_rgb_type(data):
            data = Image.convert_to_grayscale(data, numpy.double)
        start, end = vector
        start_data: numpy.typing.NDArray[typing.Any] = numpy.array([int(shape[0] * start[0]), int(shape[1] * start[1])])
        end_data: numpy.typing.NDArray[typing.Any] = numpy.array([int(shape[0] * end[0]), int(shape[1] * end[1])])
        length = math.sqrt(math.pow(end_data[1] - start_data[1], 2) + math.pow(end_data[0] - start_data[0], 2))
        if length > 1.0:
            spline_order_lookup = {"nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3}
            method = "nearest"
            spline_order = spline_order_lookup[method]
            yy, xx = get_coordinates(start_data, end_data, actual_integration_width)
            samples = scipy.ndimage.map_coordinates(data, (yy, xx), order=spline_order)
            if len(samples.shape) > 1:
                return typing.cast(_ImageDataType, numpy.sum(samples, 0, dtype=data.dtype))
            else:
                return typing.cast(_ImageDataType, samples)
        else:
            return numpy.zeros((1,))

    dimensional_calibrations = data_and_metadata.dimensional_calibrations

    dimensional_calibrations = [Calibration.Calibration(0.0, dimensional_calibrations[1].scale, dimensional_calibrations[1].units)]

    intensity_calibration = copy.deepcopy(data_and_metadata.intensity_calibration)
    intensity_calibration.scale /= actual_integration_width

    return DataAndMetadata.new_data_and_metadata(calculate_data(data), intensity_calibration=intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations)


def function_make_point(y: float, x: float) -> NormPointType:
    return y, x


def function_make_size(height: float, width: float) -> NormSizeType:
    return height, width


def function_make_vector(start: NormPointType, end: NormPointType) -> NormVectorType:
    return start, end


def function_make_rectangle_origin_size(origin: NormPointType, size: NormSizeType) -> NormRectangleType:
    return typing.cast(NormRectangleType, tuple(Geometry.FloatRect(origin, size)))


def function_make_rectangle_center_size(center: NormPointType, size: NormSizeType) -> NormRectangleType:
    return typing.cast(NormRectangleType, tuple(Geometry.FloatRect.from_center_and_size(center, size)))


def function_make_interval(start: float, end: float) -> NormIntervalType:
    return start, end


def function_make_shape(*args: typing.Any) -> DataAndMetadata.ShapeType:
    return tuple([int(arg) for arg in args])


# generic functions

def function_array(array_fn: typing.Callable[..., typing.Any], data_and_metadata_in: _DataAndMetadataLike, *args: typing.Any, **kwargs: typing.Any) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    data = array_fn(data_and_metadata.data, *args, **kwargs)
    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_scalar(op: typing.Callable[[_ImageDataType], DataAndMetadata._ScalarDataType], data_and_metadata_in: _DataAndMetadataLike) -> DataAndMetadata.ScalarAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)
    return DataAndMetadata.ScalarAndMetadata(lambda: op(data_and_metadata._data_ex), data_and_metadata.intensity_calibration)


def function_element_data_no_copy(data_and_metadata: DataAndMetadata.DataAndMetadata,
                                  sequence_index: int = 0,
                                  collection_index: typing.Optional[DataAndMetadata.PositionType] = None,
                                  slice_center: int = 0,
                                  slice_width: int = 1, *,
                                  use_slice: bool = True,
                                  flag16: bool = True) -> typing.Tuple[typing.Optional[DataAndMetadata.DataAndMetadata], bool]:
    # extract an element (2d or 1d data element) from data and metadata using the indexes and slices.
    # flag16 is for backwards compatibility with 0.15.2 and earlier. new callers should set it to False.
    result: typing.Optional[DataAndMetadata.DataAndMetadata] = data_and_metadata
    dimensional_shape = data_and_metadata.dimensional_shape
    modified = False
    next_dimension = 0
    collection_dimension_count = data_and_metadata.collection_dimension_count
    collection_dimension_shape = data_and_metadata.collection_dimension_shape
    datum_dimension_count = data_and_metadata.datum_dimension_count
    treat_as_image = flag16 and collection_dimension_count == 1 and datum_dimension_count == 1 and collection_dimension_shape[0] <= 16
    use_slice_sum = use_slice and collection_dimension_count == 2 and datum_dimension_count == 1
    if data_and_metadata.is_sequence and data_and_metadata.is_collection and not treat_as_image and not use_slice_sum:
        # optimize the case of a sequence + collection that is not treated as an image and doesn't sum slices (as a pick would do).
        # this is a typical 5D image case.
        assert collection_index is not None
        sequence_index = min(max(sequence_index, 0), dimensional_shape[next_dimension])
        data_slice = (sequence_index,) + tuple(collection_index[0:collection_dimension_count]) + (...,)
        result = DataAndMetadata.function_data_slice(data_and_metadata, DataAndMetadata.key_to_list(tuple(data_slice)))
        modified = True
    else:
        if data_and_metadata.is_sequence:
            # next dimension is treated as a sequence index, which may be time or just a sequence index
            sequence_index = min(max(sequence_index, 0), dimensional_shape[next_dimension])
            data_slice = typing.cast(DataAndMetadata._SliceKeyType, (sequence_index, ...))
            result = DataAndMetadata.function_data_slice(data_and_metadata, DataAndMetadata.key_to_list(data_slice))
            modified = True
            next_dimension += 1
        if result and result.is_collection:
            assert collection_index is not None
            # next dimensions are treated as collection indexes.
            if treat_as_image:
                pass  # this is a special case to display a few rows all at once. once true multi-data displays are available, remove this
            elif use_slice_sum:
                result = function_slice_sum(result, slice_center, slice_width)
                modified = True
            else:  # default, "pick"
                collection_slice = tuple(collection_index[0:collection_dimension_count]) + (...,)
                result = DataAndMetadata.function_data_slice(result, DataAndMetadata.key_to_list(collection_slice))
                modified = True
            next_dimension += collection_dimension_count + datum_dimension_count
    if result and functools.reduce(operator.mul, result.dimensional_shape) == 0:
        result = None
    return result, modified


def function_scalar_data_no_copy(data_and_metadata: DataAndMetadata.DataAndMetadata, complex_display_type: typing.Optional[str] = None, *, _modified: bool = False) -> typing.Tuple[typing.Optional[DataAndMetadata.DataAndMetadata], bool]:
    modified = _modified
    result: typing.Optional[DataAndMetadata.DataAndMetadata] = data_and_metadata
    if result and result.is_data_complex_type:
        if complex_display_type == "real":
            result = function_array(numpy.real, result)
        elif complex_display_type == "imaginary":
            result = function_array(numpy.imag, result)
        elif complex_display_type == "absolute":
            result = function_array(numpy.absolute, result)
        else:  # default, log-absolute
            def log_absolute(d: _ImageDataType) -> _ImageDataType:
                return typing.cast(_ImageDataType, numpy.log(numpy.abs(d).astype(numpy.float64) + numpy.nextafter(0, 1)))
            result = function_array(log_absolute, result)
        modified = True
    if result and functools.reduce(operator.mul, result.dimensional_shape) == 0:
        result = None
    return result, modified


def function_display_data_no_copy(data_and_metadata: DataAndMetadata.DataAndMetadata, sequence_index: int = 0,
                                  collection_index: typing.Optional[DataAndMetadata.PositionType] = None,
                                  slice_center: int = 0, slice_width: int = 1,
                                  complex_display_type: typing.Optional[str] = None) -> typing.Tuple[typing.Optional[DataAndMetadata.DataAndMetadata], bool]:
    result: typing.Optional[DataAndMetadata.DataAndMetadata]
    result, modified = function_element_data_no_copy(data_and_metadata, sequence_index, collection_index, slice_center, slice_width)
    if result:
        result, modified = function_scalar_data_no_copy(result, _modified=modified)
    return result, modified


def function_display_data(data_and_metadata: DataAndMetadata.DataAndMetadata, sequence_index: int = 0,
                          collection_index: typing.Optional[DataAndMetadata.PositionType] = None,
                          slice_center: int = 0, slice_width: int = 1,
                          complex_display_type: typing.Optional[str] = None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    result, modified = function_display_data_no_copy(data_and_metadata, sequence_index, collection_index, slice_center, slice_width, complex_display_type)
    return copy.deepcopy(result) if result and not modified else result


def function_display_rgba(data_and_metadata: DataAndMetadata.DataAndMetadata,
                          display_range: typing.Optional[typing.Tuple[float, float]] = None,
                          color_table: typing.Optional[_ImageDataType] = None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    data_2d = data_and_metadata._data_ex
    if Image.is_data_1d(data_2d):
        data_2d = data_2d.reshape(1, *data_2d.shape)
    if not Image.is_data_rgb_type(data_2d):
        assert display_range is not None
    assert len(Image.dimensional_shape_from_data(data_2d) or ()) == 2
    rgba_data = Image.create_rgba_image_from_array(data_2d, display_limits=display_range, lookup=color_table)
    return DataAndMetadata.new_data_and_metadata(rgba_data)


def function_extract_datum(data_and_metadata: DataAndMetadata.DataAndMetadata, sequence_index: int = 0,
                           collection_index: typing.Optional[DataAndMetadata.PositionType] = None,) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    dimensional_shape = data_and_metadata.dimensional_shape
    next_dimension = 0
    if data_and_metadata.is_sequence:
        # next dimension is treated as a sequence index, which may be time or just a sequence index
        sequence_index = min(max(sequence_index, 0), dimensional_shape[next_dimension])
        data_slice = typing.cast(DataAndMetadata._SliceKeyType, (sequence_index, ...))
        data_and_metadata = DataAndMetadata.function_data_slice(data_and_metadata, DataAndMetadata.key_to_list(data_slice))
        next_dimension += 1
    if data_and_metadata and data_and_metadata.is_collection:
        collection_dimension_count = data_and_metadata.collection_dimension_count
        assert collection_dimension_count is not None
        assert collection_index is not None
        # next dimensions are treated as collection indexes.
        collection_slice = tuple(collection_index[0:collection_dimension_count]) + (...,)
        data_and_metadata = DataAndMetadata.function_data_slice(data_and_metadata, DataAndMetadata.key_to_list(collection_slice))
    return data_and_metadata


def function_convert_to_scalar(data_and_metadata: DataAndMetadata.DataAndMetadata, complex_display_type: typing.Optional[str] = None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    result, modified = function_scalar_data_no_copy(data_and_metadata, complex_display_type)
    return result


def calibrated_subtract_spectrum(data1: DataAndMetadata.DataAndMetadata, data2: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
    assert data1.is_datum_1d
    assert data2.is_datum_1d
    assert data1.intensity_calibration == data2.intensity_calibration
    calibration1 = data1.datum_dimensional_calibrations[0]
    calibration2 = data2.datum_dimensional_calibrations[0]
    assert calibration1.units == calibration2.units
    assert calibration1.scale == calibration2.scale
    start1 = calibration1.convert_to_calibrated_value(0)
    end1 = calibration1.convert_to_calibrated_value(data1.datum_dimension_shape[0])
    start2 = calibration2.convert_to_calibrated_value(0)
    end2 = calibration2.convert_to_calibrated_value(data2.datum_dimension_shape[0])
    assert (start2 <= start1 <= end2) or (start2 <= end1 <= end2) or (start1 <= start2 <= end1) or (start1 <= end2 <= end1)
    start = max(start1, start2)
    end = min(end1, end2)
    start_index1 = round(calibration1.convert_from_calibrated_value(start))
    end_index1 = round(calibration1.convert_from_calibrated_value(end))
    start_index2 = round(calibration2.convert_from_calibrated_value(start))
    end_index2 = round(calibration2.convert_from_calibrated_value(end))
    return data1[..., start_index1:end_index1] - data2[..., start_index2:end_index2]


def _iso_data(hist: _ImageDataType, bins: _ImageDataType) -> float:
    """
    Implementation of the IsoData method: Ridler, TW & Calvard, S (1978),
    "Picture thresholding using an iterative selection method", IEEE Transactions on Systems,
    Man and Cybernetics 8: 630-632
    """
    min_value = bins[0]
    bins = bins + (bins[1] - bins[0]) / 2 - min_value
    indices = numpy.arange(1, len(hist)+1)
    h = numpy.cumsum(hist*indices)/numpy.cumsum(hist)
    l = (numpy.cumsum((hist*indices)[::-1])/numpy.cumsum(hist[::-1]))[::-1]
    g = (h + l) /2
    try:
        return float(bins[numpy.nonzero(indices > g)[0][0]] + min_value) # condition is tested in order of the array, so entry 0 is the first
                                                                         # index where indices > g
    except IndexError:
        return float(min_value)


def _yen(hist: _ImageDataType, bins: _ImageDataType) -> float:
    """
    Implementation of Yen's auto threshold method: Yen JC, Chang FJ, Chang S (1995), "A New Criterion for Automatic
    Multilevel Thresholding", IEEE Trans. on Image Processing 4 (3): 370-378 and Sezgin, M & Sankur, B (2004),
    "Survey over Image Thresholding Techniques and Quantitative Performance Evaluation", Journal of Electronic
    Imaging 13(1): 146-165
    """
    min_value = bins[0]
    bins = bins + (bins[1] - bins[0]) / 2 - min_value
    norm_hist = hist / numpy.sum(hist)
    p1 = numpy.cumsum(norm_hist)
    p1_sq = numpy.cumsum(norm_hist**2)
    p2_sq = numpy.cumsum(norm_hist[::-1]**2)[::-1]
    first_part = p1_sq * p2_sq
    second_part: _ImageDataType = p1 * (1 - p1)
    first_part = numpy.where(first_part > 0, first_part, numpy.amin(first_part[first_part>0]))
    second_part = numpy.where(second_part > 0, second_part, numpy.amin(second_part[second_part>0]))
    crit = -1.0 * numpy.log(first_part) + 2.0 * numpy.log(second_part)
    return float(bins[numpy.argmax(crit)] + min_value)


def _kittler(hist: _ImageDataType, bins: _ImageDataType) -> float:
    """
    Implementation of Kittler's auto threshold method: M. I. Sezan, "A peak detection algorithm and its application
    to histogram-based image data reduction", Graph. Models Image Process. 29, 47–59, 1985 and Sezgin, M & Sankur,
    B (2004), "Survey over Image Thresholding Techniques and Quantitative Performance Evaluation", Journal of
    Electronic Imaging 13(1): 146-165
    """
    min_value = bins[0]
    bins = bins + (bins[1] - bins[0]) / 2 - min_value
    p_g = hist/numpy.sum(hist)
    m_f = numpy.cumsum(bins[:-1]*p_g)
    P_f = numpy.cumsum(p_g)
    P_b = numpy.cumsum(p_g[::-1])[::-1]
    var_f = numpy.cumsum((bins[:-1] - m_f)**2 * p_g)
    var_b = numpy.cumsum(((bins[:-1] - m_f)**2 * p_g)[::-1])[::-1]
    crit = P_f * numpy.log(numpy.sqrt(var_f)) + P_b * numpy.log(numpy.sqrt(var_b)) - P_f * numpy.log(P_f) - P_b * numpy.log(P_b)
    return float(bins[numpy.argmin(crit)] + min_value)


def auto_threshold(data_and_metadata_in: _DataAndMetadataLike, *, auto_threshold_method: str='average', number_bins: int=1000, **kwargs: typing.Any) -> float:
    """
    Finds a good threshold value for `image` by means of `auto_threshold_method`.

    Currently, three different methods are supported:
        'iso_data'
            Implementation of the IsoData Method. See :py:func:`_iso_data` for more details.

        'yen'
            Implementation of Yen's method. See :py:func:`_yen` for more details.

        'kittler'
            Implementation of Kittler's method. See :py:func:`_kittler` for more details.

        'average'
            Returns the weighted average of the results of 'iso_data' and 'kittler'. 'iso_data' typically results in a
            rather high threshold so that dark foreground objects might be marked as background. 'kittler' on the
            other hand ususally finds a very low threshold so that bright background objects can be marked as
            foreground. Using (2 * kittler + iso_data) / 3 has shown good results. Since these are very
            fast calculations (~100 us), the performance loss is negligible.
    """
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if not Image.is_data_valid(data_and_metadata.data):
        raise ValueError("Auto threshold: invalid data")

    data = data_and_metadata._data_ex

    hist, bins = numpy.histogram(data, bins=number_bins) # type: ignore
    if auto_threshold_method == 'average':
        return (_kittler(hist, bins) * 2.0 + _iso_data(hist, bins)) / 3.0
    if auto_threshold_method == 'yen':
        return _yen(hist, bins)
    if auto_threshold_method == 'iso_data':
        return _iso_data(hist, bins)
    if auto_threshold_method == 'kittler':
        return _kittler(hist, bins)
    raise ValueError(f'Unsupported auto threshold method {auto_threshold_method}.')
