# standard libraries
import typing

# third party libraries
import numpy

# local libraries
from nion.data import DataAndMetadata
from nion.data import Image


_DataAndMetadataLike = DataAndMetadata._DataAndMetadataLike
_DataAndMetadataIndeterminateSizeLike = DataAndMetadata._DataAndMetadataIndeterminateSizeLike
_ImageDataType = Image._ImageDataType


def function_rgb_channel(data_and_metadata_in: _DataAndMetadataLike, channel: int) -> DataAndMetadata.DataAndMetadata:
    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    if channel < 0 or channel > 3:
        raise ValueError("RGB channel: invalid channel.")

    data = data_and_metadata.data
    if not Image.is_data_valid(data):
        raise ValueError("RGB channel: invalid data.")

    if not data_and_metadata.is_data_rgb_type:
        raise ValueError("RGB channel: data is not RGB type.")

    assert data is not None

    channel_data: _ImageDataType

    if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
        if channel == 3:
            channel_data = numpy.ones(data.shape, int)
        else:
            channel_data = data[..., channel].astype(int)
    elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
        channel_data = data[..., channel].astype(int)
    else:
        raise ValueError("RGB channel: unable to extract channel.")

    return DataAndMetadata.new_data_and_metadata(channel_data,
                                                 intensity_calibration=data_and_metadata.intensity_calibration,
                                                 dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_rgb_linear_combine(data_and_metadata_in: _DataAndMetadataLike, red_weight: float, green_weight: float,
                                blue_weight: float) -> DataAndMetadata.DataAndMetadata:

    data_and_metadata = DataAndMetadata.promote_ndarray(data_and_metadata_in)

    data = data_and_metadata.data
    if not Image.is_data_valid(data):
        raise ValueError("RGB linear combine: invalid data.")

    if not data_and_metadata.is_data_rgb_type:
        raise ValueError("RGB linear combine: data is not RGB type.")

    assert data is not None

    combined_data: _ImageDataType

    if Image.is_shape_and_dtype_rgb(data.shape, data.dtype):
        combined_data = numpy.sum(data[..., :] * (blue_weight, green_weight, red_weight), 2)
    elif Image.is_shape_and_dtype_rgba(data.shape, data.dtype):
        combined_data = numpy.sum(data[..., :] * (blue_weight, green_weight, red_weight, 0.0), 2)
    else:
        raise ValueError("RGB channel: unable to extract channel.")

    return DataAndMetadata.new_data_and_metadata(combined_data, intensity_calibration=data_and_metadata.intensity_calibration, dimensional_calibrations=data_and_metadata.dimensional_calibrations)


def function_rgb(red_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike,
                 green_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike,
                 blue_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike) -> DataAndMetadata.DataAndMetadata:

    red_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(red_data_and_metadata_in)
    green_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(green_data_and_metadata_in)
    blue_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(blue_data_and_metadata_in)

    shape = DataAndMetadata.determine_shape(red_data_and_metadata_c, green_data_and_metadata_c, blue_data_and_metadata_c)

    if shape is None:
        raise ValueError("RGB: data shapes do not match or are indeterminate")

    red_data_and_metadata = DataAndMetadata.promote_constant(red_data_and_metadata_c, shape)
    green_data_and_metadata = DataAndMetadata.promote_constant(green_data_and_metadata_c, shape)
    blue_data_and_metadata = DataAndMetadata.promote_constant(blue_data_and_metadata_c, shape)

    channels = (blue_data_and_metadata, green_data_and_metadata, red_data_and_metadata)

    if any([not Image.is_data_valid(data_and_metadata.data) for data_and_metadata in channels]):
        raise ValueError("RGB: invalid data")

    rgb_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(shape + (3,), numpy.uint8)
    for channel_index, channel in enumerate(channels):
        data = channel._data_ex
        if data.dtype.kind in 'iu':
            rgb_image[..., channel_index] = numpy.clip(data, 0, 255)
        elif data.dtype.kind in 'f':
            rgb_image[..., channel_index] = numpy.clip(numpy.multiply(data, 255), 0, 255)

    return DataAndMetadata.new_data_and_metadata(rgb_image,
                                                 intensity_calibration=red_data_and_metadata.intensity_calibration,
                                                 dimensional_calibrations=red_data_and_metadata.dimensional_calibrations)


def function_rgba(red_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike,
                  green_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike,
                  blue_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike,
                  alpha_data_and_metadata_in: _DataAndMetadataIndeterminateSizeLike) -> DataAndMetadata.DataAndMetadata:

    red_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(red_data_and_metadata_in)
    green_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(green_data_and_metadata_in)
    blue_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(blue_data_and_metadata_in)
    alpha_data_and_metadata_c = DataAndMetadata.promote_indeterminate_array(alpha_data_and_metadata_in)

    shape = DataAndMetadata.determine_shape(red_data_and_metadata_c, green_data_and_metadata_c, blue_data_and_metadata_c)

    if shape is None:
        raise ValueError("RGBA: data shapes do not match or are indeterminate")

    red_data_and_metadata = DataAndMetadata.promote_constant(red_data_and_metadata_c, shape)
    green_data_and_metadata = DataAndMetadata.promote_constant(green_data_and_metadata_c, shape)
    blue_data_and_metadata = DataAndMetadata.promote_constant(blue_data_and_metadata_c, shape)
    alpha_data_and_metadata = DataAndMetadata.promote_constant(alpha_data_and_metadata_c, shape)

    channels = (blue_data_and_metadata, green_data_and_metadata, red_data_and_metadata, alpha_data_and_metadata)

    if any([not Image.is_data_valid(data_and_metadata.data) for data_and_metadata in channels]):
        raise ValueError("RGB: invalid data")

    rgba_image: numpy.typing.NDArray[numpy.uint8] = numpy.empty(shape + (4,), numpy.uint8)
    for channel_index, channel in enumerate(channels):
        data = channel._data_ex
        if data.dtype.kind in 'iu':
            rgba_image[..., channel_index] = numpy.clip(data, 0, 255)
        elif data.dtype.kind in 'f':
            rgba_image[..., channel_index] = numpy.clip(numpy.multiply(data, 255), 0, 255)

    return DataAndMetadata.new_data_and_metadata(rgba_image,
                                                 intensity_calibration=red_data_and_metadata.intensity_calibration,
                                                 dimensional_calibrations=red_data_and_metadata.dimensional_calibrations)
