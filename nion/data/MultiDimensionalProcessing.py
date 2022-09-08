import copy
import gettext
import logging
import multiprocessing
import numpy
import scipy.ndimage
import threading
import typing

from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata


try:
    import mkl
except ModuleNotFoundError:
    _has_mkl = False
else:
    _has_mkl = True


_ = gettext.gettext


_ImageDataType = DataAndMetadata._ImageDataType


def function_integrate_along_axis(input_xdata: DataAndMetadata.DataAndMetadata,
                                  integration_axes: typing.Tuple[int, ...],
                                  integration_mask: typing.Optional[_ImageDataType] = None) -> DataAndMetadata.DataAndMetadata:

    navigation_shape = []
    navigation_axis_indices = []
    for i in range(len(input_xdata.data_shape)):
        if not i in integration_axes:
            navigation_shape.append(input_xdata.data_shape[i])
            navigation_axis_indices.append(i)

    data_str = ''
    mask_str = ''
    navigation_str = ''
    for i in range(len(input_xdata.data_shape)):
        char = chr(i + 97)
        data_str += char
        if i in integration_axes:
            mask_str += char
        else:
            navigation_str += char

    integration_axis_shape = tuple((input_xdata.data_shape[i] for i in integration_axes))
    # chr(97) == 'a' so we get letters in alphabetic order here (a, b, c, d, ...)
    sum_str = ''.join([chr(i + 97) for i in range(len(integration_axis_shape))])
    operands = [input_xdata.data]
    if integration_mask is not None:
        operands.append(integration_mask)
        sum_str = data_str + ',' + mask_str
    else:
        sum_str = data_str + '->' + navigation_str
    result_data = numpy.einsum(sum_str, *operands)

    result_dimensional_calibrations = []
    for i in range(len(input_xdata.data_shape)):
        if not i in integration_axes:
            result_dimensional_calibrations.append(input_xdata.dimensional_calibrations[i])

    result_data = numpy.atleast_1d(result_data)

    if len(result_dimensional_calibrations) != len(result_data.shape):
        result_dimensional_calibrations = [Calibration.Calibration() for _ in range(len(result_data.shape))]

    is_sequence = input_xdata.is_sequence
    collection_dimension_count = input_xdata.collection_dimension_count
    datum_dimension_count = input_xdata.datum_dimension_count
    for i in integration_axes:
        if i == input_xdata.sequence_dimension_index:
            is_sequence = False
        elif i in input_xdata.collection_dimension_indexes:
            collection_dimension_count -= 1
        elif i in input_xdata.datum_dimension_indexes:
            datum_dimension_count -= 1
    # 0-D data is not allowed in Swift, so we need to make the collection or the sequence axis the data axis
    # Use the collection axis preferably and only when the data is not a collection use the sequence axis
    # If the user integrated a single image we get a single number. We also make this 1D data to prevent errors
    if datum_dimension_count == 0:
        if collection_dimension_count > 0:
            datum_dimension_count = collection_dimension_count
            collection_dimension_count = 0
        elif is_sequence:
            datum_dimension_count = 1
            is_sequence = False
        else:
            # If we end up here we have reduced our input data to a single number. In this case make the data 1D in
            # the data dimensions
            datum_dimension_count = 1

    result_data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

    return DataAndMetadata.new_data_and_metadata(result_data,
                                                 intensity_calibration=input_xdata.intensity_calibration,
                                                 dimensional_calibrations=result_dimensional_calibrations,
                                                 data_descriptor=result_data_descriptor)


def ellipse_radius(polar_angle: typing.Union[float, _ImageDataType], a: float, b: float, rotation: float) -> typing.Union[float, _ImageDataType]:
    """
    Returns the radius of a point lying on an ellipse with the given parameters. The ellipse is described in polar
    coordinates here, which makes it easy to incorporate a rotation.
    Parameters
    -----------
    polar_angle : float or numpy.ndarray
                  Polar angle of a point to which the corresponding radius should be calculated (rad).
    a : float
        Length of the major half-axis of the ellipse.
    b : float
        Length of the minor half-axis of the ellipse.
    rotation : Rotation of the ellipse with respect to the x-axis (rad). Counter-clockwise is positive.
    Returns
    --------
    radius : float or numpy.ndarray
             Radius of a point lying on an ellipse with the given parameters.
    """

    return a*b/numpy.sqrt((b*numpy.cos(polar_angle+rotation))**2+(a*numpy.sin(polar_angle+rotation))**2)


def draw_ellipse(image: _ImageDataType, ellipse: typing.Tuple[float, float, float, float, float], *,
                 color: typing.Any=1.0) -> None:
    """
    Draws an ellipse on a 2D-array.
    Parameters
    ----------
    image : array
            The array on which the ellipse will be drawn. Note that the data will be modified in place.
    ellipse : tuple
              A tuple describing an ellipse. The values must be (in this order):
              [0] The y-coordinate of the center.
              [1] The x-coordinate of the center.
              [2] The length of the major half-axis
              [3] The length of the minor half-axis
              [4] The rotation of the ellipse in rad.
    color : optional
            The color to which the pixels inside the given ellipse will be set. Note that `color` will be cast to the
            type of `image` automatically. If this is not possible, an exception will be raised. The default is 1.0.
    Returns
    --------
    None
    """
    shape = image.shape
    assert len(shape) == 2, 'Can only draw an ellipse on a 2D-array.'

    top = max(int(ellipse[0] - ellipse[2]), 0)
    left = max(int(ellipse[1] - ellipse[2]), 0)
    bottom = min(int(ellipse[0] + ellipse[2]) + 1, shape[0])
    right = min(int(ellipse[1] + ellipse[2]) + 1, shape[1])
    coords = numpy.mgrid[top - ellipse[0]:bottom - ellipse[0], left - ellipse[1]:right - ellipse[1]] # type: ignore # Not working yet, see https://github.com/python/mypy/issues/2410
    radii = numpy.sqrt(numpy.sum(coords**2, axis=0))
    polar_angles = numpy.arctan2(coords[0], coords[1])
    ellipse_radii = ellipse_radius(polar_angles, *ellipse[2:])
    image[top:bottom, left:right][radii<ellipse_radii] = color


def _make_mask(max_shift: int, origin: typing.Tuple[int, ...], data_shape: typing.Tuple[int, ...]) -> _ImageDataType:
    mask = numpy.zeros(data_shape, dtype=bool)
    if len(data_shape) == 2:
        half_shape = (data_shape[0] // 2, data_shape[1] // 2)
        offset = (origin[0] + half_shape[0], origin[1] + half_shape[1])
        draw_ellipse(mask, offset + (max_shift, max_shift, 0))
    elif len(data_shape) == 1:
        # Use different name to make typing happy
        half_shape_1d = data_shape[0] // 2
        mask[max(0, origin[0] + half_shape_1d - max_shift):min(data_shape[0], origin[0] + half_shape_1d + max_shift + 1)] = True
    else:
        raise ValueError('Only data of 1 or 2 dimensions is allowed.')
    return mask


def function_measure_multi_dimensional_shifts(xdata: DataAndMetadata.DataAndMetadata,
                                              shift_axes: typing.Tuple[int, ...],
                                              reference_index: typing.Optional[int] = None,
                                              bounds: typing.Optional[typing.Union[Core.NormIntervalType, Core.NormRectangleType]] = None,
                                              max_shift: typing.Optional[int] = None,
                                              origin: typing.Optional[typing.Tuple[int, ...]] = None) -> DataAndMetadata.DataAndMetadata:
    """
    "max_shift" defines the maximum allowed template shift in pixels. "max_shift" is calculated around "origin", which
    is the offset from the center of the image.
    """

    iteration_shape: typing.Tuple[int, ...] = tuple()
    dimensional_calibrations = list()
    intensity_calibration = None
    for i in range(len(xdata.data_shape)):
        if not i in shift_axes:
            iteration_shape += (xdata.data_shape[i],)
            dimensional_calibrations.append(xdata.dimensional_calibrations[i])
        else:
            intensity_calibration = Calibration.Calibration(scale=xdata.dimensional_calibrations[i].scale, units=xdata.dimensional_calibrations[i].units)

    shape: typing.Tuple[int, ...]
    register_slice: typing.Union[slice, typing.Tuple[slice, slice]]

    # If we are shifting along more than one axis the shifts will have an extra axis to hold the shifts for these axes.
    if len(shift_axes) > 1:
        shifts_ndim = 1
    else:
        shifts_ndim = 0

    if shifts_ndim == 1:
        result_shape = iteration_shape + (2,)
        dimensional_calibrations.append(Calibration.Calibration())
        if bounds is not None:
            assert numpy.ndim(bounds) == 2
            # Use an new variable for the 2D case to make typing happy.
            bounds_2d = typing.cast(typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]], bounds)
            shape = (xdata.data_shape[shift_axes[0]], xdata.data_shape[shift_axes[1]])
            register_slice = (slice(max(0, int(round(bounds_2d[0][0] * shape[0]))), min(int(round((bounds_2d[0][0] + bounds_2d[1][0]) * shape[0])), shape[0])),
                              slice(max(0, int(round(bounds_2d[0][1] * shape[1]))), min(int(round((bounds_2d[0][1] + bounds_2d[1][1]) * shape[1])), shape[1])))
        else:
            register_slice = (slice(0, None), slice(0, None))
    else:
        result_shape = iteration_shape + (1,)
        if bounds is not None:
            assert numpy.ndim(bounds) == 1
            # Also needed for typing
            bounds_1d = typing.cast(typing.Tuple[float, float], bounds)
            shape = (xdata.data_shape[shift_axes[0]],)
            register_slice = slice(max(0, int(round(bounds_1d[0] * shape[0]))), min(int(round(bounds_1d[1] * shape[0])), shape[0]))
        else:
            register_slice = slice(0, None)

    reference_data = None
    if reference_index is not None:
        coords = numpy.unravel_index(reference_index, iteration_shape)
        data_coords = coords[:shift_axes[0]] + (...,) + coords[shift_axes[0]:]
        reference_data = xdata.data[data_coords]

    mask = None
    # If we measure shifts relative to the last frame, we can always use a mask that is centered around the input origin
    if max_shift is not None and reference_index is None:
        coords = numpy.unravel_index(0, iteration_shape)
        data_coords = coords[:shift_axes[0]] + (...,) + coords[shift_axes[0]:]
        data_shape = xdata.data[data_coords][register_slice].shape
        if origin is None:
            origin = tuple([0] * len(data_shape))
        mask = _make_mask(max_shift, origin, data_shape)

    shifts = numpy.zeros(result_shape, dtype=numpy.float32)
    start_index = 0 if reference_index is not None else 1
    navigation_len = int(numpy.prod(iteration_shape, dtype=numpy.int64))
    num_cpus = 8
    try:
        num_cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        logging.warning('Could not determine the number of CPU cores. Defaulting to 8.')
    finally:
        # Use a little bit more than half the CPU cores, but not more than 20 because then we actually get a slowdown
        # because of our HDF5 storage handler not being able to grant parallel access to the data
        num_threads = min(int(round(num_cpus * 0.6)), 20)

    # Unfortunately multi-threading cannot be used when we cross-correlate with the first frame and max_shift
    # is not None because in order to create the mask for each frame we need the shift from the previous frame.
    # This does not work for the "block boundaries" where the data is split between the frames, so we have to
    # do a single-threaded calculation in this case.
    # If the shifts reference is not the first or last frame in the sequence, we can actually use two threads, both
    # starting from reference_index and iterating away from it.
    if max_shift is not None and reference_index is not None:
        if reference_index == 0 or reference_index == navigation_len - 1:
            sections = [start_index, navigation_len]
        else:
            sections = [start_index, reference_index, navigation_len]
    else:
        sections = list(range(start_index, navigation_len, max(1, navigation_len//num_threads)))
        sections.append(navigation_len)
    barrier = threading.Barrier(len(sections))

    def run_on_thread(range_: range) -> None:
        if _has_mkl:
            mkl.set_num_threads_local(1)
        local_mask = mask
        local_reference_data = typing.cast(_ImageDataType, reference_data)
        try:
            for i in range_:
                coords = numpy.unravel_index(i, iteration_shape)
                data_coords = coords[:shift_axes[0]] + (...,) + coords[shift_axes[0]:]
                if reference_index is None:
                    coords_ref = numpy.unravel_index(i - range_.step, iteration_shape)
                    data_coords_ref = coords_ref[:shift_axes[0]] + (...,) + coords_ref[shift_axes[0]:]
                    local_reference_data = xdata.data[data_coords_ref]
                elif max_shift is not None and i != range_.start:
                    last_coords = numpy.unravel_index(i - range_.step, iteration_shape)
                    last_shift = shifts[last_coords]
                    data_shape = local_reference_data[register_slice].shape
                    # Use a local copy of origin here to avoid threading issues.
                    local_origin = origin
                    if local_origin is None:
                        local_origin = tuple([0] * len(data_shape))
                    if len(data_shape) == 2:
                        local_mask = _make_mask(max_shift, (local_origin[0] + round(last_shift[0]), local_origin[1] + round(last_shift[1])), data_shape)
                    else:
                        local_mask = _make_mask(max_shift, (local_origin[0] + round(last_shift[0]),), data_shape)
                shifts[coords] = Core.function_register_template(local_reference_data[register_slice], xdata.data[data_coords][register_slice], ccorr_mask=local_mask)[1]
        finally:
            barrier.wait()

    if max_shift is not None and reference_index is not None:
        # Set up the threads for the case with max_shift and reference index: As explained above, we need a special
        # setup because the result relies on the previous shift.
        if len(sections) == 2:
            if reference_index == 0:
                threading.Thread(target=run_on_thread, args=(range(sections[0], sections[1]),)).start()
            # Reference index is the last frame, so go backwards from there
            else:
                threading.Thread(target=run_on_thread, args=(range(sections[1] - 1, sections[0] - 1, -1),)).start()
        else:
            # If the reference index is somewhere inside the sequence, we can use two threads, one going from
            # reference_index to 0 (backwards) and one gaing from reference_index to the end.
            threading.Thread(target=run_on_thread, args=(range(sections[1], sections[0] - 1, -1),)).start()
            threading.Thread(target=run_on_thread, args=(range(sections[1], sections[2]),)).start()
    else:
        for i in range(len(sections) - 1):
            threading.Thread(target=run_on_thread, args=(range(sections[i], sections[i+1]),)).start()
    barrier.wait()

    # For debugging it is helpful to run a non-threaded version of the code. Comment out the 3 lines above and uncomment
    # the line below to do so. You also need to comment out "barrier.wait()" in the function running on the thread.
    # run_on_thread(range(start_index, navigation_len))

    shifts = numpy.squeeze(shifts)

    if reference_index is None:
        if len(iteration_shape) == 2:
            shifts = numpy.cumsum(shifts, axis=1)
        shifts = numpy.cumsum(shifts, axis=0)

    return DataAndMetadata.new_data_and_metadata(shifts,
                                                 intensity_calibration=intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations)


def function_apply_multi_dimensional_shifts(xdata: DataAndMetadata.DataAndMetadata,
                                            shifts: _ImageDataType,
                                            shift_axes: typing.Tuple[int, ...],
                                            out: typing.Optional[DataAndMetadata.DataAndMetadata] = None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:

    # Find the axes that we do not want to shift (== iteration shape)
    iteration_shape: typing.Tuple[int, ...] = tuple()
    iteration_shape_offset = 0
    for i in range(len(xdata.data_shape)):
        if not i in shift_axes:
            iteration_shape += (xdata.data_shape[i],)
        elif len(iteration_shape) == 0:
            iteration_shape_offset += 1
    # If we are shifting along more than one axis the shifts will have an extra axis to hold the shifts for these axes.
    # For finding matching iteration axis we only consider the iteration shape though, so we need to remove this last
    # axis.
    if len(shift_axes) > 1:
        shifts_shape = shifts.shape[:-1]
    else:
        shifts_shape = shifts.shape
    # Now we need to find matching axes between the iteration shape and the provided shifts. We can then iterate over
    # these matching axis and apply the shifts.
    for i in range(len(iteration_shape) - len(shifts_shape) + 1):
        if iteration_shape[i:i+len(shifts_shape)] == shifts_shape:
            shifts_start_axis = i
            shifts_end_axis = i + len(shifts_shape)
            break
    else:
        raise ValueError("Did not find any axis matching the shifts shape.")

    # Now drop all iteration axes after the last shift axis. This will greatly improve speed because we don't have
    # to iterate and shift each individual element but can work in larger sub-arrays. It will also be beneficial for
    # working with chunked hdf5 files because we usually have our chunks along the last axes.
    squeezed_iteration_shape = iteration_shape[:shifts_end_axis]
    # Chunking it up finer (still aligned with chunks on disk) does not make it faster (actually slower by about a factor
    # of 3). This might change with a storage handler that allows multi-threaded access but for now with h5py we don't
    # want to use this.
    # squeezed_iteration_shape = iteration_shape[:max(shifts_end_axis, shift_axes_indices[0])]

    if out is None:
        result = numpy.empty(xdata.data_shape, dtype=xdata.data_dtype)
    else:
        result = out.data

    navigation_len = int(numpy.prod(squeezed_iteration_shape, dtype=numpy.int64))
    sections = list(range(0, navigation_len, max(1, navigation_len//8)))
    sections.append(navigation_len)
    barrier = threading.Barrier(len(sections))

    def run_on_thread(range_: range) -> None:
        try:
            shifts_array = numpy.zeros(len(shift_axes) + (len(iteration_shape) - len(squeezed_iteration_shape)))
            if shifts_end_axis < len(shifts.shape):
                for i in range_:
                    coords = numpy.unravel_index(i, squeezed_iteration_shape)
                    shift_coords = coords[:shifts_end_axis]
                    for j, ind in enumerate(shift_axes):
                        shifts_array[ind - len(squeezed_iteration_shape)] = shifts[shift_coords][j]
                    # if i % max((range_.stop - range_.start) // 4, 1) == 0:
                    #     print(f'Working on slice {coords}: shifting by {shifts_array}')
                    result[coords] = scipy.ndimage.shift(xdata.data[coords], shifts_array, order=1)
            # Note: Once we have multi-dimensional sequences, we need and implementation for iteration_shape_offset != 0
            # and shifts for more than 1-D data (so similar to the loop above but with offset)
            elif iteration_shape_offset != 0:
                offset_slices = tuple([slice(None) for _ in range(iteration_shape_offset)])
                for i in range_:
                    shift_coords = numpy.unravel_index(i, squeezed_iteration_shape)
                    # need a different name here to make typing happy
                    coords2 = offset_slices + shift_coords
                    shifts_array[0] = shifts[shift_coords]
                    result[coords2] = scipy.ndimage.shift(xdata.data[coords2], shifts_array, order=1)
            else:
                for i in range_:
                    coords = numpy.unravel_index(i, squeezed_iteration_shape)
                    shifts_array[0] = shifts[coords]
                    result[coords] = scipy.ndimage.shift(xdata.data[coords], shifts_array, order=1)
        finally:
            barrier.wait()

    for i in range(len(sections) - 1):
        threading.Thread(target=run_on_thread, args=(range(sections[i], sections[i+1]),)).start()
    barrier.wait()
    # For debugging it is helpful to run a non-threaded version of the code. Comment out the 3 lines above and uncomment
    # the line below to do so. You also need to comment out "barrier.wait()" in the function running on the thread.
    # run_on_thread(range(0, navigation_len))

    if out is None:
        return DataAndMetadata.new_data_and_metadata(result,
                                                     intensity_calibration=xdata.intensity_calibration,
                                                     dimensional_calibrations=xdata.dimensional_calibrations,
                                                     metadata=xdata.metadata,
                                                     data_descriptor=xdata.data_descriptor)
    return None


def function_make_tableau_image(xdata: DataAndMetadata.DataAndMetadata,
                                scale: float = 1.0) -> DataAndMetadata.DataAndMetadata:
    assert xdata.is_collection or xdata.is_sequence
    assert xdata.datum_dimension_count == 2

    iteration_shape: typing.Tuple[int, ...] = tuple()
    tableau_shape: typing.Tuple[int, ...] = tuple()
    iteration_start_index: typing.Optional[int] = None
    data_descriptor: typing.Optional[DataAndMetadata.DataDescriptor] = None
    if xdata.is_collection:
        iteration_shape = tuple([xdata.data.shape[index] for index in xdata.collection_dimension_indexes])
        iteration_start_index = xdata.collection_dimension_indexes[0]
        data_descriptor = DataAndMetadata.DataDescriptor(xdata.is_sequence, 0, 2)
    elif xdata.is_sequence:
        iteration_start_index = xdata.sequence_dimension_index
        assert iteration_start_index is not None
        iteration_shape = (xdata.data.shape[iteration_start_index],)
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
    assert iteration_start_index is not None

    tableau_height = int(numpy.sqrt(numpy.prod(iteration_shape, dtype=numpy.int64)))
    tableau_width = int(numpy.ceil(numpy.prod(iteration_shape, dtype=numpy.int64) / tableau_height))
    tableau_shape = (tableau_height, tableau_width)

    result = typing.cast(_ImageDataType, None)
    for i in range(numpy.prod(iteration_shape, dtype=numpy.int64)):
        coords = numpy.unravel_index(i, iteration_shape)
        data_coords = tuple([slice(None) for k in range(iteration_start_index)]) + coords
        if scale != 1.0:
            scale_sequence = [1.0] * iteration_start_index + [scale] * 2
            scaled_data = scipy.ndimage.zoom(xdata.data[data_coords], scale_sequence, order=1)
        else:
            scaled_data = xdata.data[data_coords]

        if i==0:
            result = numpy.zeros(xdata.data.shape[:iteration_start_index] + (scaled_data.shape[-2] * tableau_height, scaled_data.shape[-1] * tableau_width), dtype=xdata.data.dtype)

        coords = numpy.unravel_index(i, tableau_shape)
        pos = (coords[0] * scaled_data.shape[-2], coords[1] * scaled_data.shape[-1])
        result_coords = tuple([slice(None) for k in range(iteration_start_index)]) + (slice(pos[0], pos[0] + scaled_data.shape[-2]), slice(pos[1], pos[1] + scaled_data.shape[-1]))
        result[result_coords] = scaled_data

    dimensional_calibrations = list(copy.deepcopy(xdata.dimensional_calibrations))
    [dimensional_calibrations.pop(iteration_start_index) for _ in range(len(iteration_shape))]
    return DataAndMetadata.new_data_and_metadata(result,
                                                 intensity_calibration=xdata.intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations,
                                                 metadata=xdata.metadata,
                                                 data_descriptor=data_descriptor)
