import numpy
import scipy.fft
import scipy.ndimage
import typing

from nion.data import Image

_ShapeType = Image.ShapeType
_ImageDataType = Image._ImageDataType


def normalized_corr(image: _ImageDataType, template: _ImageDataType) -> _ImageDataType:
    """
    Correctly normalized template matching by cross-correlation. The result should be the same as what you get from
    openCV's "match_template" function with method set to "ccoeff_normed", except for the output shape, which will
    be image.shape here (as opposed to openCV, where only the valid portion of the image is returned).
    Used ideas from here:
    http://scribblethink.org/Work/nvisionInterface/nip.pdf (which is an extended version of this paper:
    J. P. Lewis, "FastTemplateMatching", Vision Interface, p. 120-123, 1995)
    """
    template = template.astype(numpy.float64)
    image = image.astype(numpy.float64)
    normalized_template = template - numpy.mean(template)
    # inverting the axis of a real image is the same as taking the conjugate of the fourier transform
    fft_normalized_template_conj = scipy.fft.fft2(normalized_template[::-1, ::-1], s=image.shape)  # type: ignore
    fft_image = scipy.fft.fft2(image)  # type: ignore
    fft_image_squared = scipy.fft.fft2(image ** 2)  # type: ignore
    fft_image_squared_means = scipy.ndimage.fourier_uniform(fft_image_squared, template.shape)
    image_means_squared = (scipy.fft.ifft2(
        scipy.ndimage.fourier_uniform(fft_image, template.shape)).real) ** 2  # type: ignore
    # only normalizing the template is equivalent to normalizing both (see paper in docstring for details)
    fft_corr = fft_image * fft_normalized_template_conj
    # we need to shift the result back by half the template size
    shift = (int(-1 * (template.shape[0] - 1) / 2), int(-1 * (template.shape[1] - 1) / 2))
    corr = numpy.roll(scipy.fft.ifft2(fft_corr).real, shift=shift, axis=(0, 1))  # type: ignore
    # use Var(X) = E(X^2) - E(X)^2 to calculate variance
    image_variance = scipy.fft.ifft2(fft_image_squared_means).real - image_means_squared  # type: ignore
    denom = image_variance * template.size * numpy.sum(normalized_template ** 2)
    denom[denom < 0] = numpy.amax(denom)
    return typing.cast(_ImageDataType, corr / numpy.sqrt(denom))


def parabola_through_three_points(p1: typing.Tuple[int, int], p2: typing.Tuple[int, int], p3: typing.Tuple[int, int]) -> typing.Tuple[float, float, float]:
    """
    Calculates the parabola a*(x-b)**2+c through three points. The points should be given as (y, x) tuples.
    Returns a tuple (a, b, c)
    """
    # formula taken from http://stackoverflow.com/questions/4039039/fastest-way-to-fit-a-parabola-to-set-of-points
    # Avoid division by zero in calculation of s
    if p2[0] == p3[0]:
        temp = p2
        p2 = p1
        p1 = temp

    s = (p1[0] - p2[0]) / (p2[0] - p3[0])
    b = (-p1[1] ** 2 + p2[1] ** 2 + s * (p2[1] ** 2 - p3[1] ** 2)) / (2 * (-p1[1] + p2[1] + s * p2[1] - s * p3[1]))
    a = (p1[0] - p2[0]) / ((p1[1] - b) ** 2 - (p2[1] - b) ** 2)
    c = p1[0] - a * (p1[1] - b) ** 2
    return (a, b, c)


def find_ccorr_max(ccorr: _ImageDataType) -> typing.Tuple[int, typing.Optional[float], typing.Optional[typing.Tuple[float, ...]]]:
    max_pos: typing.Tuple[int, ...] = typing.cast(typing.Tuple[int, ...], numpy.unravel_index(numpy.argmax(ccorr), ccorr.shape))
    if ccorr.ndim == 2:
        if typing.cast(_ImageDataType, numpy.array(max_pos) < numpy.array((1,1))).any() or typing.cast(_ImageDataType, numpy.array(max_pos) > numpy.array(ccorr.shape) - 2).any():
            return 1, ccorr[max_pos], tuple(float(p) for p in max_pos)
    elif ccorr.ndim == 1:
        if max_pos[0] < 1 or max_pos[0] > ccorr.shape[0] - 2:
            return 1, ccorr[max_pos], tuple(float(p) for p in max_pos)
    else:
        return 1, None, None

    if ccorr.ndim == 2:
        max_y = ccorr[max_pos[0]-1:max_pos[0]+2, max_pos[1]]
        parabola_y = parabola_through_three_points((max_y[0], max_pos[0]-1),
                                                   (max_y[1], max_pos[0]  ),
                                                   (max_y[2], max_pos[0]+1))
        max_x = ccorr[max_pos[0], max_pos[1]-1:max_pos[1]+2]
        parabola_x = parabola_through_three_points((max_x[0], max_pos[1]-1),
                                                   (max_x[1], max_pos[1]  ),
                                                   (max_x[2], max_pos[1]+1))
        return 0, ccorr[max_pos], (parabola_y[1], parabola_x[1])

    max_ = ccorr[max_pos[0]-1:max_pos[0]+2]
    parabola = parabola_through_three_points((max_[0], max_pos[0]-1),
                                             (max_[1], max_pos[0]  ),
                                             (max_[2], max_pos[0]+1))

    return 0, ccorr[max_pos], (parabola[1],)


def match_template(image: _ImageDataType, template: _ImageDataType) -> _ImageDataType:
    ccorr = normalized_corr(image, template)
    ccorr[ccorr > 1.1] = 0
    return ccorr
