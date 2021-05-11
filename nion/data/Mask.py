import typing

import numpy

from nion.data import ReferenceFrame
from nion.data import Shape


class Mask:
    def get_mask_array_1d(self, reference_frame: ReferenceFrame.ReferenceFrame1D) -> numpy.ndarray:
        raise NotImplementedError

    def get_mask_array(self, reference_frame: ReferenceFrame.ReferenceFrame2D) -> numpy.ndarray:
        mask = numpy.zeros(reference_frame.shape, dtype=float)
        self.apply_mask(reference_frame, mask, Shape.Operation.OR)
        return mask

    def apply_mask(self, reference_frame: ReferenceFrame.ReferenceFrame2D, mask: numpy.ndarray,
                   operator: Shape.Operation) -> numpy.ndarray:
        raise NotImplementedError


class LogicalBinaryMask(Mask):
    def __init__(self, operator: Shape.Operation, mask1: Mask, mask2: Mask):
        self.operator = operator
        self.mask1 = mask1
        self.mask2 = mask2

    def apply_mask(self, reference_frame: ReferenceFrame.ReferenceFrame2D, mask: numpy.ndarray,
                   operator: Shape.Operation) -> numpy.ndarray:
        mask_data = self.mask1.get_mask_array(reference_frame)
        self.mask2.apply_mask(reference_frame, mask_data, self.operator)
        if operator == Shape.Operation.OR:
            numpy.logical_or(mask, mask_data, out=mask)
        elif operator == Shape.Operation.AND:
            numpy.logical_and(mask, mask_data, out=mask)
        elif operator == Shape.Operation.NOT:
            numpy.logical_not(mask, mask_data, out=mask)
        return mask


class LogicalUnaryMask(Mask):
    def __init__(self, operator, mask: Mask):
        self.operator = operator
        self.mask = mask


class PointMask(Mask):
    def __init__(self, point: Shape.Point, radius: typing.Optional[ReferenceFrame.Metric] = None):
        self.point = point
        self.radius = radius or ReferenceFrame.Coordinate(ReferenceFrame.CoordinateType.PIXEL, 1)


class LineMask(Mask):
    def __init__(self, line: Shape.Line):
        self.line = line


class RectangleMask(Mask):
    def __init__(self, rectangle: Shape.Rectangle):
        self.rectangle = rectangle

    def apply_mask(self, reference_frame: ReferenceFrame.ReferenceFrame2D, mask: numpy.ndarray,
                   operator: Shape.Operation) -> numpy.ndarray:
        return self.rectangle.apply_mask(reference_frame, mask, operator)


class EllipseMask(Mask):
    def __init__(self, ellipse: Shape.Ellipse):
        self.ellipse = ellipse


class SpotMask(Mask):
    def __init__(self, rectangle: Shape.Rectangle):
        self.rectangle = rectangle


class WedgeMask(Mask):
    def __init__(self, start_angle: float, end_angle: float):
        self.start_angle = start_angle
        self.end_angle = end_angle


class RingMask(Mask):
    def __init__(self, radius: ReferenceFrame.Metric):
        self.radius = radius


class LatticeMask(Mask):
    def __init__(self, u_rectangle: Shape.Rectangle, v_rectangle: Shape.Rectangle):
        self.u_rectangle = u_rectangle
        self.v_rectangle = v_rectangle
