import enum
import math
import typing

import numpy

from nion.data import ReferenceFrame
from nion.utils import Geometry


class Operation(enum.IntEnum):
    OR = 0
    AND = 1
    NOT = 2


class Point:
    def __init__(self, x: ReferenceFrame.Coordinate, y: ReferenceFrame.Coordinate):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point ({self.x!r} {self.y!r})"


class Size:
    def __init__(self, width: ReferenceFrame.Metric, height: ReferenceFrame.Metric):
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Size ({self.width!r} {self.height!r})"


class Rectangle:
    def __init__(self, center: Point, size: Size, rotation: float = 0.0):
        self.center = center
        self.size = size
        self.rotation = rotation

    def __repr__(self):
        return f"Rectangle ({self.center!r} {self.size!r} rotation {self.rotation})"

    def get_bounds_int(self, reference_frame: ReferenceFrame.ReferenceFrame2D) -> Geometry.IntRect:
        cy = reference_frame.y_axis.convert_to_normalized(self.center.y)
        cx = reference_frame.x_axis.convert_to_normalized(self.center.x)
        h = reference_frame.y_axis.convert_to_normalized_size(self.size.height)
        w = reference_frame.x_axis.convert_to_normalized_size(self.size.width)
        t = reference_frame.y_axis.convert_to_pixel(ReferenceFrame.Coordinate(ReferenceFrame.CoordinateType.NORMALIZED, cy.value - h.value / 2)).int_value
        l = reference_frame.x_axis.convert_to_pixel(ReferenceFrame.Coordinate(ReferenceFrame.CoordinateType.NORMALIZED, cx.value - w.value / 2)).int_value
        b = reference_frame.y_axis.convert_to_pixel(ReferenceFrame.Coordinate(ReferenceFrame.CoordinateType.NORMALIZED, cy.value + h.value / 2)).int_value
        r = reference_frame.x_axis.convert_to_pixel(ReferenceFrame.Coordinate(ReferenceFrame.CoordinateType.NORMALIZED, cx.value + w.value / 2)).int_value
        return Geometry.IntRect.from_tlbr(t, l, b, r)

    def apply_mask(self, reference_frame: ReferenceFrame.ReferenceFrame2D, mask: numpy.ndarray,
                   operator: Operation) -> numpy.ndarray:
        data_shape = reference_frame.shape
        bounds_int = self.get_bounds_int(reference_frame)
        rotation = self.rotation
        a, b = bounds_int.top + bounds_int.height * 0.5, bounds_int.left + bounds_int.width * 0.5
        y, x = numpy.ogrid[-a:data_shape[0] - a, -b:data_shape[1] - b]
        if rotation:
            angle_sin = math.sin(rotation)
            angle_cos = math.cos(rotation)
            mask_eq = ((-1 <= (x * angle_cos - y * angle_sin) / (bounds_int.width / 2)) & (
                        (x * angle_cos - y * angle_sin) / (bounds_int.width / 2) < 1)) & (
                                  (-1 <= (y * angle_cos + x * angle_sin) / (bounds_int.height / 2)) & (
                                      (y * angle_cos + x * angle_sin) / (bounds_int.height / 2) < 1))
        else:
            mask_eq = ((-1 <= x / (bounds_int.width / 2)) & (x / (bounds_int.width / 2) < 1)) & (
                    (-1 <= y / (bounds_int.height / 2)) & (y / (bounds_int.height / 2) < 1))
        if operator == Operation.OR:
            mask[mask_eq] = 1
        elif operator == Operation.AND:
            mask[mask_eq] = 0
        elif operator == Operation.NOT:
            mask[mask_eq] = 1 - mask[mask_eq]
        return mask


class Ellipse:
    def __init__(self, center: Point, size: Size, rotation: float):
        self.center = center
        self.size = size
        self.rotation = rotation

    def __repr__(self):
        return f"Ellipse ({self.center!r} {self.size!r} rotation {self.rotation})"


class Line:
    def __init__(self, start: Point, end: Point, width: typing.Optional[ReferenceFrame.Metric] = None):
        self.start = start
        self.end = end
        self.width = width


class Position:
    def __init__(self, position: ReferenceFrame.Coordinate):
        self.position = position

    def __repr__(self):
        return f"Position {self.position!r}"


class Interval:
    def __init__(self, start: ReferenceFrame.Coordinate, end: ReferenceFrame.Coordinate):
        assert start.coordinate_type == end.coordinate_type
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Interval [{self.start!r} {self.end!r})"

    @property
    def length(self) -> ReferenceFrame.Coordinate:
        return ReferenceFrame.Coordinate(self.start.coordinate_type, self.end.value - self.start.value)
