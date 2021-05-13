import math
import typing

import numpy

from nion.data import Calibration
from nion.utils import Geometry


class Mask1DShape:
    # this is a protocol; but no way to enforce this until Python 3.8 is the earliest target version

    def get_mask_data_1d(self, reference_frame: Calibration.ReferenceFrame1D) -> numpy.ndarray:
        raise NotImplementedError()


class Mask2DShape:
    # this is a protocol; but no way to enforce this until Python 3.8 is the earliest target version

    def get_mask_data_2d(self, reference_frame: Calibration.ReferenceFrame2D) -> numpy.ndarray:
        raise NotImplementedError()


class Point:
    def __init__(self, *, x: Calibration.Coordinate, y: Calibration.Coordinate):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point ({self.x!r} {self.y!r})"


class Size:
    def __init__(self, *, width: Calibration.Metric, height: Calibration.Metric):
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Size ({self.width!r} {self.height!r})"


class Rectangle(Mask2DShape):
    def __init__(self, center: Point, size: Size, rotation: float = 0.0):
        self.center = center
        self.size = size
        self.rotation = rotation

    @classmethod
    def from_tlbr_fractional(cls, top: float, left: float, bottom: float, right: float):
        center_y = Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, (top + bottom) / 2)
        center_x = Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, (left + right) / 2)
        height = Calibration.Metric(Calibration.CoordinateType.NORMALIZED, bottom - top)
        width = Calibration.Metric(Calibration.CoordinateType.NORMALIZED, right - left)
        center = Point(y=center_y, x=center_x)
        size = Size(height=height, width=width)
        return cls(center, size)

    def __repr__(self):
        return f"Rectangle ({self.center!r} {self.size!r} rotation {self.rotation})"

    def get_bounds_int(self, reference_frame: Calibration.ReferenceFrame2D) -> Geometry.IntRect:
        cy = reference_frame.y_axis.convert_to_normalized(self.center.y)
        cx = reference_frame.x_axis.convert_to_normalized(self.center.x)
        h = reference_frame.y_axis.convert_to_normalized_size(self.size.height)
        w = reference_frame.x_axis.convert_to_normalized_size(self.size.width)
        t = reference_frame.y_axis.convert_to_pixel(Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, cy.value - h.value / 2)).int_value
        l = reference_frame.x_axis.convert_to_pixel(Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, cx.value - w.value / 2)).int_value
        b = reference_frame.y_axis.convert_to_pixel(Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, cy.value + h.value / 2)).int_value
        r = reference_frame.x_axis.convert_to_pixel(Calibration.Coordinate(Calibration.CoordinateType.NORMALIZED, cx.value + w.value / 2)).int_value
        return Geometry.IntRect.from_tlbr(t, l, b, r)

    def get_mask_data_2d(self, reference_frame: Calibration.ReferenceFrame2D) -> numpy.ndarray:
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
        mask = numpy.zeros(reference_frame.shape, dtype=float)
        mask[mask_eq] = 1
        return mask


class Ellipse(Mask2DShape):
    def __init__(self, center: Point, size: Size, rotation: float):
        self.center = center
        self.size = size
        self.rotation = rotation

    def __repr__(self):
        return f"Ellipse ({self.center!r} {self.size!r} rotation {self.rotation})"


class Line:
    def __init__(self, start: Point, end: Point, width: typing.Optional[Calibration.Metric] = None):
        self.start = start
        self.end = end
        self.width = width


class Position:
    def __init__(self, position: Calibration.Coordinate):
        self.position = position

    def __repr__(self):
        return f"Position {self.position!r}"


class Interval(Mask1DShape):
    def __init__(self, start: Calibration.Coordinate, end: Calibration.Coordinate):
        assert start.coordinate_type == end.coordinate_type
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Interval [{self.start!r} {self.end!r})"

    @property
    def length(self) -> Calibration.Coordinate:
        return Calibration.Coordinate(self.start.coordinate_type, self.end.value - self.start.value)


setattr(Calibration, "CalibratedInterval", Interval)  # TODO: backwards compatibility for EELS analysis; remove once it uses shape.


class _Composite2DShape(Mask2DShape):
    def __init__(self, mask1: Mask2DShape, mask2: Mask2DShape, operator_fn):
        self.mask1 = mask1
        self.mask2 = mask2
        self.__operator_fn = operator_fn

    def get_mask_data_2d(self, reference_frame: Calibration.ReferenceFrame2D) -> numpy.ndarray:
        mask_data = self.mask1.get_mask_data_2d(reference_frame)
        self.__operator_fn(mask_data, self.mask2.get_mask_data_2d(reference_frame), out=mask_data)
        return mask_data


class OrComposite2DShape(_Composite2DShape):
    def __init__(self, mask1: Mask2DShape, mask2: Mask2DShape):
        super().__init__(mask1, mask2, numpy.logical_or)


class AndComposite2DShape(_Composite2DShape):
    def __init__(self, mask1: Mask2DShape, mask2: Mask2DShape):
        super().__init__(mask1, mask2, numpy.logical_and)


class XorComposite2DShape(_Composite2DShape):
    def __init__(self, mask1: Mask2DShape, mask2: Mask2DShape):
        super().__init__(mask1, mask2, numpy.logical_xor)


class SpotShape(Mask2DShape):
    def __init__(self, rectangle: Rectangle):
        self.rectangle = rectangle


class WedgeShape(Mask2DShape):
    def __init__(self, start_angle: float, end_angle: float):
        self.start_angle = start_angle
        self.end_angle = end_angle


class RingShape(Mask2DShape):
    def __init__(self, radius: Calibration.Metric):
        self.radius = radius


class LatticeShape(Mask2DShape):
    def __init__(self, u_rectangle: Rectangle, v_rectangle: Rectangle):
        self.u_rectangle = u_rectangle
        self.v_rectangle = v_rectangle
