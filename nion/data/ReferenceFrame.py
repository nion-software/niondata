import enum
import typing

from nion.data import Calibration


class CoordinateType(enum.IntEnum):
    CALIBRATED = 0
    NORMALIZED = 1
    PIXEL = 2


class Coordinate:
    def __init__(self, coordinate_type: CoordinateType, value: float):
        self.coordinate_type = coordinate_type
        self.value = value
        self.int_value = int(value)

    def __repr__(self):
        return f"{str(self.coordinate_type).split('.')[-1]}:{self.value}"


class Metric:
    def __init__(self, coordinate_type: CoordinateType, value: float):
        self.coordinate_type = coordinate_type
        self.value = value
        self.int_value = int(value)

    def __repr__(self):
        return f"{str(self.coordinate_type).split('.')[-1]}:{self.value}"


class ReferenceFrameAxis:
    def __init__(self, calibration: Calibration.Calibration, n: int):
        self.calibration = calibration
        self.n = n

    def convert_to_calibrated(self, c: Coordinate) -> Coordinate:
        if c.coordinate_type == CoordinateType.CALIBRATED:
            return Coordinate(CoordinateType.CALIBRATED, c.value)
        if c.coordinate_type == CoordinateType.NORMALIZED:
            return Coordinate(CoordinateType.CALIBRATED, self.calibration.convert_to_calibrated_value(c.value * self.n))
        if c.coordinate_type == CoordinateType.PIXEL:
            return Coordinate(CoordinateType.CALIBRATED, self.calibration.convert_to_calibrated_value(c.value / self.n))
        raise NotImplementedError()

    def convert_to_pixel(self, c: Coordinate) -> Coordinate:
        if c.coordinate_type == CoordinateType.CALIBRATED:
            return Coordinate(CoordinateType.PIXEL, self.calibration.convert_from_calibrated_value(c.value))
        if c.coordinate_type == CoordinateType.NORMALIZED:
            return Coordinate(CoordinateType.PIXEL, c.value * self.n)
        if c.coordinate_type == CoordinateType.PIXEL:
            return Coordinate(CoordinateType.PIXEL, c.value)
        raise NotImplementedError()

    def convert_to_normalized(self, c: Coordinate) -> Coordinate:
        if c.coordinate_type == CoordinateType.CALIBRATED:
            return Coordinate(CoordinateType.NORMALIZED, self.calibration.convert_from_calibrated_value(c.value) / self.n)
        if c.coordinate_type == CoordinateType.NORMALIZED:
            return Coordinate(CoordinateType.NORMALIZED, c.value)
        if c.coordinate_type == CoordinateType.PIXEL:
            return Coordinate(CoordinateType.NORMALIZED, c.value / self.n)
        raise NotImplementedError()

    def convert_to_calibrated_size(self, m: Metric) -> Metric:
        if m.coordinate_type == CoordinateType.CALIBRATED:
            return Metric(CoordinateType.CALIBRATED, m.value)
        if m.coordinate_type == CoordinateType.NORMALIZED:
            return Metric(CoordinateType.CALIBRATED, self.calibration.convert_to_calibrated_size(m.value * self.n))
        if m.coordinate_type == CoordinateType.PIXEL:
            return Metric(CoordinateType.CALIBRATED, self.calibration.convert_to_calibrated_size(m.value / self.n))
        raise NotImplementedError()

    def convert_to_pixel_size(self, m: Metric) -> Metric:
        if m.coordinate_type == CoordinateType.CALIBRATED:
            return Metric(CoordinateType.PIXEL, self.calibration.convert_from_calibrated_size(m.value))
        if m.coordinate_type == CoordinateType.NORMALIZED:
            return Metric(CoordinateType.PIXEL, m.value * self.n)
        if m.coordinate_type == CoordinateType.PIXEL:
            return Metric(CoordinateType.PIXEL, m.value)
        raise NotImplementedError()

    def convert_to_normalized_size(self, m: Metric) -> Metric:
        if m.coordinate_type == CoordinateType.CALIBRATED:
            return Metric(CoordinateType.NORMALIZED, self.calibration.convert_from_calibrated_size(m.value) / self.n)
        if m.coordinate_type == CoordinateType.NORMALIZED:
            return Metric(CoordinateType.NORMALIZED, m.value)
        if m.coordinate_type == CoordinateType.PIXEL:
            return Metric(CoordinateType.NORMALIZED, m.value / self.n)
        raise NotImplementedError()


class ReferenceFrame1D:
    def __init__(self, axis: ReferenceFrameAxis):
        self.axis = axis


class ReferenceFrame2D:
    def __init__(self, y_axis: ReferenceFrameAxis, x_axis: ReferenceFrameAxis):
        self.y_axis = y_axis
        self.x_axis = x_axis

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return self.y_axis.n, self.x_axis.n
