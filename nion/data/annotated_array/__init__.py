"""Public annotated-array API surface.

Import this package as the single entry point for annotated-array related functionality:

    from nion.data import annotated_array as aa
"""

from ._implementation import AffineCalibration
from ._implementation import AnnotatedArray
from ._implementation import ArrayDescriptor
from ._implementation import ArrayHeader
from ._implementation import ArrayMetadata
from ._implementation import ArrayProtocol
from ._implementation import Axis
from ._implementation import AxisGroup
from ._implementation import Calibration
from ._implementation import CalibrationSet
from ._implementation import CoordinateCalibration
from ._implementation import ExtensionRecord
from ._implementation import ValueType
from ._implementation import from_data_and_metadata
from ._implementation import infer_value_type
from ._implementation import to_data_and_metadata
from ._implementation import zeros_annotated_array

__all__ = [
    "AffineCalibration",
    "AnnotatedArray",
    "ArrayDescriptor",
    "ArrayHeader",
    "ArrayMetadata",
    "ArrayProtocol",
    "Axis",
    "AxisGroup",
    "Calibration",
    "CalibrationSet",
    "CoordinateCalibration",
    "ExtensionRecord",
    "ValueType",
    "from_data_and_metadata",
    "infer_value_type",
    "to_data_and_metadata",
    "zeros_annotated_array",
]


