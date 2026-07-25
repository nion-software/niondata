import datetime
import io
import typing
import unittest
import zoneinfo

import h5py
import numpy

from nion.data import AnnotatedArray


class TestAnnotatedArray(unittest.TestCase):

    def test_calibration_set_uses_explicit_calibration_accessors(self) -> None:
        primary = AnnotatedArray.AffineCalibration(unit="nm")
        alternate = AnnotatedArray.AffineCalibration(unit="rad")
        calibrations = AnnotatedArray.CalibrationSet.from_calibration(primary, key="primary").with_calibration("alternate", alternate)

        self.assertTrue(calibrations.has_calibration("alternate"))
        self.assertIs(primary, calibrations.primary_calibration)
        self.assertIs(alternate, calibrations.get_calibration("alternate"))
        self.assertIs(alternate, calibrations.with_primary_calibration("alternate").primary_calibration)

    def test_array_descriptor_describes_shape_and_rank(self) -> None:
        collection = AnnotatedArray.AxisGroup.from_1d_size(3)
        signal = AnnotatedArray.AxisGroup.from_2d_size((4, 5))
        descriptor = AnnotatedArray.ArrayDescriptor((collection, signal))

        self.assertEqual((3, 4, 5), descriptor.shape)
        self.assertEqual(3, descriptor.ndim)

    def test_array_descriptor_uses_identity_intensity_calibration_when_no_primary_is_designated(self) -> None:
        descriptor = AnnotatedArray.ArrayDescriptor((AnnotatedArray.AxisGroup.from_1d_size(3),))

        calibration = descriptor.get_intensity_calibration()
        self.assertIsInstance(calibration, AnnotatedArray.AffineCalibration)
        affine_calibration = typing.cast(AnnotatedArray.AffineCalibration, calibration)
        self.assertEqual(1.0, affine_calibration.scale)
        self.assertEqual(0.0, affine_calibration.offset)
        self.assertEqual("", affine_calibration.unit)

        with self.assertRaises(KeyError):
            descriptor.get_intensity_calibration("missing")

    def test_axis_group_size_factories_accept_optional_coordinate_calibrations(self) -> None:
        single_calibration = AnnotatedArray.CoordinateCalibration(calibrations=(AnnotatedArray.AffineCalibration(unit="nm"),))
        group_1d = AnnotatedArray.AxisGroup.from_1d_size(
            3,
            coordinate_calibrations={"spatial": single_calibration},
            primary_calibration_key="spatial",
        )
        self.assertEqual(("spatial",), group_1d.calibration_keys)
        self.assertEqual("spatial", group_1d.primary_calibration_key)

        map_calibration = AnnotatedArray.CoordinateCalibration(
            calibrations=(AnnotatedArray.AffineCalibration(unit="nm"), AnnotatedArray.AffineCalibration(unit="nm"))
        )
        group_2d = AnnotatedArray.AxisGroup.from_2d_size(
            (2, 3),
            coordinate_calibrations={"camera": map_calibration},
            primary_calibration_key="camera",
        )
        self.assertEqual(("camera",), group_2d.calibration_keys)
        self.assertEqual("camera", group_2d.primary_calibration_key)

    def test_array_descriptor_requires_valid_axis_group_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            AnnotatedArray.ArrayDescriptor(())

        scalar_group = AnnotatedArray.AxisGroup()
        vector_group = AnnotatedArray.AxisGroup.from_1d_size(3)
        with self.assertRaisesRegex(ValueError, "Only the final"):
            AnnotatedArray.ArrayDescriptor((scalar_group, vector_group))

    def test_array_metadata_controls_extension_access(self) -> None:
        extension = AnnotatedArray.ExtensionRecord("org.nion.test", 1, "value=42")
        metadata = AnnotatedArray.ArrayMetadata(extensions=(extension,))

        self.assertEqual("org.nion.test", extension.extension_type_id)
        self.assertEqual(("org.nion.test",), metadata.extension_type_ids)
        self.assertTrue(metadata.has_extension("org.nion.test"))
        self.assertIs(extension, metadata.get_extension("org.nion.test"))
        with self.assertRaises(KeyError):
            metadata.get_extension("org.nion.missing")

        replacement = AnnotatedArray.ExtensionRecord("org.nion.test", 2, "value=43")
        replaced_metadata = metadata.with_extension(replacement)
        self.assertEqual(2, replaced_metadata.get_extension("org.nion.test").schema_version)
        self.assertEqual(1, metadata.get_extension("org.nion.test").schema_version)
        self.assertFalse(replaced_metadata.without_extension("org.nion.test").has_extension("org.nion.test"))

    def test_array_metadata_validates_extensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            AnnotatedArray.ExtensionRecord("", 1, "")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            AnnotatedArray.ExtensionRecord("org.nion.test", 0, "")
        with self.assertRaisesRegex(TypeError, "must be str"):
            AnnotatedArray.ExtensionRecord("org.nion.test", 1, bytearray())  # type: ignore[arg-type]

        extension = AnnotatedArray.ExtensionRecord("org.nion.test", 1, "")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            AnnotatedArray.ArrayMetadata(extensions=(extension, extension))

    def test_array_metadata_snapshots_attributes_and_requires_timezone(self) -> None:
        source = {"note": "original"}
        created = datetime.datetime(2026, 7, 16, tzinfo=datetime.timezone.utc)
        metadata = AnnotatedArray.ArrayMetadata(created=created, attributes=source)
        source["note"] = "changed"

        self.assertEqual("original", metadata.attributes["note"])
        self.assertEqual("replacement", metadata.with_attributes({"note": "replacement"}).attributes["note"])
        self.assertEqual(datetime.timedelta(0), metadata.created.utcoffset())
        with self.assertRaises(TypeError):
            metadata.attributes["note"] = "changed"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            AnnotatedArray.ArrayMetadata(created=datetime.datetime(2026, 7, 16))

    def test_array_metadata_created_retains_iana_zone_and_offset(self) -> None:
        created = datetime.datetime(2026, 7, 16, 12, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles"))
        metadata = AnnotatedArray.ArrayMetadata(created=created)

        self.assertEqual("America/Los_Angeles", getattr(metadata.created.tzinfo, "key", None))
        self.assertEqual(datetime.timedelta(hours=-7), metadata.created.utcoffset())

    def test_array_header_can_be_passed_without_data(self) -> None:
        descriptor = AnnotatedArray.ArrayDescriptor((AnnotatedArray.AxisGroup.from_2d_size((4, 5)),))
        metadata = AnnotatedArray.ArrayMetadata(attributes={"note": "test"})
        header = AnnotatedArray.ArrayHeader(descriptor, "float32", metadata)

        self.assertEqual((4, 5), header.shape)
        self.assertEqual(numpy.dtype(numpy.float32), header.dtype)

        first = AnnotatedArray.AnnotatedArray.from_header(numpy.zeros(header.shape, dtype=header.dtype), header)
        second = AnnotatedArray.AnnotatedArray.from_header(numpy.ones(header.shape, dtype=header.dtype), header)
        self.assertEqual(header, first.header)
        self.assertIsNot(header, first.header)
        self.assertEqual(first.header, second.header)
        self.assertIs(descriptor, first.descriptor)
        self.assertIs(metadata, first.metadata)

    def test_annotated_array_validates_data_against_header(self) -> None:
        descriptor = AnnotatedArray.ArrayDescriptor((AnnotatedArray.AxisGroup.from_1d_size(3),))
        header = AnnotatedArray.ArrayHeader(descriptor, numpy.float32)

        with self.assertRaisesRegex(ValueError, "shape"):
            AnnotatedArray.AnnotatedArray(numpy.zeros((4,), dtype=numpy.float32), descriptor)
        with self.assertRaisesRegex(ValueError, "dtype"):
            AnnotatedArray.AnnotatedArray.from_header(numpy.zeros((3,), dtype=numpy.float64), header)

    def test_zeros_annotated_array_constructs_matching_shape_and_dtype(self) -> None:
        group = AnnotatedArray.AxisGroup.from_2d_size((2, 3))
        array = AnnotatedArray.zeros_annotated_array((group,), dtype=numpy.float32)

        self.assertEqual((2, 3), array.data.shape)
        self.assertEqual(numpy.dtype(numpy.float32), array.header.dtype)

    def test_annotated_array_is_numpy_passable(self) -> None:
        group = AnnotatedArray.AxisGroup.from_1d_size(4)
        array = AnnotatedArray.zeros_annotated_array((group,), dtype=numpy.float64)

        # AnnotatedArray itself is directly usable with numpy functions via __array__
        self.assertEqual(0.0, float(numpy.sum(array)))
        as_array = numpy.asarray(array)
        self.assertIsInstance(as_array, numpy.ndarray)
        self.assertEqual((4,), as_array.shape)
        self.assertEqual(numpy.dtype(numpy.float64), as_array.dtype)

    def test_annotated_array_data_is_numpy_passable(self) -> None:
        group = AnnotatedArray.AxisGroup.from_2d_size((2, 3))
        array = AnnotatedArray.zeros_annotated_array((group,), dtype=numpy.float32)

        # data satisfies ArrayProtocol including __array__, so it is directly usable with numpy
        self.assertEqual(0.0, float(numpy.sum(array.data)))
        result = numpy.asarray(array.data)
        self.assertIsInstance(result, numpy.ndarray)
        self.assertEqual((2, 3), result.shape)
        self.assertEqual(numpy.dtype(numpy.float32), result.dtype)

    def test_annotated_array_accepts_h5py_dataset_as_data(self) -> None:
        # h5py.Dataset satisfies ArrayProtocol: it exposes .shape, .dtype, and __array__.
        # AnnotatedArray must accept it without forcing an eager copy into memory.
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            ds = f.create_dataset("data", data=numpy.arange(6, dtype=numpy.float32).reshape(2, 3))
            group = AnnotatedArray.AxisGroup.from_2d_size((2, 3))
            annotated = AnnotatedArray.AnnotatedArray(ds, AnnotatedArray.ArrayDescriptor((group,)))

            # shape and dtype are read directly from the dataset without materialising it
            self.assertEqual((2, 3), annotated.data.shape)
            self.assertEqual(numpy.dtype(numpy.float32), annotated.data.dtype)

            # numpy functions work via __array__ (materialises on demand)
            result = numpy.asarray(annotated)
            self.assertIsInstance(result, numpy.ndarray)
            self.assertEqual((2, 3), result.shape)
            numpy.testing.assert_array_equal(result, numpy.arange(6, dtype=numpy.float32).reshape(2, 3))



if __name__ == "__main__":
    unittest.main()


