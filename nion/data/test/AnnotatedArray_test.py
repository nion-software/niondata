import datetime
import unittest
import zoneinfo

import numpy

from nion.data import AnnotatedArray


class TestAnnotatedArray(unittest.TestCase):

    def test_calibration_set_uses_explicit_calibration_accessors(self) -> None:
        primary = AnnotatedArray.AffineCalibration(unit="nm")
        alternate = AnnotatedArray.AffineCalibration(unit="rad")
        calibrations = AnnotatedArray.CalibrationSet.from_calibration(primary).with_calibration("alternate", alternate)

        self.assertTrue(calibrations.has_calibration("alternate"))
        self.assertIs(primary, calibrations.primary_calibration)
        self.assertIs(alternate, calibrations.get_calibration("alternate"))
        self.assertIs(alternate, calibrations.with_primary_calibration("alternate").primary_calibration)

    def test_array_descriptor_describes_shape_and_rank(self) -> None:
        collection = AnnotatedArray.BoundAxisGroup.from_1d_size(3)
        signal = AnnotatedArray.BoundAxisGroup.from_2d_size((4, 5))
        descriptor = AnnotatedArray.ArrayDescriptor((collection, signal))

        self.assertEqual((3, 4, 5), descriptor.shape)
        self.assertEqual(3, descriptor.ndim)

    def test_array_descriptor_requires_valid_axis_group_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            AnnotatedArray.ArrayDescriptor(())

        scalar_group = AnnotatedArray.BoundAxisGroup()
        vector_group = AnnotatedArray.BoundAxisGroup.from_1d_size(3)
        with self.assertRaisesRegex(ValueError, "Only the final"):
            AnnotatedArray.ArrayDescriptor((scalar_group, vector_group))

    def test_array_metadata_controls_extension_access(self) -> None:
        extension = AnnotatedArray.ExtensionRecord("org.nion.test", 1, b"value=42")
        metadata = AnnotatedArray.ArrayMetadata(extensions=(extension,))

        self.assertEqual("org.nion.test", extension.extension_type_id)
        self.assertEqual(("org.nion.test",), metadata.extension_type_ids)
        self.assertTrue(metadata.has_extension("org.nion.test"))
        self.assertIs(extension, metadata.get_extension("org.nion.test"))
        with self.assertRaises(KeyError):
            metadata.get_extension("org.nion.missing")

        replacement = AnnotatedArray.ExtensionRecord("org.nion.test", 2, b"value=43")
        replaced_metadata = metadata.with_extension(replacement)
        self.assertEqual(2, replaced_metadata.get_extension("org.nion.test").schema_version)
        self.assertEqual(1, metadata.get_extension("org.nion.test").schema_version)
        self.assertFalse(replaced_metadata.without_extension("org.nion.test").has_extension("org.nion.test"))

    def test_array_metadata_validates_extensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            AnnotatedArray.ExtensionRecord("", 1, b"")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            AnnotatedArray.ExtensionRecord("org.nion.test", 0, b"")
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            AnnotatedArray.ExtensionRecord("org.nion.test", 1, bytearray())  # type: ignore[arg-type]

        extension = AnnotatedArray.ExtensionRecord("org.nion.test", 1, b"")
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
        descriptor = AnnotatedArray.ArrayDescriptor((AnnotatedArray.BoundAxisGroup.from_2d_size((4, 5)),))
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
        descriptor = AnnotatedArray.ArrayDescriptor((AnnotatedArray.BoundAxisGroup.from_1d_size(3),))
        header = AnnotatedArray.ArrayHeader(descriptor, numpy.float32)

        with self.assertRaisesRegex(ValueError, "shape"):
            AnnotatedArray.AnnotatedArray(numpy.zeros((4,), dtype=numpy.float32), descriptor)
        with self.assertRaisesRegex(ValueError, "dtype"):
            AnnotatedArray.AnnotatedArray.from_header(numpy.zeros((3,), dtype=numpy.float64), header)

    def test_zeros_annotated_array_constructs_matching_header(self) -> None:
        group = AnnotatedArray.BoundAxisGroup.from_2d_size((2, 3), unit="nm")
        array = AnnotatedArray.zeros_annotated_array((group,), dtype=numpy.float32)

        self.assertEqual((2, 3), array.data.shape)
        self.assertEqual(numpy.dtype(numpy.float32), array.header.dtype)
        self.assertEqual((group,), array.descriptor.bound_axis_groups)
        self.assertEqual(2, len(array.get_flat_axis_calibrations()))


if __name__ == "__main__":
    unittest.main()


