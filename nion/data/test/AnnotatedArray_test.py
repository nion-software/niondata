import datetime
import io
import typing
import unittest
import zoneinfo

import h5py
import numpy

from nion.data import AnnotatedArray
from nion.data import Calibration
from nion.data import DataAndMetadata


class TestAnnotatedArray(unittest.TestCase):

    _descriptor_variants: tuple[tuple[bool, int, int], ...] = (
        (False, 0, 1),
        (False, 0, 2),
        (False, 1, 1),
        (False, 1, 2),
        (False, 2, 1),
        (False, 2, 2),
        (True, 0, 1),
        (True, 0, 2),
        (True, 1, 1),
        (True, 1, 2),
        (True, 2, 1),
        (True, 2, 2),
    )

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

    def test_array_metadata_default_created_has_iana_timezone(self) -> None:
        # tzlocal.get_localzone() returns a zoneinfo.ZoneInfo with a proper IANA key
        # (e.g. "America/New_York").  A plain datetime.timezone fixed-offset object
        # would not satisfy the isinstance check and would have no .key attribute.
        metadata = AnnotatedArray.ArrayMetadata()

        self.assertIsInstance(metadata.created.tzinfo, zoneinfo.ZoneInfo)
        self.assertIsNotNone(metadata.created.tzinfo.key)  # type: ignore[union-attr]

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

    def test_from_data_and_metadata_covers_sequence_collection_and_datum_variations(self) -> None:
        def flatten_affine_calibrations(annotated_array: AnnotatedArray.AnnotatedArray) -> tuple[AnnotatedArray.AffineCalibration, ...]:
            calibrations = list[AnnotatedArray.AffineCalibration]()
            for axis_group in annotated_array.descriptor.axis_groups:
                for axis_index in range(axis_group.rank):
                    calibration = axis_group.get_calibration(axis_index) if axis_group.primary_calibration_key else AnnotatedArray.AffineCalibration()
                    calibrations.append(typing.cast(AnnotatedArray.AffineCalibration, calibration))
            return tuple(calibrations)

        for is_sequence, collection_rank, datum_rank in self._descriptor_variants:
            with self.subTest(is_sequence=is_sequence, collection_rank=collection_rank, datum_rank=datum_rank):
                total_rank = (1 if is_sequence else 0) + collection_rank + datum_rank
                shape = tuple(range(2, 2 + total_rank))
                data = numpy.arange(int(numpy.prod(shape)), dtype=numpy.float32).reshape(shape)
                dimensional_calibrations = tuple(Calibration.Calibration(offset=i + 0.25, scale=i + 1.5, units=f"u{i}") for i in range(total_rank))
                xdata = DataAndMetadata.new_data_and_metadata(
                    data=data,
                    intensity_calibration=Calibration.Calibration(offset=7.0, scale=3.0, units="counts"),
                    dimensional_calibrations=dimensional_calibrations,
                    metadata={"descriptor_variant": f"legacy-{int(is_sequence)}-{collection_rank}-{datum_rank}"},
                    data_descriptor=DataAndMetadata.DataDescriptor(is_sequence, collection_rank, datum_rank),
                    timestamp=datetime.datetime(2026, 7, 16, 19, 0, 0),
                    timezone="America/Los_Angeles",
                    timezone_offset="-0700",
                )

                annotated = AnnotatedArray.from_data_and_metadata(xdata)
                expected_axis_group_ranks = (1, datum_rank) if is_sequence and collection_rank == 0 else (1, collection_rank, datum_rank) if is_sequence else (datum_rank,) if collection_rank == 0 else (collection_rank, datum_rank)
                self.assertEqual(expected_axis_group_ranks, tuple(axis_group.rank for axis_group in annotated.descriptor.axis_groups))
                self.assertEqual(xdata.data_shape, annotated.descriptor.shape)
                self.assertEqual("counts", typing.cast(AnnotatedArray.AffineCalibration, annotated.get_intensity_calibration()).unit)

                flattened_annotated_calibrations = flatten_affine_calibrations(annotated)
                self.assertEqual(len(xdata.dimensional_calibrations), len(flattened_annotated_calibrations))
                for i, (legacy_calibration, annotated_calibration) in enumerate(zip(xdata.dimensional_calibrations, flattened_annotated_calibrations)):
                    with self.subTest(calibration_index=i):
                        self.assertEqual(legacy_calibration.offset, annotated_calibration.offset)
                        self.assertEqual(legacy_calibration.scale, annotated_calibration.scale)
                        self.assertEqual(legacy_calibration.units, annotated_calibration.unit)

                self.assertEqual(xdata.metadata["descriptor_variant"], annotated.metadata.attributes["descriptor_variant"])

    def test_to_data_and_metadata_covers_sequence_collection_and_datum_variations(self) -> None:
        def flatten_affine_calibrations(annotated_array: AnnotatedArray.AnnotatedArray) -> tuple[AnnotatedArray.AffineCalibration, ...]:
            calibrations = list[AnnotatedArray.AffineCalibration]()
            for axis_group in annotated_array.descriptor.axis_groups:
                for axis_index in range(axis_group.rank):
                    calibration = axis_group.get_calibration(axis_index) if axis_group.primary_calibration_key else AnnotatedArray.AffineCalibration()
                    calibrations.append(typing.cast(AnnotatedArray.AffineCalibration, calibration))
            return tuple(calibrations)

        for is_sequence, collection_rank, datum_rank in self._descriptor_variants:
            with self.subTest(is_sequence=is_sequence, collection_rank=collection_rank, datum_rank=datum_rank):
                if is_sequence and collection_rank == 0:
                    # (1, datum_rank) is ambiguous in legacy form and round-trips as non-sequence collection rank 1.
                    continue

                expected_axis_group_ranks = (1, collection_rank, datum_rank) if is_sequence else (datum_rank,) if collection_rank == 0 else (collection_rank, datum_rank)
                dim_sizes = tuple(range(2, 2 + sum(expected_axis_group_ranks)))

                dim_index = 0
                axis_groups = list[AnnotatedArray.AxisGroup]()
                for axis_group_rank in expected_axis_group_ranks:
                    axes = list[AnnotatedArray.Axis]()
                    calibrations = list[AnnotatedArray.AffineCalibration]()
                    for _ in range(axis_group_rank):
                        size = dim_sizes[dim_index]
                        calibration_index = dim_index
                        axes.append(AnnotatedArray.Axis(label=f"a{calibration_index}", size=size))
                        calibrations.append(AnnotatedArray.AffineCalibration(offset=calibration_index + 0.5, scale=calibration_index + 1.25, unit=f"ua{calibration_index}"))
                        dim_index += 1
                    axis_groups.append(
                        AnnotatedArray.AxisGroup(
                            axes=tuple(axes),
                            coordinate_calibrations={"calibrated": AnnotatedArray.CoordinateCalibration(calibrations=tuple(calibrations))},
                            primary_calibration_key="calibrated",
                        )
                    )

                descriptor = AnnotatedArray.ArrayDescriptor(
                    axis_groups=tuple(axis_groups),
                    intensity_calibrations=AnnotatedArray.CalibrationSet.from_calibration(
                        AnnotatedArray.AffineCalibration(offset=4.0, scale=2.0, unit="counts"),
                        "calibrated",
                    ),
                )
                metadata = AnnotatedArray.ArrayMetadata(
                    created=datetime.datetime(2026, 7, 16, 12, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles")),
                    attributes={"descriptor_variant": f"annotated-{int(is_sequence)}-{collection_rank}-{datum_rank}"},
                )
                annotated = AnnotatedArray.AnnotatedArray(
                    data=numpy.arange(int(numpy.prod(dim_sizes)), dtype=numpy.float32).reshape(dim_sizes),
                    descriptor=descriptor,
                    metadata=metadata,
                )

                xdata = AnnotatedArray.to_data_and_metadata(annotated)
                self.assertEqual(is_sequence, xdata.is_sequence)
                self.assertEqual(collection_rank, xdata.collection_dimension_count)
                self.assertEqual(datum_rank, xdata.datum_dimension_count)
                self.assertEqual(annotated.descriptor.shape, xdata.data_shape)
                self.assertEqual("counts", xdata.intensity_calibration.units)

                flattened_annotated_calibrations = flatten_affine_calibrations(annotated)
                self.assertEqual(len(flattened_annotated_calibrations), len(xdata.dimensional_calibrations))
                for i, (annotated_calibration, legacy_calibration) in enumerate(zip(flattened_annotated_calibrations, xdata.dimensional_calibrations)):
                    with self.subTest(calibration_index=i):
                        self.assertEqual(annotated_calibration.offset, legacy_calibration.offset)
                        self.assertEqual(annotated_calibration.scale, legacy_calibration.scale)
                        self.assertEqual(annotated_calibration.unit, legacy_calibration.units)

                round_tripped = AnnotatedArray.from_data_and_metadata(xdata)
                self.assertEqual(tuple(axis_group.rank for axis_group in annotated.descriptor.axis_groups), tuple(axis_group.rank for axis_group in round_tripped.descriptor.axis_groups))
                self.assertEqual(annotated.descriptor.shape, round_tripped.descriptor.shape)
                self.assertEqual(annotated.metadata.attributes["descriptor_variant"], round_tripped.metadata.attributes["descriptor_variant"])
                numpy.testing.assert_array_equal(numpy.asarray(annotated.data), numpy.asarray(round_tripped.data))

    def test_data_and_metadata_timezone_round_trip_through_annotated_array(self) -> None:
        xdata = DataAndMetadata.new_data_and_metadata(
            data=numpy.arange(6, dtype=numpy.float32).reshape(2, 3),
            intensity_calibration=Calibration.Calibration(offset=7.0, scale=3.0, units="counts"),
            dimensional_calibrations=(
                Calibration.Calibration(offset=0.25, scale=1.5, units="u0"),
                Calibration.Calibration(offset=1.25, scale=2.5, units="u1"),
            ),
            metadata={"descriptor_variant": "legacy-0-1-1"},
            data_descriptor=DataAndMetadata.DataDescriptor(False, 1, 1),
            timestamp=datetime.datetime(2026, 7, 16, 19, 0, 0),
            timezone="America/Los_Angeles",
            timezone_offset="-0700",
        )

        annotated = AnnotatedArray.from_data_and_metadata(xdata)
        self.assertEqual("America/Los_Angeles", getattr(annotated.metadata.created.tzinfo, "key", None))
        self.assertEqual(datetime.timedelta(hours=-7), annotated.metadata.created.utcoffset())

        round_tripped = AnnotatedArray.to_data_and_metadata(annotated)
        self.assertEqual(xdata.timestamp, round_tripped.timestamp)
        self.assertEqual("America/Los_Angeles", round_tripped.timezone)
        self.assertEqual("-0700", round_tripped.timezone_offset)



if __name__ == "__main__":
    unittest.main()


