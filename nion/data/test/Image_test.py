# standard libraries
import os
import shutil
import typing
import unittest

# third party libraries
import h5py
import numpy
import numpy.typing

# local libraries
from nion.data import Image

_ImageDataType = Image._ImageDataType



class TestImageClass(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_create_rgba_image_from_array(self) -> None:
        image_1d_16: numpy.typing.NDArray[numpy.double] = numpy.zeros((16, ), dtype=numpy.double)
        image_1d_16x1: numpy.typing.NDArray[numpy.double] = numpy.zeros((16, 1), dtype=numpy.double)
        self.assertIsNotNone(Image.create_rgba_image_from_array(image_1d_16))
        self.assertIsNotNone(Image.create_rgba_image_from_array(image_1d_16x1))
        image_1d_rgb: numpy.typing.NDArray[numpy.uint8] = numpy.zeros((16, 3), dtype=numpy.uint8)
        self.assertIsNotNone(Image.create_rgba_image_from_array(image_1d_rgb))

    def test_rebin_expand_has_even_expansion(self) -> None:
        # NOTE: statistical tests are only valid if expanded length is multiple of src length
        src = numpy.arange(0, 10)
        expanded = Image.rebin_1d(src, 50)
        self.assertAlmostEqual(numpy.mean(src).item(), numpy.mean(expanded).item())
        self.assertAlmostEqual(numpy.var(src).item(), numpy.var(expanded).item())
        src = numpy.arange(0, 10)
        expanded = Image.rebin_1d(src, 500)
        self.assertAlmostEqual(numpy.mean(src).item(), numpy.mean(expanded).item())
        self.assertAlmostEqual(numpy.var(src).item(), numpy.var(expanded).item())
        # test larger values to make sure linear mapping works (failed once)
        src = numpy.arange(0, 200)
        expanded = Image.rebin_1d(src, 600)
        self.assertAlmostEqual(numpy.mean(src).item(), numpy.mean(expanded).item())
        self.assertAlmostEqual(numpy.var(src).item(), numpy.var(expanded).item())

    def test_scale_cubic_is_symmetry(self) -> None:
        src1 = numpy.zeros((8, 8))
        src2 = numpy.zeros((9, 9))
        src1[3:5, 3:5] = 1
        src2[3:6, 3:6] = 1
        src1s: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (12, 12), 'cubic')*1000).astype(numpy.int32)
        src2s: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (12, 12), 'cubic')*1000).astype(numpy.int32)
        src1t: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (13, 13), 'cubic')*1000).astype(numpy.int32)
        src2t: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (13, 13), 'cubic')*1000).astype(numpy.int32)
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[0:6, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[12:5:-1, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[12:5:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[0:6, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[12:5:-1, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[12:5:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[0:6, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[13:6:-1, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[13:6:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[0:6, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[13:6:-1, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[13:6:-1, 0:6]))

    def test_scale_linear_is_symmetry(self) -> None:
        src1 = numpy.zeros((8, 8))
        src2 = numpy.zeros((9, 9))
        src1[3:5, 3:5] = 1
        src2[3:6, 3:6] = 1
        src1s: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (12, 12), 'linear')*1000).astype(numpy.int32)
        src2s: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (12, 12), 'linear')*1000).astype(numpy.int32)
        src1t: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (13, 13), 'linear')*1000).astype(numpy.int32)
        src2t: _ImageDataType = typing.cast(_ImageDataType, Image.scaled(src1, (13, 13), 'linear')*1000).astype(numpy.int32)
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[0:6, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[12:5:-1, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src1s[0:6, 0:6], src1s[12:5:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[0:6, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[12:5:-1, 12:5:-1]))
        self.assertTrue(numpy.array_equal(src2s[0:6, 0:6], src2s[12:5:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[0:6, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[13:6:-1, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src1t[0:6, 0:6], src1t[13:6:-1, 0:6]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[0:6, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[13:6:-1, 13:6:-1]))
        self.assertTrue(numpy.array_equal(src2t[0:6, 0:6], src2t[13:6:-1, 0:6]))

    def test_rgba_can_be_created_from_h5py_array(self) -> None:
        current_working_directory = os.getcwd()
        workspace_dir = os.path.join(current_working_directory, "__Test")
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
        os.makedirs(workspace_dir)
        try:
            with h5py.File(os.path.join(workspace_dir, "file.h5"), "w") as f:
                dataset = f.create_dataset("data", data=numpy.ones((4, 4, 4), dtype=numpy.uint8))
                Image.create_rgba_image_from_array(dataset)
        finally:
            # print(f"rmtree {workspace_dir}")
            shutil.rmtree(workspace_dir)
