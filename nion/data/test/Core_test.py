# standard libraries
import logging
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Context
from nion.data import DataAndMetadata


class TestCore(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_something(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.uint32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = ctx.line_profile(a, ctx.vector(ctx.normalized_point(0.25, 0.25), ctx.normalized_point(0.5, 0.5)), 2)
        # print(b.data)

    def test_something_else(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.int32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = ctx.amin(a) - a
        # print(a.data)
        # print(b.data_shape)
        # print(b.data)

    def test_something_again(self):
        ctx = Context.context()
        src_data = ((numpy.abs(numpy.random.randn(12, 8)) + 1) * 10).astype(numpy.uint32)
        a = DataAndMetadata.DataAndMetadata.from_data(src_data)
        b = a - ctx.amin(a)
        # print(a.data)
        # print(b.data_shape)
        # print(b.data)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
