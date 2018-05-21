# -*- coding: utf-8 -*-

import setuptools
import os

setuptools.setup(
    name="niondata",
    version="0.13.1",
    packages=["nion.data", "nion.data.test"],
    install_requires=['scipy', 'numpy', 'nionutils'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
    ],
    include_package_data=True,
    test_suite="nion.data.test"
)
