# -*- coding: utf-8 -*-

import setuptools
import os


setuptools.setup(
    name="niondata",
    version="0.13.5",
    author="Nion Software",
    author_email="swift@nion.com",
    description="A data processing library for Nion Swift.",
    long_description=open("README.rst").read(),
    url="https://github.com/nion-software/niondata",
    packages=["nion.data", "nion.data.test"],
    install_requires=['scipy', 'numpy', 'nionutils'],
    license='Apache 2.0',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
    ],
    include_package_data=True,
    test_suite="nion.data.test"
)
