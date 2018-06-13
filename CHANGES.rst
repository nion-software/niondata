Changelog (niondata)
====================

UNRELEASED
----------

- Ensure that data_descriptor is a copy, not a reference, when accessed from DataAndMetadata.

- Add calibration and data_descriptor creation methods to xdata_1_0.

- Change crop to always produce the same size crop, even if out of bounds. Fill out of bounds with zero.

- Add crop_rotated to handle crop with rotation (slower).

0.13.2 (2018-05-23)
-------------------

- Automatically promote ndarray and constants (where possible) to xdata in operations.

- Fix FFT-1D scaling and shifting inconsistency.

- Add average_region function (similar to sum_region).

0.13.1 (2018-05-21)
-------------------

- Fix timezone bug.

0.13.0 (2018-05-10)
-------------------

- Initial version online.
