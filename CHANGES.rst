Changelog (niondata)
====================

UNRELEASED
----------

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
