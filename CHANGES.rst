Changelog (niondata)
====================

0.13.8 (Unreleased)
-------------------

- Added optional registration area bounds to align and register functions.

0.13.7 (2019-02-27)
-------------------

- Added mean function. Add keepdim param to mean/sum. Allow negative indices.

0.13.6 (2018-12-28)
-------------------

- Fix display RGB calculation on integer images.

- Add methods for better control of data ref count.

0.13.5 (2018-12-11)
-------------------

- Add setters for timezone, timezone_offset, and timestamp.

0.13.4 (2018-11-13)
-------------------

- Add measure_relative_translation function to xdata. Utilize in align.

- Generalize align and register sequence to accept any combo of sequence and collection dimensions.

- Provide more descriptive data dimensions string.

0.13.3 (2018-06-15)
-------------------

- Fix squeeze to not remove last datum dimension.

- Add re-dimension function (changes data description, keeps data layout in memory the same).

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
