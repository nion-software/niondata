# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 18:26:28 2014

Implementation of algorithm from:
   Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
   "Efficient subpixel image registration algorithms," Opt. Lett. 33,
   156-158 (2008).

Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

@author: Michael Sarahan
"""

import numpy as np

def dftregistration(data1, data2, upsample_factor=1):
    """
    Efficient subpixel image registration by crosscorrelation. This code
    gives the same precision as the FFT upsampled cross correlation in a
    small fraction of the computation time and with reduced memory
    requirements. It obtains an initial estimate of the crosscorrelation peak
    by an FFT and then refines the shift estimation by upsampling the DFT
    only in a small neighborhood of that estimate by means of a
    matrix-multiply DFT. With this procedure all the image points are used to
    compute the upsampled crosscorrelation.

    Parameters
    ----------
    data1 : array_like
            Reference image (real space)
    data2 : array_like
            Image to register (real space)
    upsample_factor : int
            Upsampling factor. Images will be registered to
              within 1/upsample_factor of a pixel. For example
              upsample_factor = 20 means the images will be registered
              within 1/20 of a pixel. (default = 1)

    Result
    row_shift : float
            Y-shift required to align images (in pixels)
    col_shift : float
            X-shift required to align images (in pixels)
    """
    data1 = np.fft.fft2(data1)
    data2 = np.fft.fft2(data2)

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the peak
    rows, cols = data1.shape
    cross_correlation = np.fft.ifft2(data1*data2.conj())
    # Locate maximum
    row_max, col_max = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    mid_row = np.fix(rows/2)
    mid_col = np.fix(cols/2)
    if row_max > mid_row:
        row_shift = row_max - rows
    else:
        row_shift = row_max
    if col_max > mid_col:
        col_shift = col_max - cols
    else:
        col_shift = col_max
    if upsample_factor == 1:
        return row_shift, col_shift
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # %%% DFT computation %%%
        # % Initial shift estimate in upsampled grid
        row_shift = round(row_shift*upsample_factor)/upsample_factor
        col_shift = round(col_shift*upsample_factor)/upsample_factor
        upsampled_pixels = np.ceil(upsample_factor*1.5)
        dftshift = np.fix(upsampled_pixels/2) # Center of output array at dftshift+1
        # Matrix multiply DFT around the current shift estimate
        cross_correlation = dftups(data2*data1.conj(), upsampled_pixels, upsampled_pixels, upsample_factor,
                    dftshift-row_shift*upsample_factor, dftshift-col_shift*upsample_factor).conj()/(max(mid_row,1)*max(mid_col,1)*max(upsample_factor,1)**2)
        # Locate maximum and map back to original pixel grid
        row_max, col_max = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
        row_max = row_max - dftshift
        col_max = col_max - dftshift
        row_shift = row_shift + row_max/upsample_factor
        col_shift = col_shift + col_max/upsample_factor

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    if mid_row == 1:
        row_shift = 0
    if mid_col == 1:
        col_shift = 0
    # output=[error,diffphase,row_shift,col_shift];
    return row_shift, col_shift

def dftups(data, upsampled_rows=None, upsampled_cols=None, upsample_factor=1, row_offset=0, col_offset=0):
    """
    Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    a small region.

    This code is intended to provide the same result as if the following
    operations were performed
      - Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      - Take the FFT of the larger array
      - Extract an [upsampled_rows, upsampled_cols] region of the result. Starting with the
        [row_offset+1 col_offset+1] element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if (upsampled_rows, upsampled_cols) are much
    smaller than (rows*upsample_factor, cols*upsample_factor)

    Parameters
    ----------
    data : array_like, real
           The input data array (DFT of original data) to upsample
    upsampled_rows : integer or None
           The row size of the region to be sampled
    upsampled_cols : integer or None
           The column size of the region to be sampled
    upsample_factor : integer
           The upsampling factor
    row_offset : int, optional, default is image center
            The row offset of the region to be sampled
    col_offset : int, optional, default is image center
            The column offset of the region to be sampled

    Returns
    -------
    array_like
            The upsampled DFT of the specified region
    """
    rows, cols = data.shape
    if upsampled_rows is None:
        upsampled_rows = rows
    if upsampled_cols is None:
        upsampled_cols = cols
    # Compute kernels and obtain DFT by matrix products
    col_kernel = np.exp((-1j*2*np.pi/(cols*upsample_factor)) * (np.fft.ifftshift(np.arange(cols))[:, np.newaxis] - np.floor(cols/2)).dot(np.arange(upsampled_cols)[np.newaxis, :] - col_offset))
    row_kernel = np.exp((-1j*2*np.pi/(rows*upsample_factor)) * (np.arange(upsampled_rows)[:, np.newaxis] - row_offset).dot(np.fft.ifftshift(np.arange(rows))[np.newaxis, :] - np.floor(rows/2)))
    return row_kernel.dot(data).dot(col_kernel)

