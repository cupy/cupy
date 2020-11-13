----------------------------------
Multi-dimensional image processing
----------------------------------

CuPy provides multi-dimensional image processing functions.
It supports a subset of :mod:`scipy.ndimage` interface.

.. module:: cupyx.scipy.ndimage

.. https://docs.scipy.org/doc/scipy/reference/ndimage.html

Filters
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.convolve
   cupyx.scipy.ndimage.convolve1d
   cupyx.scipy.ndimage.correlate
   cupyx.scipy.ndimage.correlate1d
   cupyx.scipy.ndimage.gaussian_filter
   cupyx.scipy.ndimage.gaussian_filter1d
   cupyx.scipy.ndimage.gaussian_gradient_magnitude
   cupyx.scipy.ndimage.gaussian_laplace
   cupyx.scipy.ndimage.generic_filter
   cupyx.scipy.ndimage.generic_filter1d
   cupyx.scipy.ndimage.generic_gradient_magnitude
   cupyx.scipy.ndimage.generic_laplace
   cupyx.scipy.ndimage.laplace
   cupyx.scipy.ndimage.maximum_filter
   cupyx.scipy.ndimage.maximum_filter1d
   cupyx.scipy.ndimage.median_filter
   cupyx.scipy.ndimage.minimum_filter
   cupyx.scipy.ndimage.minimum_filter1d
   cupyx.scipy.ndimage.percentile_filter
   cupyx.scipy.ndimage.prewitt
   cupyx.scipy.ndimage.rank_filter
   cupyx.scipy.ndimage.sobel
   cupyx.scipy.ndimage.uniform_filter
   cupyx.scipy.ndimage.uniform_filter1d


Fourier Filters
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.fourier_gaussian
   cupyx.scipy.ndimage.fourier_shift
   cupyx.scipy.ndimage.fourier_uniform


Interpolation
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.affine_transform
   cupyx.scipy.ndimage.map_coordinates
   cupyx.scipy.ndimage.rotate
   cupyx.scipy.ndimage.shift
   cupyx.scipy.ndimage.spline_filter
   cupyx.scipy.ndimage.spline_filter1d
   cupyx.scipy.ndimage.zoom


Measurements
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.extrema
   cupyx.scipy.ndimage.label
   cupyx.scipy.ndimage.maximum
   cupyx.scipy.ndimage.maximum_position
   cupyx.scipy.ndimage.mean
   cupyx.scipy.ndimage.median
   cupyx.scipy.ndimage.minimum
   cupyx.scipy.ndimage.minimum_position
   cupyx.scipy.ndimage.standard_deviation
   cupyx.scipy.ndimage.sum
   cupyx.scipy.ndimage.variance


Morphology
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.binary_closing
   cupyx.scipy.ndimage.binary_dilation
   cupyx.scipy.ndimage.binary_erosion
   cupyx.scipy.ndimage.binary_fill_holes
   cupyx.scipy.ndimage.binary_hit_or_miss
   cupyx.scipy.ndimage.binary_opening
   cupyx.scipy.ndimage.binary_propagation
   cupyx.scipy.ndimage.black_tophat
   cupyx.scipy.ndimage.generate_binary_structure
   cupyx.scipy.ndimage.grey_closing
   cupyx.scipy.ndimage.grey_dilation
   cupyx.scipy.ndimage.grey_erosion
   cupyx.scipy.ndimage.grey_opening
   cupyx.scipy.ndimage.iterate_structure
   cupyx.scipy.ndimage.morphological_gradient
   cupyx.scipy.ndimage.morphological_laplace
   cupyx.scipy.ndimage.white_tophat


OpenCV mode
-----------
:mod:`cupyx.scipy.ndimage` supports additional mode, ``opencv``.
If it is given, the function performs like `cv2.warpAffine <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983>`_ or `cv2.resize <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d>`_. Example:


.. code:: python

   import cupyx.scipy.ndimage
   import cupy as cp
   import cv2

   im = cv2.imread('TODO') # pls fill in your image path

   trans_mat = cp.eye(4)
   trans_mat[0][0] = trans_mat[1][1] = 0.5

   smaller_shape = (im.shape[0] // 2, im.shape[1] // 2, 3)
   smaller = cp.zeros(smaller_shape) # preallocate memory for resized image

   cupyx.scipy.ndimage.affine_transform(im, trans_mat, output_shape=smaller_shape,
                                        output=smaller, mode='opencv')

   cv2.imwrite('smaller.jpg', cp.asnumpy(smaller)) # smaller image saved locally

