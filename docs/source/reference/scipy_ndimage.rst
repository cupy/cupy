.. module:: cupyx.scipy.ndimage

Multidimensional image processing (:mod:`cupyx.scipy.ndimage`)
==============================================================

.. Hint:: `SciPy API Reference: Multidimensional image processing (scipy.ndimage) <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_


Filters
-------

.. autosummary::
   :toctree: generated/

   convolve
   convolve1d
   correlate
   correlate1d
   gaussian_filter
   gaussian_filter1d
   gaussian_gradient_magnitude
   gaussian_laplace
   generic_filter
   generic_filter1d
   generic_gradient_magnitude
   generic_laplace
   laplace
   maximum_filter
   maximum_filter1d
   median_filter
   minimum_filter
   minimum_filter1d
   percentile_filter
   prewitt
   rank_filter
   sobel
   uniform_filter
   uniform_filter1d


Fourier filters
---------------

.. autosummary::
   :toctree: generated/

   fourier_ellipsoid
   fourier_gaussian
   fourier_shift
   fourier_uniform


Interpolation
-------------

.. autosummary::
   :toctree: generated/

   affine_transform
   map_coordinates
   rotate
   shift
   spline_filter
   spline_filter1d
   zoom


Measurements
------------

.. autosummary::
   :toctree: generated/

   center_of_mass
   extrema
   histogram
   label
   labeled_comprehension
   maximum
   maximum_position
   mean
   median
   minimum
   minimum_position
   standard_deviation
   sum_labels
   value_indices
   variance


Morphology
----------

.. autosummary::
   :toctree: generated/

   binary_closing
   binary_dilation
   binary_erosion
   binary_fill_holes
   binary_hit_or_miss
   binary_opening
   binary_propagation
   black_tophat
   distance_transform_edt
   generate_binary_structure
   grey_closing
   grey_dilation
   grey_erosion
   grey_opening
   iterate_structure
   morphological_gradient
   morphological_laplace
   white_tophat


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

