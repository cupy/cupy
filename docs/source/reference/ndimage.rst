----------------------------------
Multi-dimensional image processing
----------------------------------

CuPy provides multi-dimensional image processing functions.
It supports a subset of :mod:`scipy.ndimage` interface.

.. module:: cupyx.scipy.ndimage

.. https://docs.scipy.org/doc/scipy/reference/ndimage.html

Interpolation
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.affine_transform
   cupyx.scipy.ndimage.convolve
   cupyx.scipy.ndimage.correlate
   cupyx.scipy.ndimage.map_coordinates
   cupyx.scipy.ndimage.rotate
   cupyx.scipy.ndimage.shift
   cupyx.scipy.ndimage.zoom


Measurements
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.ndimage.label
   cupyx.scipy.ndimage.mean
   cupyx.scipy.ndimage.standard_deviation
   cupyx.scipy.ndimage.sum
   cupyx.scipy.ndimage.variance


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

