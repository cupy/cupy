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
   cupyx.scipy.ndimage.map_coordinates
   cupyx.scipy.ndimage.rotate
   cupyx.scipy.ndimage.shift
   cupyx.scipy.ndimage.zoom


OpenCV mode
-----------
:mod:`cupyx.scipy.ndimage` supports additional mode, ``opencv``.
If it is given, the function performs like `cv2.warpAffine <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983>`_ or `cv2.resize <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d>`_.

Example to resize image to half size:
```
import cupyx.scipy.ndimage
import cupy as cp
import cv2

im = cv2.imread('YOUR_IMAGE_PATH')

M = cp.eye(4)
M[0][0] = M[1][1] = 0.5

smaller_shape = (im.shape[0]/2, im.shape[1]/2, 3)
smaller = cp.zeros(smaller_shape) # preallocate memory for resized image
smaller = cupyx.scipy.ndimage.affine_transform(im, M, output_shape=smaller_shape, output=smaller, mode='opencv')
cv2.imwrite('smaller.jpg', cp.asnumpy(smaller)) # smaller image should be saved to your current path
```
