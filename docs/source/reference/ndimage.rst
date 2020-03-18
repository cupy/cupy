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


OpenCV mode
-----------
:mod:`cupyx.scipy.ndimage` supports additional mode, ``opencv``.
If it is given, the function performs like `cv2.warpAffine <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983>`_ or `cv2.resize <https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d>`_.
