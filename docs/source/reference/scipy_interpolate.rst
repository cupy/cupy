.. module:: cupyx.scipy.interpolate

Interpolation (:mod:`cupyx.scipy.interpolate`)
==============================================

.. Hint:: `SciPy API Reference: Interpolation functions (scipy.interpolate) <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_

Univariate interpolation
------------------------

.. autosummary::
   :toctree: generated/

   BarycentricInterpolator
   KroghInterpolator
   barycentric_interpolate
   krogh_interpolate
   pchip_interpolate
   CubicHermiteSpline
   PchipInterpolator
   Akima1DInterpolator
   PPoly
   BPoly

1-D Splines
-----------

.. autosummary::
   :toctree: generated/

   BSpline
   make_interp_spline

   splder
   splantider


Multivariate interpolation
--------------------------

Unstructured data:

.. autosummary::
   :toctree: generated/

   RBFInterpolator


For data on a grid:

.. autosummary::
   :toctree: generated/

   interpn
   RegularGridInterpolator

Tensor product polynomials:

.. autosummary::
   :toctree: generated/

   NdPPoly
