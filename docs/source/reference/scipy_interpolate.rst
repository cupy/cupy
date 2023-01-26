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

.. autosummary::
   :toctree: generated/

   RBFInterpolator
   interpn
   RegularGridInterpolator

