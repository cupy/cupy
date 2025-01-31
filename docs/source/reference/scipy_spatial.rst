.. module:: cupyx.scipy.spatial

Spatial algorithms and data structures  (:mod:`cupyx.scipy.spatial`)
====================================================================

.. Hint:: `SciPy API Reference: Spatial (scipy.spatial) <https://docs.scipy.org/doc/scipy/reference/spatial.html>`_

.. note::

   The ``spatial`` module uses the ``cuVS`` library as a backend.
   You need to install `cuVS package <https://anaconda.org/rapidsai/cuvs>` from ``rapidsai`` Conda channel to use features listed on this page.

.. note::
   Currently, the ``spatial`` module is not supported on AMD ROCm platforms.


Nearest-neighbor queries
------------------------

.. autosummary::
   :toctree: generated/

   KDTree


Delaunay triangulation
----------------------

.. autosummary::
   :toctree: generated/

   Delaunay


Functions
---------

.. autosummary::
   :toctree: generated/

    distance_matrix
