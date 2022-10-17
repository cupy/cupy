.. module:: cupyx.scipy.spatial.distance

Distance computations (:mod:`cupyx.scipy.spatial.distance`)
===========================================================

.. note::

   The ``distance`` module uses ``pylibraft`` as a backend.
   You need to install `pylibraft package <https://anaconda.org/rapidsai/pylibraft>` from ``rapidsai`` Conda channel to use features listed on this page.

.. note::
   Currently, the ``distance`` module is not supported on AMD ROCm platforms.

.. Hint:: `SciPy API Reference: Spatial distance routines (scipy.spatial.distance) <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_


Distance matrix computations
----------------------------

Distance matrix computation from a collection of raw observation vectors stored in a rectangular array.

.. autosummary::
   :toctree: generated/

   pdist
   cdist
   distance_matrix


Distance functions
------------------

Distance functions between two numeric vectors `u` and `v`. Computing distances over a large collection of vectors is inefficient for these functions. Use `cdist` for this purpose.

.. autosummary::
   :toctree: generated/

   minkowski
   canberra
   chebyshev
   cityblock
   correlation
   cosine
   hamming
   euclidean
   jensenshannon
   russellrao
   sqeuclidean
   hellinger
   kl_divergence

