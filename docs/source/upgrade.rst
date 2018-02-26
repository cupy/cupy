.. currentmodule:: cupy

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.
Most changes are carefully designed not to break existing code; however changes that may possibly break them are highlighted with a box.


CuPy v2
=======

Changed Behavior of count_nonzero Function
------------------------------------------

For performance reasons, :func:`cupy.count_nonzero` has been changed to return zero-dimensional :class:`ndarray` instead of `int` when `axis=None`.
See the discussion in `#154 <https://github.com/cupy/cupy/pull/154>`_ for more details.
