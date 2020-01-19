[Experimental] Installation Guide for ROCm environemt
=====================================================

.. contents:: :local:

This is an experimental feature. We recommend only for advanced users to use this.

Recommended Environments
------------------------

We recommend the following Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`_ 16.04 / 18.04 LTS (64-bit)


Requirements
------------

You need to have the following components to use CuPy.

* GPU supported by ROCm (AMD GPUs or NVIDIA GPUs)
* `ROCm <https://rocm.github.io/install.html>`_
    * Supported Versions: ROCm 2.6+.
* `Python <https://python.org/>`_
* `NumPy <http://www.numpy.org/>`_

And please install ROCm libraries.

::

  $ sudo apt install hipblas hipsparse rocrand rocthrust


Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip


.. _install_hip:

Install CuPy from Source
------------------------

It is recommended to use wheels whenever possible.
However, there is currently no wheels for the ROCm environment, so you have to build it from source.

When installing from source, C++ compiler such as ``g++`` is required.
You need to install it before installing CuPy.
This is typical installation method for each platform::

  # Ubuntu 16.04
  $ apt-get install g++

.. note::

   If you want to upgrade or downgrade the version of ROCm, you may need to reinstall CuPy after that.
   See :ref:`rocm_install_reinstall` for details.

Using pip
~~~~~~~~~

You can install `CuPy package <https://pypi.python.org/pypi/cupy>`_ via ``pip``.
It builds CuPy from source.

::

  $ export HCC_AMDGPU_TARGET=gfx900  # This value should be changed based on your GPU
  $ export __HIP_PLATFORM_HCC__
  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install cupy

Using Tarball
~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download cupy`` or from `the release notes page <https://github.com/cupy/cupy/releases>`_.
You can install CuPy from the tarball::

  $ pip install cupy-x.x.x.tar.gz

You can also install the development version of CuPy from a cloned Git repository::

  $ git clone https://github.com/cupy/cupy.git
  $ cd cupy
  $ export HCC_AMDGPU_TARGET=gfx900  # This value should be changed based on your GPU
  $ export __HIP_PLATFORM_HCC__
  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install .

If you are using the source tree downloaded from GitHub, you need to install Cython 0.28.0 or later (``pip install cython``).

Uninstall CuPy
--------------

Use pip to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   When you upgrade Chainer, ``pip`` sometimes installs the new version without removing the old one in ``site-packages``.
   In this case, ``pip uninstall`` only removes the latest one.
   To ensure that CuPy is completely removed, run the above command repeatedly until ``pip`` returns an error.

Upgrade CuPy
------------

Just use ``pip install`` with ``-U`` option::

  $ export HCC_AMDGPU_TARGET=gfx900  # This value should be changed based on your GPU
  $ export __HIP_PLATFORM_HCC__
  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install -U cupy

.. _rocm_install_reinstall:

Reinstall CuPy
--------------

If you want to reinstall CuPy, please uninstall CuPy first, and then install again.
When reinstalling CuPy, we recommend to use ``--no-cache-dir`` option as ``pip`` caches the previously built binaries::

  $ pip uninstall cupy
  $ export HCC_AMDGPU_TARGET=gfx900  # This value should be changed based on your GPU
  $ export __HIP_PLATFORM_HCC__
  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install cupy --no-cache-dir

FAQ
---

.. _rocm_install_error:

``pip`` fails to install CuPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please make sure that you are using the latest ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip

Use ``-vvvv`` option with ``pip`` command to investigate the details of errors.
This will display all logs of installation::

  $ pip install cupy -vvvv

If you are using ``sudo`` to install CuPy, note that ``sudo`` command does not propagate environment variables.
If you need to pass environment variable (e.g., ``ROCM_HOME``), you need to specify them inside ``sudo`` like this::

  $ sudo ROCM_HOME=/opt/rocm pip install cupy

If you are using certain versions of conda, it may fail to build CuPy with error ``g++: error: unrecognized command line option ‘-R’``.
This is due to a bug in conda (see `conda/conda#6030 <https://github.com/conda/conda/issues/6030>`_ for details).
If you encounter this problem, please downgrade or upgrade it.
