Installation Guide
==================

Requirements
------------

The following Linux distributions are recommended.

* `Ubuntu <https://www.ubuntu.com/>`_ 18.04 LTS (x86_64)
* `CentOS <https://www.centos.org/>`_ 7 (x86_64)

These components must be installed to use CuPy:

* `NVIDIA CUDA GPU <https://developer.nvidia.com/cuda-gpus>`_ with the Compute Capability 3.0 or larger.

* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_: v9.2 / v10.0 / v10.1 / v10.2 / v11.0 / v11.1

    * If you have multiple versions of CUDA Toolkit installed, CuPy will automatically choose one of the CUDA installations.
      See :ref:`install_cuda` for details.

* `Python <https://python.org/>`_: v3.6.0+ / v3.7.0+ / v3.8.0+

.. note::

   On Windows, CuPy only supports Python 3.6.0 or later.

Python Dependencies
~~~~~~~~~~~~~~~~~~~

NumPy/SciPy-compatible API in CuPy v8 is based on NumPy 1.19 and SciPy 1.5, and has been tested against the following versions:

* `NumPy <https://numpy.org/>`_: v1.17 / v1.18 / v1.19

* `SciPy <https://scipy.org/>`_ (*optional*): v1.4 / v1.5

    * Required only when using :doc:`reference/scipy` (``cupyx.scipy``).

* `Optuna <https://optuna.org/>`_ (*optional*): v2.x

    * Required only when using :doc:`reference/optimize`.

.. note::

   SciPy and Optuna are optional dependencies and will not be installed automatically.

.. note::

   Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

    $ python -m pip install -U setuptools pip

Additional CUDA Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~

Part of the CUDA features in CuPy will be activated only when the corresponding libraries are installed.

* `cuTENSOR <https://developer.nvidia.com/cutensor>`_: v1.2

    * The library to accelerate tensor operations. See :doc:`reference/environment` for the details.

* `NCCL <https://developer.nvidia.com/nccl>`_: v2.4 (CUDA 9.2) / v2.6 (CUDA 10.0) / v2.7 (CUDA 10.1+)

    * The library to perform collective multi-GPU / multi-node computations.

* `cuDNN <https://developer.nvidia.com/cudnn>`_: v7.6 (CUDA 9.2 & 10.x) / v8.0 (CUDA 10.1+)

    * The library to accelerate deep neural network computations.


Installing CuPy
---------------

Wheels (precompiled binary packages) are available for Linux and Windows.
Package names are different depending on your CUDA Toolkit version.

.. list-table::
   :header-rows: 1

   * - CUDA
     - Command
   * - v9.2
     - ``$ pip install cupy-cuda92``
   * - v10.0
     - ``$ pip install cupy-cuda100``
   * - v10.1
     - ``$ pip install cupy-cuda101``
   * - v10.2
     - ``$ pip install cupy-cuda102``
   * - v11.0
     - ``$ pip install cupy-cuda110``
   * - v11.1
     - ``$ pip install cupy-cuda111``

.. note::

   Wheel packages are built with NCCL (Linux only) and cuDNN support enabled.

   * NCCL library is bundled with these packages.
     You don't have to install it manually.

   * cuDNN library is bundled with these packages except for CUDA 10.1+.
     For CUDA 10.1+, you need to manually download and install cuDNN v8.0.x library to use cuDNN features.

.. note::

   Use ``pip install --pre cupy-cudaXXX`` if you want to install prerelease (development) versions.


When using wheels, please be careful not to install multiple CuPy packages at the same time.
Any of these packages and ``cupy`` package (source installation) conflict with each other.
Please make sure that only one CuPy package (``cupy`` or ``cupy-cudaXX`` where XX is a CUDA version) is installed::

  $ pip freeze | grep cupy


Installing CuPy from conda-forge
--------------------------------

Conda/Anaconda is a cross-platform package management solution widely used in scientific computing and other fields.
The above ``pip install`` instruction is compatible with ``conda`` environments. Alternatively, for Linux 64 systems
once the CUDA driver is correctly set up, you can install CuPy from the ``conda-forge`` channel::

    $ conda install -c conda-forge cupy

and ``conda`` will install pre-built CuPy and most of the optional dependencies for you, including CUDA runtime libraries
(``cudatoolkit``), NCCL, and cuDNN. It is not necessary to install CUDA Toolkit in advance. If you need to enforce
the installation of a particular CUDA version (say 10.0) for driver compatibility, you can do::

    $ conda install -c conda-forge cupy cudatoolkit=10.0

.. note::

    Currently cuTENSOR is not yet available on ``conda-forge``.

.. note::

    If you encounter any problem with CuPy from ``conda-forge``, please feel free to report to `cupy-feedstock
    <https://github.com/conda-forge/cupy-feedstock/issues>`_, and we will help investigate if it is just a packaging
    issue in ``conda-forge``'s recipe or a real issue in CuPy.

.. note::

    If you did not install CUDA Toolkit yourselves, the ``nvcc`` compiler might not be available.
    The ``cudatoolkit`` package from Anaconda does not have ``nvcc`` included.

.. _install_cupy_from_source:

Installing CuPy from Source
---------------------------

Use of wheel packages is recommended whenever possible.
However, if wheels cannot meet your requirements (e.g., you are running non-Linux environment or want to use a version of CUDA / cuDNN / NCCL not supported by wheels), you can also build CuPy from source.

.. note::

   CuPy source build requires ``g++-6`` or later.
   For Ubuntu 18.04, run ``apt-get install g++``.
   For Ubuntu 16.04, CentOS 6 or 7, follow the instructions :ref:`here <install_gcc6>`.

.. note::

   When installing CuPy from source, features provided by additional CUDA libraries will be disabled if these libraries are not available at the build time.
   See :ref:`install_cudnn` for the instructions.

.. note::

   If you upgrade or downgrade the version of CUDA Toolkit, cuDNN, NCCL or cuTENSOR, you may need to reinstall CuPy.
   See :ref:`install_reinstall` for details.

You can install the latest stable release version of the `CuPy source package <https://pypi.python.org/pypi/cupy>`_ via ``pip``.

::

  $ pip install cupy

If you want to install the latest development version of CuPy from a cloned Git repository::

  $ git clone --recursive https://github.com/cupy/cupy.git
  $ cd cupy
  $ pip install .

.. note::

   To build the source tree downloaded from GitHub, you need to install Cython 0.28.0 or later (``pip install cython``).
   You don't have to install Cython to build source packages hosted on PyPI as they include pre-generated C++ source files.


Uninstalling CuPy
-----------------

Use ``pip`` to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).

.. note::

   If CuPy is installed via ``conda``, please do ``conda uninstall cupy`` instead.


Upgrading CuPy
---------------

Just use ``pip install`` with ``-U`` option::

  $ pip install -U cupy

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


.. _install_reinstall:


Reinstalling CuPy
-----------------

To reinstall CuPy, please uninstall CuPy and then install it.
When reinstalling CuPy, we recommend using ``--no-cache-dir`` option as ``pip`` caches the previously built binaries::

  $ pip uninstall cupy
  $ pip install cupy --no-cache-dir

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


Using CuPy inside Docker
------------------------

We are providing the `official Docker images <https://hub.docker.com/r/cupy/cupy/>`_.
Use `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ docker run --gpus all -it cupy/cupy /bin/bash

Or run the interpreter directly::

  $ docker run --gpus all -it cupy/cupy /usr/bin/python


FAQ
---

.. _install_error:

``pip`` fails to install CuPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please make sure that you are using the latest ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip

Use ``-vvvv`` option with ``pip`` command.
This will display all logs of installation::

  $ pip install cupy -vvvv

If you are using ``sudo`` to install CuPy, note that ``sudo`` command does not propagate environment variables.
If you need to pass environment variable (e.g., ``CUDA_PATH``), you need to specify them inside ``sudo`` like this::

  $ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy

If you are using certain versions of conda, it may fail to build CuPy with error ``g++: error: unrecognized command line option ‘-R’``.
This is due to a bug in conda (see `conda/conda#6030 <https://github.com/conda/conda/issues/6030>`_ for details).
If you encounter this problem, please upgrade your conda.

.. _install_cudnn:

Installing cuDNN and NCCL
~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing cuDNN and NCCL using binary packages (i.e., using ``apt`` or ``yum``) provided by NVIDIA.

If you want to install tar-gz version of cuDNN and NCCL, we recommend installing it under the ``CUDA_PATH`` directory.
For example, if you are using Ubuntu, copy ``*.h`` files to ``include`` directory and ``*.so*`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.

If you want to use cuDNN or NCCL installed in another directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  $ export CFLAGS=-I/path/to/cudnn/include
  $ export LDFLAGS=-L/path/to/cudnn/lib
  $ export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

.. _install_cuda:

Working with Custom CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed CUDA on the non-default directory or multiple CUDA versions on the same host, you may need to manually specify the CUDA installation directory to be used by CuPy.

CuPy uses the first CUDA installation directory found by the following order.

#. ``CUDA_PATH`` environment variable.
#. The parent directory of ``nvcc`` command. CuPy looks for ``nvcc`` command from ``PATH`` environment variable.
#. ``/usr/local/cuda``

For example, you can build CuPy using non-default CUDA directory by ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy

.. note::

   CUDA installation discovery is also performed at runtime using the rule above.
   Depending on your system configuration, you may also need to set ``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64`` at runtime.

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CuPy raises a ``CompileException`` for almost everything, it is possible that CuPy cannot detect CUDA installed on your system correctly.
The followings are error messages commonly observed in such cases.

* ``nvrtc: error: failed to load builtins``
* ``catastrophic error: cannot open source file "cuda_fp16.h"``
* ``error: cannot overload functions distinguished by return type alone``
* ``error: identifier "__half_raw" is undefined``

Please try setting ``LD_LIBRARY_PATH`` and ``CUDA_PATH`` environment variable.
For example, if you have CUDA installed at ``/usr/local/cuda-9.2``::

  $ export CUDA_PATH=/usr/local/cuda-9.2
  $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

Also see :ref:`install_cuda`.

.. _install_gcc6:

Build fails on Ubuntu 16.04, CentOS 6 or 7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to build CuPy from source on systems with legacy GCC (g++-5 or earlier), you need to manually set up g++-6 or later and configure ``NVCC`` environment variable.

On Ubuntu 16.04::

  $ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  $ sudo apt update
  $ sudo apt install g++-6
  $ export NVCC="nvcc --compiler-bindir gcc-6"

On CentOS 6 / 7::

  $ sudo yum install centos-release-scl
  $ sudo yum install devtoolset-7-gcc-c++
  $ source /opt/rh/devtoolset-7/enable
  $ export NVCC="nvcc --compiler-bidir gcc-7"
