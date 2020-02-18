Installation Guide
==================

.. contents:: :local:

Recommended Environments
------------------------

We recommend the following Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`_ 16.04 / 18.04 LTS (64-bit)
* `CentOS <https://www.centos.org/>`_ 7 (64-bit)

.. note::

   We are automatically testing CuPy on all the recommended environments above.
   We cannot guarantee that CuPy works on other environments including Windows and macOS, even if CuPy may seem to be running correctly.


Requirements
------------

You need to have the following components to use CuPy.

* `NVIDIA CUDA GPU <https://developer.nvidia.com/cuda-gpus>`_
    * Compute Capability of the GPU must be at least 3.0.
* `CUDA Toolkit <https://developer.nvidia.com/cuda-zone>`_
    * Supported Versions: 8.0, 9.0, 9.1, 9.2, 10.0, 10.1 and 10.2
    * If you have multiple versions of CUDA Toolkit installed, CuPy will choose one of the CUDA installations automatically.
      See :ref:`install_cuda` for details.
* `Python <https://python.org/>`_
    * Supported Versions: 3.5.1+, 3.6.0+, 3.7.0+ and 3.8.0+.
* `NumPy <http://www.numpy.org/>`_
    * Supported Versions: 1.15, 1.16, 1.17 and 1.18.
    * NumPy will be installed automatically during the installation of CuPy.

Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip

.. note::

   On Windows, CuPy only supports Python 3.6.0 or later.

.. note::

   Python 2 is not supported in CuPy v7.x releases.
   Please consider migrating Python 3 or use CuPy v6.x, which is the last version that supports Python 2.

Optional Libraries
~~~~~~~~~~~~~~~~~~

Some features in CuPy will only be enabled if the corresponding libraries are installed.

* `cuDNN <https://developer.nvidia.com/cudnn>`_ (library to accelerate deep neural network computations)
    * Supported Versions: v5, v5.1, v6, v7, v7.1, v7.2, v7.3, v7.4 and v7.5.
* `NCCL <https://developer.nvidia.com/nccl>`_  (library to perform collective multi-GPU / multi-node computations)
    * Supported Versions: v1.3.4, v2, v2.1, v2.2, v2.3 and v2.4.
* `cuTENSOR <https://developer.nvidia.com/cuda-math-library-early-access-program-page>`_ (library for high-performance tensor operations)
    * Supported Versions: v1.0.0 (experimental)


Install CuPy
------------

Wheels (precompiled binary packages) are available for Linux (Python 2.7 or later) and Windows (Python 3.6 or later).
Package names are different depending on the CUDA version you have installed on your host.

::

  (For CUDA 8.0)
  $ pip install cupy-cuda80

  (For CUDA 9.0)
  $ pip install cupy-cuda90

  (For CUDA 9.1)
  $ pip install cupy-cuda91

  (For CUDA 9.2)
  $ pip install cupy-cuda92

  (For CUDA 10.0)
  $ pip install cupy-cuda100

  (For CUDA 10.1)
  $ pip install cupy-cuda101

  (For CUDA 10.2)
  $ pip install cupy-cuda102

.. note::

   The latest version of cuDNN and NCCL libraries are included in these wheels.
   You don't have to install them manually.

When using wheels, please be careful not to install multiple CuPy packages at the same time.
Any of these packages and ``cupy`` package (source installation) conflict with each other.
Please make sure that only one CuPy package (``cupy`` or ``cupy-cudaXX`` where XX is a CUDA version) is installed::

  $ pip freeze | grep cupy


Install CuPy from Source
------------------------

It is recommended to use wheels whenever possible.
However, if wheels cannot meet your requirements (e.g., you are running non-Linux environment or want to use a version of CUDA / cuDNN / NCCL not supported by wheels), you can also build CuPy from source.

When installing from source, C++ compiler such as ``g++`` is required.
You need to install it before installing CuPy.
This is typical installation method for each platform::

  # Ubuntu 16.04
  $ apt-get install g++

  # CentOS 7
  $ yum install gcc-c++

.. note::

   When installing CuPy from source, features provided by optional libraries (cuDNN and NCCL) will be disabled if these libraries are not available at the time of installation.
   See :ref:`install_cudnn` for the instructions.

.. note::

   If you upgrade or downgrade the version of CUDA Toolkit, cuDNN or NCCL, you may need to reinstall CuPy.
   See :ref:`install_reinstall` for details.

Using pip
~~~~~~~~~

You can install `CuPy package <https://pypi.python.org/pypi/cupy>`_ via ``pip``.

::

  $ pip install cupy

Using Tarball
~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download cupy`` or from `the release notes page <https://github.com/cupy/cupy/releases>`_.
You can install CuPy from the tarball::

  $ pip install cupy-x.x.x.tar.gz

You can also install the development version of CuPy from a cloned Git repository::

  $ git clone https://github.com/cupy/cupy.git
  $ cd cupy
  $ pip install .

If you are using source tree downloaded from GitHub, you need to install Cython 0.28.0 or later (``pip install cython``).

Uninstall CuPy
--------------

Use pip to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   When you upgrade Chainer, ``pip`` sometimes installs the new version without removing the old one in ``site-packages``.
   In this case, ``pip uninstall`` only removes the latest one.
   To ensure that CuPy is completely removed, run the above command repeatedly until ``pip`` returns an error.

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


Upgrade CuPy
------------

Just use ``pip install`` with ``-U`` option::

  $ pip install -U cupy

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


.. _install_reinstall:

Reinstall CuPy
--------------

If you want to reinstall CuPy, please uninstall CuPy and then install it.
When reinstalling CuPy, we recommend to use ``--no-cache-dir`` option as ``pip`` caches the previously built binaries::

  $ pip uninstall cupy
  $ pip install cupy --no-cache-dir

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


Run CuPy with Docker
--------------------

We are providing the `official Docker image <https://hub.docker.com/r/cupy/cupy/>`_.
Use `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ nvidia-docker run -it cupy/cupy /bin/bash

Or run the interpreter directly::

  $ nvidia-docker run -it cupy/cupy /usr/bin/python


FAQ
---

Warning message "cuDNN is not enabled" appears when using Chainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.

See :ref:`install_cudnn` and :ref:`install_error` for details.

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
If you encounter this problem, please downgrade or upgrade it.

.. _install_cudnn:

Installing cuDNN and NCCL
~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing cuDNN and NCCL using binary packages (i.e., using ``apt`` or ``yum``) provided by NVIDIA.

If you want to install tar-gz version of cuDNN and NCCL, we recommend you to install it under CUDA directory.
For example, if you are using Ubuntu, copy ``*.h`` files to ``include`` directory and ``*.so*`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.

If you want to use cuDNN or NCCL installed in another directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  export CFLAGS=-I/path/to/cudnn/include
  export LDFLAGS=-L/path/to/cudnn/lib
  export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

.. note::

   Use full paths for the environment variables.
   ``distutils`` that is used in the setup script does not expand the home directory mark ``~``.

.. _install_cuda:

Working with Custom CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed CUDA on the non-default directory or have multiple CUDA versions installed, you may need to manually specify the CUDA installation directory to be used by CuPy.

CuPy uses the first CUDA installation directory found by the following order.

#. ``CUDA_PATH`` environment variable.
#. The parent directory of ``nvcc`` command. CuPy looks for ``nvcc`` command in each directory set in ``PATH`` environment variable.
#. ``/usr/local/cuda``

For example, you can tell CuPy to use non-default CUDA directory by ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy

.. note::

   CUDA installation discovery is also performed at runtime using the rule above.
   Depending on your system configuration, you may also need to set ``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64`` at runtime.

Using custom ``nvcc`` command during installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a custom ``nvcc`` compiler (for example, to use ``ccache``) to build CuPy, please set ``NVCC`` environment variables before installing CuPy::

  export NVCC='ccache nvcc'

.. note::

   During runtime, you don't need to set this environment variable since CuPy doesn't use the nvcc command.

Installation for Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are hacking CuPy source code, we recommend you to use ``pip`` with ``-e`` option for editable mode::

  $ cd /path/to/cupy/source
  $ pip install -e .

Please note that even with ``-e``, you will have to rerun ``pip install -e .`` to regenerate C++ sources using Cython if you modified Cython source files (e.g., ``*.pyx`` files).

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CuPy does not work at all with ``CompileException``, it is possible that CuPy cannot detect CUDA installed on your system correctly.
The followings are error messages commonly observed in such cases.

* ``nvrtc: error: failed to load builtins``
* ``catastrophic error: cannot open source file "cuda_fp16.h"``
* ``error: cannot overload functions distinguished by return type alone``
* ``error: identifier "__half_raw" is undefined``

Please try setting ``LD_LIBRARY_PATH`` and ``CUDA_PATH`` environment variable.
For example, if you have CUDA installed at ``/usr/local/cuda-9.0``::

  export CUDA_PATH=/usr/local/cuda-9.0
  export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

Also see :ref:`install_cuda`.

If you are installing CuPy on Anaconda environment, also make sure that the following packages are not installed.

* `cudatoolkit <https://anaconda.org/anaconda/cudatoolkit>`__
* `cudnn <https://anaconda.org/anaconda/cudnn>`__
* `nccl <https://anaconda.org/anaconda/nccl>`__

Use ``conda uninstall cudatoolkit cudnn nccl`` to remove these package.
