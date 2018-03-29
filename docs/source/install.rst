Installation Guide
==================

.. contents:: :local:

Recommended Environments
------------------------

We recommend these Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`_ 14.04/16.04 LTS 64bit
* `CentOS <https://www.centos.org/>`_ 7 64bit

The following versions of Python can be used: 2.7.6+, 3.4.3+, 3.5.1+, and 3.6.0+.

.. warning::

   If you are using certain versions of conda, it may fail to build CuPy with error
   ``g++: error: unrecognized command line option ‘-R’``.
   This is due to a bug in conda (see `conda/conda#6030 <https://github.com/conda/conda/issues/6030>`_ for details).
   If you encounter this problem, please downgrade or upgrade it.

.. note::

   We are testing CuPy automatically with Jenkins, where all the *recommended* environments above are tested.
   We cannot guarantee that CuPy works on other environments including Windows and macOS, even if CuPy looks running correctly.

Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip


Requirements
------------

You need to have the following components to use CuPy.

* `NVIDIA CUDA GPU <https://developer.nvidia.com/cuda-gpus>`_
    * Compute Capability of the GPU must be at least 3.0.
* `CUDA Toolkit <https://developer.nvidia.com/cuda-zone>`_
    * Supported Versions: 7.0, 7.5, 8.0, 9.0 and 9.1.
    * If you have multiple versions of CUDA Toolkit installed, CuPy will choose one of the CUDA installation automatically.
      See :ref:`install_cuda` for details.
* `NumPy <http://www.numpy.org/>`_
    * Supported Versions: 1.9, 1.10, 1.11, 1.12 and 1.13.
    * NumPy will be installed automatically during the installation of CuPy.

Optional Libraries
~~~~~~~~~~~~~~~~~~

Some features in CuPy will only be enabled if the corresponding libraries are installed.

* `cuDNN <https://developer.nvidia.com/cudnn>`_ (library to accelerate deep neural network computations)
    * Supported Versions: v4, v5, v5.1, v6, v7
* `NCCL <https://developer.nvidia.com/nccl>`_  (library to perform collective multi-GPU / multi-node computations)
    * Supported Versions: v1.3.4, v2


Install CuPy
------------

Wheels (precompiled binary packages) are available for the recommended environments above.
Package names are different depending on the CUDA version you have installed on your host.

::

  (For CUDA 8.0)
  $ pip install cupy-cuda80

  (For CUDA 9.0)
  $ pip install cupy-cuda90

  (For CUDA 9.1)
  $ pip install cupy-cuda91

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

  # Ubuntu 14.04
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

If you are using source tree downloaded from GitHub, you need to install Cython 0.26.1 or later (``pip install cython``).

Uninstall CuPy
--------------

Use pip to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   When you upgrade Chainer, ``pip`` sometimes install the new version without removing the old one in ``site-packages``.
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

If you want to use a custom ``nvcc`` compiler (for example, to use ``ccache`` ), please set ``NVCC`` environment variables before installing CuPy::

  export NVCC='ccache nvcc'

.. note::

   Setting ``NVCC`` environment variable does not affect at runtime, as CuPy does not use ``nvcc`` command at runtime.

Installation for Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are hacking CuPy source code, we recommend you to use ``pip`` with ``-e`` option for editable mode::

  $ cd /path/to/cupy/source
  $ pip install -e .

Please note that even with ``-e``, you will have to rerun ``pip install -e .`` to regenerate C++ sources using Cython if you modified Cython source files (e.g., ``*.pyx`` files)

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CuPy does not work at all with ``CompileException``, it is possible that CuPy cannot detect CUDA installed on your system correctly.
The followings are error messages commonly observed in such case.

* ``nvrtc: error: failed to load builtins``
* ``catastrophic error: cannot open source file "cuda_fp16.h"``
* ``error: cannot overload functions distinguished by return type alone``
* ``error: identifier "__half_raw" is undefined``

Please try setting ``LD_LIBRARY_PATH`` and ``CUDA_PATH`` environment variable.
For example, if you have CUDA installed at ``/usr/local/cuda-9.0``::

  export CUDA_PATH=/usr/local/cuda-9.0
  export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

Also see :ref:`install_cuda`.
