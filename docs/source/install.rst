Install Guide
=============

.. _before_install:

Before installing CuPy
----------------------

We recommend these platforms.

* `Ubuntu <http://www.ubuntu.com/>`_ 14.04 LTS 64bit
* `CentOS <https://www.centos.org/>`_ 7 64bit

CuPy is supported on Python 2.7.6+, 3.4.3+, 3.5.1+, 3.6.0+.
CuPy uses C++ compiler such as g++.
You need to install it before installing CuPy.
This is typical installation method for each platform::

  # Ubuntu 14.04
  $ apt-get install g++

  # CentOS 7
  $ yum install gcc-c++

If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Install CuPy
---------------

CuPy depends on these Python packages:

* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11, 1.12
* `Six <https://pythonhosted.org/six/>`_ 1.9

CUDA support

* `CUDA <https://developer.nvidia.com/cuda-zone>`_ 6.5, 7.0, 7.5, 8.0
* `filelock <https://filelock.readthedocs.org>`_

cuDNN support

* `cuDNN <https://developer.nvidia.com/cudnn>`_ v2, v3, v4, v5, v5.1, v6

All these libraries are automatically installed with ``pip`` or ``setup.py``.


Install CuPy via pip
~~~~~~~~~~~~~~~~~~~~

We recommend to install CuPy via pip::

  $ pip install cupy


Install CuPy from source
~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install CuPy from source::

  $ tar zxf cupy-x.x.x.tar.gz
  $ cd cupy-x.x.x
  $ python setup.py install


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation. It may helps you::

  $ pip install cupy -vvvv


.. _install_cuda:

Install CuPy with CUDA
~~~~~~~~~~~~~~~~~~~~~~

You need to install CUDA Toolkit before installing CuPy.
If you have CUDA in a default directory or set ``CUDA_PATH`` correctly, CuPy installer finds CUDA automatically::

  $ pip install cupy


.. note::

   CuPy installer looks up ``CUDA_PATH`` environment variable first.
   If it is empty, the installer looks for ``nvcc`` command from ``PATH`` environment variable and use its parent directory as the root directory of CUDA installation.
   If ``nvcc`` command is also not found, the installer tries to use the default directory for Ubuntu ``/usr/local/cuda``.


If you installed CUDA into a non-default directory, you need to specify the directory with ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy


.. warning::

   If you want to use ``sudo`` to install CuPy, note that ``sudo`` command initializes all environment variables.
   Please specify ``CUDA_PATH`` environment variable inside ``sudo`` like this::

      $ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy


.. _install_cudnn:

Install CuPy with CUDA and cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuDNN is a library for Deep Neural Networks that NVIDIA provides.
CuPy can use cuDNN.
If you want to enable cuDNN, install cuDNN and CUDA before installing CuPy.
We recommend you to install developer library of deb package of cuDNN.

If you want to install tar-gz version, we recommend you to install it to CUDA directory.
For example if you uses Ubuntu Linux, copy ``.h`` files to ``include`` directory and ``.so`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.

If you want to use cuDNN installed in other directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  export CFLAGS=-I/path/to/cudnn/include
  export LDFLAGS=-L/path/to/cudnn/lib
  export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH


Install CuPy for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy uses Cython (>=0.24).
Developers need to use Cython to regenerate C++ sources from ``pyx`` files.
We recommend to use ``pip`` with ``-e`` option for editable mode::

  $ pip install -U cython
  $ cd /path/to/cupy/source
  $ pip install -e .

Users need not to install Cython as a distribution package of CuPy only contains generated sources.


Uninstall CuPy
--------------

Use pip to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   When you upgrade CuPy, ``pip`` sometimes installed various version of CuPy in ``site-packages``.
   Please uninstall it repeatedly until ``pip`` returns an error.


Upgrade CuPy
---------------

Just use ``pip`` with ``-U`` option::

  $ pip install -U cupy


Reinstall CuPy
---------------

If you want to reinstall CuPy, please uninstall CuPy and then install it.
We recommend to use ``--no-cache-dir`` option as ``pip`` sometimes uses cache::

  $ pip uninstall cupy
  $ pip install cupy --no-cache-dir

When you install CuPy without CUDA, and after that you want to use CUDA, please reinstall CuPy.
You need to reinstall CuPy when you want to upgrade CUDA.


Run CuPy with Docker
-----------------------

We provide the official Docker image.
Use `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ nvidia-docker run -it cupy/cupy /bin/bash

Or, run the interpreter directly::

  $ nvidia-docker run -it cupy/cupy /usr/bin/python


What "recommend" means?
-----------------------

We tests CuPy automatically with Jenkins.
All supported environments are tested in this environment.
We cannot guarantee that CuPy works on other environments.


FAQ
---

MemoryError happens
~~~~~~~~~~~~~~~~~~~

You maybe failed to install Cython.
Please install it manually.
See :ref:`install_error`.


Examples says "cuDNN is not enabled"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.
``-vvvv`` option helps you.
See :ref:`install_cudnn`.
