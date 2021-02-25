Using CuPy on AMD GPU (experimental)
====================================

CuPy has an experimental support for AMD GPU (ROCm).

Requirements
------------

* `AMD GPU supported by ROCm <https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support>`_

* `ROCm <https://rocmdocs.amd.com/en/latest/index.html>`_: v3.5+
    * See the `ROCm Installation Guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_ for details.

The following ROCm libraries are required:

::

  $ sudo apt install hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl

Environment Variables
---------------------

When running CuPy for ROCm, the following environment variables need to be set.

* ``HCC_AMDGPU_TARGET``: ISA name supported by your GPU.
  Run ``rocminfo`` and use the value displayed in ``Name:`` line (e.g., ``gfx900``).
  You can specify comma-separated list of ISAs if you have multiple GPUs of different architecture.

* ``ROCM_HOME``: directory containing the ROCm software (e.g., ``/opt/rocm``).

Exmaple:

::

  $ export HCC_AMDGPU_TARGET=gfx900
  $ export ROCM_HOME=/opt/rocm

Docker
------

You can try running CuPy for ROCm using Docker.

::

  $ docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --env HCC_AMDGPU_TARGET=gfx900 cupy/cupy-rocm

.. _install_hip:

Installing Binary Packages
--------------------------

Wheels (precompiled binary packages) are available for Linux (x86_64).

Currently we only offer wheels for ROCm v4.0.x.

::

  $ pip install --pre cupy-rocm-4-0

Building CuPy for ROCm
-----------------------

To build CuPy from source, set ``CUPY_INSTALL_USE_HIP`` environment variable.

::

  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install --pre cupy
