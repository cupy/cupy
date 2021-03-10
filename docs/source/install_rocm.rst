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

When building or running CuPy for ROCm, the following environment variables are effective.

* ``ROCM_HOME``: directory containing the ROCm software (e.g., ``/opt/rocm``).

Docker
------

You can try running CuPy for ROCm using Docker.

::

  $ docker run -it --device=/dev/kfd --device=/dev/dri --group-add video cupy/cupy-rocm

.. _install_hip:

Installing Binary Packages
--------------------------

Wheels (precompiled binary packages) are available for Linux (x86_64).

Currently we only offer wheels for ROCm v4.0.x.

::

  $ pip install --pre cupy-rocm-4-0

Building CuPy for ROCm From Source
----------------------------------

To build CuPy from source, set the ``CUPY_INSTALL_USE_HIP``, ``ROCM_HOME``, and ``HCC_AMDGPU_TARGET`` environment variables.
(``HCC_AMDGPU_TARGET`` is the ISA name supported by your GPU.
Run ``rocminfo`` and use the value displayed in ``Name:`` line (e.g., ``gfx900``).
You can specify a comma-separated list of ISAs if you have multiple GPUs of different architectures.)

::

  $ export CUPY_INSTALL_USE_HIP=1
  $ export ROCM_HOME=/opt/rocm
  $ export HCC_AMDGPU_TARGET=gfx906
  $ pip install --pre cupy

.. note::

  If you don't specify the ``HCC_AMDGPU_TARGET`` environment variable, CuPy will be built for the GPU architectures available on the build host.
  This behavior is specific to ROCm builds; when building CuPy for NVIDIA CUDA, the build result is not affected by the host configuration.
