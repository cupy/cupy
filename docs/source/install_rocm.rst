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

  $ sudo apt install hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim


Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip


.. _install_hip:

Building CuPy for ROCm
-----------------------

Currently, you need to build CuPy from source to run on AMD GPU.

::

  $ export HCC_AMDGPU_TARGET=gfx900  # This value should be changed based on your GPU
  $ export CUPY_INSTALL_USE_HIP=1
  $ pip install cupy

Note that ``HCC_AMDGPU_TARGET`` must be set to the ISA name supported by your GPU.
Run ``rocminfo`` and use the value displayed in ``Name:`` line (e.g., ``gfx900``).

You may also need to set ``ROCM_HOME`` (e.g., ``ROCM_HOME=/opt/rocm``).
