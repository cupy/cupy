These files are copied from CUDA Toolkit Distribution and redistributed under the following license:

https://docs.nvidia.com/cuda/archive/9.2/eula/#nvidia-cuda-toolkit-license-agreement

For CUDA 11.2+, we enable CUDA Enhanced Compatibility by hosting the fp16 headers from the latest
CUDA Toolkit release.

* ``cuda-11`` contains headers from CTK 11.8.0.
* ``cuda-12`` contains headers from CTK 12.1.1.

For CUDA 12.2+, we don't bundle header files as `cuda_fp16.h` started to depend
on header files that cannot be bundled (e.g. `vector_types.h`).
https://docs.cupy.dev/en/latest/install.html#cupy-always-raises-nvrtc-error-compilation-6
