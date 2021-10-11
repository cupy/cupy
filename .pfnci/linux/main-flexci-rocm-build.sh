#!/bin/bash

set -eu

################################################################################
# Main function
################################################################################
main() {
      # ROCM_REPO can be set via config.pbtxt and will be used as a
      # part of ROCm repository URL (http://repo.radeon.com/rocm/apt/).
      ROCM_REPO=${ROCM_REPO:-debian}
      cd "$(dirname "${BASH_SOURCE}")"/../..
      apt update -qqy
      apt install python3.7-dev python3-apt python3-pip python3-setuptools -qqy
      python3.7 -m pip install cython numpy

      wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
      echo "deb [arch=amd64] http://repo.radeon.com/rocm/apt/${ROCM_REPO}/ xenial main" | tee /etc/apt/sources.list.d/rocm.list

      # Uninstall CUDA to ensure it's a clean ROCm environment
      # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver
      apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "cuda*" "nsight*" -qqy
      apt-get autoremove -qqy

      apt update -qqy
      apt install $ROCM_PACKAGES -qqy
      export HCC_AMDGPU_TARGET=gfx900
      export ROCM_HOME=/opt/rocm
      export CUPY_INSTALL_USE_HIP=1
      export PATH=$PATH:$ROCM_HOME/bin:$ROCM_HOME/profiler/bin:$ROCM_HOME/opencl/bin/x86_64
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_HOME/lib
      echo $(hipconfig)
      python3.7 -m pip install -v .
      # Make sure that CuPy is importable.
      # Note that CuPy cannot be imported from the source directory.
      pushd /
      python3.7 -c "import cupy; cupy.show_config(_full=True)"
      popd
}

################################################################################
# Bootstrap
################################################################################
main "$@"
