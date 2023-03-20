#!/bin/bash

# Updates alternatives configuration for cuTENSOR package

set -ue

case "${CUDA_VERSION:-}" in
    10.2.* )  CUTENSOR_CUDA_PREFIX="10.2" ;;
    11.0.* )  CUTENSOR_CUDA_PREFIX="11.0" ;;
    11.*   )  CUTENSOR_CUDA_PREFIX="11"   ;;
    12.*   )  CUTENSOR_CUDA_PREFIX="12"   ;;
    *      )  echo "Unknown CUDA_VERSION: ${CUDA_VERSION:-(not set)}" && exit 1 ;;
esac
echo "CUDA version prefix for cuTENSOR derived from CUDA_VERSION (${CUDA_VERSION}): ${CUTENSOR_CUDA_PREFIX}"

# TODO(kmaehashi): this assumes x86_64 env
if which apt &> /dev/null; then
    # APT
    SYSTEM_LIBDIR="/usr/lib/x86_64-linux-gnu"
else
    # RPM
    SYSTEM_LIBDIR="/usr/lib64"
fi

CUTENSOR_LIBNAME="$(basename ${SYSTEM_LIBDIR}/libcutensor.so.?.*)"
CUTENSORMG_LIBNAME="$(basename ${SYSTEM_LIBDIR}/libcutensorMg.so.?.*)"

echo "Changing alternatives configuration:"
set -x
update-alternatives --set "${CUTENSOR_LIBNAME}" "${SYSTEM_LIBDIR}/libcutensor/${CUTENSOR_CUDA_PREFIX}/${CUTENSOR_LIBNAME}"
update-alternatives --set "${CUTENSORMG_LIBNAME}" "${SYSTEM_LIBDIR}/libcutensor/${CUTENSOR_CUDA_PREFIX}/${CUTENSORMG_LIBNAME}"
set +x

echo "Done setting up alternatives for cuTENSOR!"
