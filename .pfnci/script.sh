#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .pfnci/script.sh py37".  If a machine running the script has no GPUs,
# this should fall back to CPU testing automatically.  This script requires that
# a corresponding Docker image is accessible from the machine.
# TODO(imos): Enable external contributors to test this script on their
# machines.
#
# Usage: .pfnci/script.sh [target]
# - target is a test target (e.g., "py37").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.  GPU=0 disables GPU
#       testing.
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.
# - XPYTEST ... Set XPYTEST=/path/to/xpytest-linux for testing xpytest.  It will
#       replace xpytest installed inside a Docker image with the given binary.
#       It should be useful to test xpytest.
# - SPREADSHEET_ID ... Set SPREADSHEET_ID (e.g.,
#       "1u5OYiPOL3XRppn73XBSgR-XyDuHKb_4Ilmx1kgJfa-k") to enable xpytest to
#       report to a spreadsheet.

set -eu

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"

  # Initialization.
  prepare_docker &
  wait

  # Prepare docker args.
  docker_args=(docker run  --rm)

  # Base development branch name.
  base_branch="$(get_base_branch)"

  # Run target-specific commands.
  case "${TARGET}" in
    # Unit tests.
    'py37' | 'py27and35' )
      docker_args+=(
          --volume="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd):/src:ro")
      if [ "${GPU:-0}" != '0' ]; then
        docker_args+=(
            --ipc=host --privileged --env="GPU=${GPU}" --runtime=nvidia)
      fi
      if [ "${XPYTEST:-}" != '' ]; then
        docker_args+=(--volume="${XPYTEST}:/usr/local/bin/xpytest:ro")
      fi
      docker_args+=(
          --env="XPYTEST_NUM_THREADS=${XPYTEST_NUM_THREADS:-$(nproc)}")
      if [ "${SPREADSHEET_ID:-}" != '' ]; then
        docker_args+=(--env="SPREADSHEET_ID=${SPREADSHEET_ID}")
      fi
      run "${docker_args[@]}" \
          "asia.gcr.io/pfn-public-ci/cupy-ci-prep.${TARGET}:${base_branch}" \
          bash /src/.pfnci/run.sh "${TARGET}"
      ;;
    py37.rocm-* )
      # ROCM_REPO can be set via config.pbtxt and will be used as a
      # part of ROCm repository URL (http://repo.radeon.com/rocm/apt/).
      ROCM_REPO=${ROCM_REPO:-debian}
      cd "$(dirname "${BASH_SOURCE}")"/..
      apt update -qqy
      apt install python3.7-dev python3-apt python3-pip python3-setuptools -qqy
      python3.7 -m pip install cython numpy

      wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
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
      python3.7 -c "import cupy"
      popd
      ;;
    # Docker builds.
    docker.* )
      # Parse the target as "docker.{target}.{action}".
      local fragments
      IFS=. fragments=(${TARGET})
      local target="${fragments[1]}"
      local action="${fragments[2]}"
      if [ "${action}" != 'push' -a "${action}" != 'test' ]; then
        echo "Unsupported docker target action: ${action}" >&2
        exit 1
      fi
      if [ "${action}" = "push" ] && ! is_known_base_branch "${FLEXCI_BRANCH}"; then
        echo "Branch invalid for docker push: ${FLEXCI_BRANCH}" >&2
        exit 1
      fi

      local cupy_directory="$(pwd)"
      run docker build -t \
          "asia.gcr.io/pfn-public-ci/cupy-ci-prep.${target}:${base_branch}" \
          -f "$(dirname "${BASH_SOURCE}")/${target}.Dockerfile" \
          "${cupy_directory}"
      if [ "${action}" == 'push' ]; then
        run docker push "asia.gcr.io/pfn-public-ci/cupy-ci-prep.${target}:${base_branch}"
      fi
      ;;
    # Unsupported targets.
    * )
      echo "Unsupported target: ${TARGET}" >&2
      exit 1
      ;;
  esac
}

################################################################################
# Utility functions
################################################################################

# run executes a command.  If DRYRUN is enabled, run just prints the command.
run() {
  echo '+' "$@" >&2
  if [ "${DRYRUN:-}" == '' ]; then
    "$@"
  fi
}

# prepare_docker makes docker use tmpfs to speed up.
# CAVEAT: Do not use docker during this is running.
prepare_docker() {
  # Mount tmpfs to docker's root directory to speed up.
  if [ "${CI:-}" != '' ]; then
    run service docker stop
    run mount -t tmpfs -o size=100% tmpfs /var/lib/docker
    run service docker start
  fi
  # Configure docker to pull images from gcr.io.
  run gcloud auth configure-docker
}

KNOWN_BASE_BRANCHES="master v8 v9"

# is_known_base_branch returns 0 only if the given branch name is a known
# base development branch.
is_known_base_branch() {
  local branch="${1##refs/heads/}"
  for BASE_BRANCH in ${KNOWN_BASE_BRANCHES}; do
    if [ "${branch}" = "${BASE_BRANCH}" ]; then
      return 0
    fi
  done
  return 1
}

# get_base_branch returns the base development branch for the current HEAD.
get_base_branch() {
  for BASE_BRANCH in ${KNOWN_BASE_BRANCHES}; do
    git merge-base --is-ancestor "origin/${BASE_BRANCH}" HEAD && echo "${BASE_BRANCH}" && return 0
  done
  echo "Base branch of HEAD is not valid." >&2
  return 1
}

################################################################################
# Bootstrap
################################################################################
main "$@"
