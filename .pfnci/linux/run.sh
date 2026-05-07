#!/bin/bash

USAGE="
${0} TARGET STAGE [STAGE ...]

Arguments:

- TARGET: Name of the test target. Targets are defined in 'tests' directory.
- STAGE: Test stage(s) to execute. Possible stages are:
  - build: Build a docker image used for testing.
  - rmi: Remove a docker image used for testing.
  - push: Push the built docker image so that further test runs can reuse
          the image.
  - cache_get: Pull cache from Google Cloud Storage to CACHE_DIR if available.
  - cache_put: Push cache from CACHE_DIR to Google Cloud Storage.
  - test: Run tests.
  - shell: Start an interactive shell in the docker image for debugging.
           The source tree will be read-write mounted for convenience.
  - benchmark: Run performance benchmarks.

Environment variables:

- PULL_REQUEST: ID of the pull-request to test; should be empty when testing
                a branch.
- GPU: Number of GPUs available for testing.
- CACHE_DIR: Path to the local directory to store cache files.
- CACHE_GCS_DIR: Path to the GCS directory to store a cache archive.
- CACHE_KERNEL_TO_GCS: Set to 1 to enable GCS bucket based kernel cache.
- DOCKER_IMAGE: Base name of the Docker image (without a tag).
- DOCKER_IMAGE_CACHE: Set to 0 to disable using cache when building a docker
                      image.
- BENCHMARK_DIR: Path to the directory to store benchmark results.
"

set -eu


################################################################################
# Main function
################################################################################

main() {
  if (( $# < 2 )); then
    echo "${USAGE}"
    exit 1;
  fi

  TARGET="$1"; shift
  STAGES="$@"

  repo_root="$(cd "$(dirname "${BASH_SOURCE}")/../.."; pwd)"
  base_branch="$(cat "${repo_root}/.pfnci/BRANCH")"
  docker_image="${DOCKER_IMAGE:-asia-northeast1-docker.pkg.dev/pfn-artifactregistry/tmp-public-ci-dlfw/cupy-ci}:${TARGET}-${base_branch}"
  docker_cache_from="${docker_image}"
  cache_archive="linux-${TARGET}-${base_branch}.tar.gz"
  cache_gcs_dir="${CACHE_GCS_DIR:-gs://tmp-asia-pfn-public-ci/cupy-ci/cache}"
  cache_pr_gcs_dir="${cache_gcs_dir}-pr-${PULL_REQUEST:-0}"

  if [[ "${DOCKER_IMAGE_CACHE:-1}" = "0" ]]; then
    docker_cache_from=""
  fi

  echo "
    =====================================================================
    Test Configuration
    =====================================================================
    Target              : ${TARGET}
    Stages              : ${STAGES}
    Pull-Request        : ${PULL_REQUEST:-no}
    GPUs                : ${GPU:-(not set)}
    Repository Root     : ${repo_root}
    Base Branch         : ${base_branch}
    Docker Image        : ${docker_image}
    Docker Image Cache  : ${docker_cache_from}
    Remote Cache        : ${cache_gcs_dir}/${cache_archive}
    Remote Cache (PR)   : ${cache_pr_gcs_dir}/${cache_archive}
    Local Cache         : ${CACHE_DIR:-(not set)}
    =====================================================================
  "

  for stage in ${STAGES}; do
    echo "*** Running stage: ${stage}"
    set -x
    case "${stage}" in
    build )
      tests_dir="${repo_root}/.pfnci/linux/tests"
      DOCKER_BUILDKIT=1 docker build \
          -t "${docker_image}" \
          --cache-from "${docker_cache_from}" \
          --build-arg BUILDKIT_INLINE_CACHE=1 \
          -f "${tests_dir}/${TARGET}.Dockerfile" \
          "${tests_dir}"
      ;;

    rmi )
      docker rmi "${docker_image}"
      ;;

    push )
      docker push --quiet "${docker_image}"
      ;;

    cache_get )
      # Download from GCS and extract to $CACHE_DIR.
      if [[ "${CACHE_DIR:-}" = "" ]]; then
        echo "ERROR: CACHE_DIR is not set!"
        exit 1
      fi
      mkdir -p "${CACHE_DIR}"
      gsutil_with_retry -m -q cp "${cache_gcs_dir}/${cache_archive}" . &&
        du -h "${cache_archive}" &&
        tar -x -f "${cache_archive}" -C "${CACHE_DIR}" &&
        rm -f "${cache_archive}" || echo "WARNING: Remote cache could not be retrieved."
      if [[ "${PULL_REQUEST:-0}" != "0" ]]; then
        gsutil_with_retry -m -q cp "${cache_pr_gcs_dir}/${cache_archive}" . &&
          du -h "${cache_archive}" &&
          tar -x -f "${cache_archive}" -C "${CACHE_DIR}" &&
          rm -f "${cache_archive}" || echo "WARNING: Remote cache (for pull-request) could not be retrieved."
      fi
      ;;

    cache_put )
      # Compress $CACHE_DIR and upload to GCS.
      if [[ "${CACHE_DIR:-}" = "" ]]; then
        echo "ERROR: CACHE_DIR is not set!"
        exit 1
      fi
      tar -c -f "${cache_archive}" -C "${CACHE_DIR}" .
      du -h "${cache_archive}"
      if [[ "${PULL_REQUEST:-0}" != "0" ]]; then
        gsutil -m -q cp "${cache_archive}" "${cache_pr_gcs_dir}/"
      else
        gsutil -m -q cp "${cache_archive}" "${cache_gcs_dir}/"
      fi
      rm -f "${cache_archive}"
      ;;

    test | shell | benchmark )
      container_name="cupy_ci_$$_$RANDOM"
      docker_args=(
        docker run
        --rm
        --name "${container_name}"
        --env "BASE_BRANCH=${base_branch}"
        --shm-size=512m
      )
      if [[ -t 1 ]]; then
        docker_args+=(--interactive)
      fi
      if [[ "${CACHE_DIR:-}" != "" ]]; then
        docker_args+=(--volume="${CACHE_DIR}:${CACHE_DIR}" --env "CACHE_DIR=${CACHE_DIR}")
      fi
      if [[ "${CACHE_KERNEL_TO_GCS:-0}" == "1" ]]; then
        docker_args+=(--env "CUPY_CI_ENABLE_GCP_KERNEL_CACHE=1")
      fi
      if [[ "${PULL_REQUEST:-}" != "" ]]; then
        docker_args+=(--env "PULL_REQUEST=${PULL_REQUEST}")
      fi
      if [[ "${GPU:-}" != "" ]]; then
        docker_args+=(--env "GPU=${GPU}")
      fi
      if [[ "${TARGET}" == *rocm* ]]; then
        docker_args+=(--device=/dev/kfd --device=/dev/dri)
      elif [[ "${TARGET}" == cuda-build ]]; then
        docker_args+=()
      else
        docker_args+=(--runtime=nvidia)
      fi

      if [[ "${stage}" = "benchmark" ]]; then
        mkdir -p ${BENCHMARK_DIR}
        docker_args+=(--volume="${BENCHMARK_DIR}:/perf-results")
      fi

      if [[ ${stage} = test || ${stage} = benchmark ]]; then
        "${docker_args[@]}" --volume="${repo_root}:/src:ro" --workdir "/src" \
            "${docker_image}" timeout 8h bash "/src/.pfnci/linux/tests/${TARGET}.sh" &
        docker_pid=$!
        trap "kill -KILL ${docker_pid}; docker kill '${container_name}' & wait; exit 1" TERM INT HUP
        wait $docker_pid
        trap TERM INT HUP
      elif [[ ${stage} = shell ]]; then
        set +x
        echo "==================== INTERACTIVE SHELL IN CI IMAGE ===================="
        echo "Tips:"
        echo "  - To reproduce CI: bash '${repo_root}/.pfnci/linux/tests/${TARGET}.sh'"
        echo "  - To build CuPy: pip install --no-build-isolation -v -e '.[test]'"
        echo "  - To run tests: pytest tests/path_to_test.py"
        echo "  - Several env vars are automatically set for convenience; to check: env"
        echo "  - To build for current GPU device only: export CUPY_NVCC_GENERATE_CODE=current"
        echo "  - To accelerate build: export CUPY_NUM_BUILD_JOBS=\$(nproc)"
        echo "  - In shell mode, ccache is activated for gcc/g++ but not for nvcc; "
        echo "    to activate: export NVCC='ccache nvcc'"
        echo "  - To persist build/kernel cache across multiple shell session runs: "
        echo "    set CACHE_DIR env var BEFORE starting run.sh"
        echo "======================================================================="
        uid_gid="$(id -u):$(id -g)"
        set -x
        "${docker_args[@]}" --volume="${repo_root}:${repo_root}:rw" --workdir "${repo_root}" \
            --tty --user "${uid_gid}" \
            --env "USER=cupy-user" --env "HOME=/home/cupy-user" --env "SHELL_MODE=yes" \
            "${docker_image}" /bin/bash -c "source ${repo_root}/.pfnci/linux/tests/actions/_environment.sh && exec bash"
      fi
      ;;
    * )
      echo "Unsupported stage: ${stage}" >&2
      exit 1
      ;;
  esac;
  set +x
  done
}

gsutil_with_retry() {
    gsutil "$@" || gsutil "$@" || gsutil "$@"
}


################################################################################
# Bootstrap
################################################################################
main "$@"
