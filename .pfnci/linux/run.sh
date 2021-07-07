#!/bin/bash

USAGE="
${0} TARGET STAGE [STAGE ...]

Arguments:

- TARGET: Name of the test target. Targets are defined in 'tests' directory.
- STAGE: Test stage(s) to execute. Possible stages are:
  - build: Build a docker image used for testing.
  - push: Push the built docker image so that further test runs can reuse
          the image.
  - cache_get: Pull cache from Google Cloud Storage to CACHE_DIR.
  - cache_put: Push cache from CACHE_DIR to Google Cloud Storage.
  - test: Run tests.

Environment variables:

- GPU: Number of GPUs available for testing.
- CACHE_DIR: Path to the local directory to store cache files.
- CACHE_GCS_DIR: Path to the GCS directory to store a cache archive.
- DOCKER_IMAGE: Base name of the Docker image (without a tag).
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
  docker_image="${DOCKER_IMAGE:-asia.gcr.io/pfn-public-ci/cupy-ci}:${TARGET}-${base_branch}"
  cache_archive="cupy_ci_cache_${TARGET}-${base_branch}.tar.gz"
  cache_gcs_dir="${CACHE_GCS_DIR:-gs://tmp-asia-pfn-public-ci/cupy-ci}"

  echo "
    =====================================================================
    Test Configuration
    =====================================================================
    Target              : ${TARGET}
    Stages              : ${STAGES}
    GPUs                : ${GPU:-(not set)}
    Repository Root     : ${repo_root}
    Base Branch         : ${base_branch}
    Docker Image        : ${docker_image}
    Remote Cache        : ${cache_gcs_dir}/${cache_archive}
    Local Cache         : ${CACHE_DIR:-(not set)}
    =====================================================================
  "

  if which gcloud &> /dev/null; then
    gcloud auth configure-docker || echo "Failed to configure access to GCR"
  else
    echo "Skipping GCR configuration"
  fi

  set -x
  for stage in ${STAGES}; do case "${stage}" in
    build )
      tests_dir="${repo_root}/.pfnci/linux/tests"
      docker build -t "${docker_image}" -f "${tests_dir}/${TARGET}.Dockerfile" "${tests_dir}"
      ;;

    push )
      docker push "${docker_image}"
      ;;

    cache_get )
      # Download from GCS and extract to $CACHE_DIR.
      if [[ "${CACHE_DIR:-}" = "" ]]; then
        echo "ERROR: CACHE_DIR is not set!"
        exit 1
      fi
      mkdir -p "${CACHE_DIR}"
      gsutil -m -q cp "${cache_gcs_dir}/${cache_archive}" .
      tar -x -f "${cache_archive}" -C "${CACHE_DIR}"
      rm -f "${cache_archive}"
      ;;

    cache_put )
      # Compress $CACHE_DIR and upload to GCS.
      if [[ "${CACHE_DIR:-}" = "" ]]; then
        echo "ERROR: CACHE_DIR is not set!"
        exit 1
      fi
      tar -c -f "${cache_archive}" -C "${CACHE_DIR}" .
      gsutil -m -q cp "${cache_archive}" "${cache_gcs_dir}"
      rm -f "${cache_archive}"
      ;;

    test )
      container_name="cupy_ci_$$_$RANDOM"
      docker_args=(
        docker run
        --rm
        --name "${container_name}"
        --volume="${repo_root}:/src:ro"
        --workdir "/src"
      )
      if [[ -t 1 ]]; then
        docker_args+=(--interactive)
      fi
      if [[ "${CACHE_DIR:-}" != "" ]]; then
        docker_args+=(--volume="${CACHE_DIR}:${CACHE_DIR}" --env "CACHE_DIR=${CACHE_DIR}")
      fi
      if [[ "${GPU:-}" != "" ]]; then
        docker_args+=(--env "GPU=${GPU}")
      fi
      if [[ "${TARGET}" == *cuda* ]]; then
        docker_args+=(--runtime=nvidia)
      elif [[ "${TARGET}" == *rocm* ]]; then
        docker_args+=(--device=/dev/kfd --device=/dev/dri)
      else
        echo "ERROR: Unknown platform!"
        exit 1
      fi

      docker_args+=("${docker_image}" timeout 8h "/src/.pfnci/linux/tests/${TARGET}.sh")
      "${docker_args[@]}" &
      docker_pid=$!
      trap "docker kill '${container_name}'; exit 1" TERM INT HUP
      wait $docker_pid
      trap TERM INT HUP
      ;;

    * )
      echo "Unsupported stage: ${stage}" >&2
      exit 1
      ;;
  esac; done
}


################################################################################
# Bootstrap
################################################################################
main "$@"
