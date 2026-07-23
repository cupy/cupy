#!/bin/bash

set -uex

# Fetch the CuPy wheel built by GHA (.github/workflows/ci.yml -> build-wheel.yml)
# for the commit under test, then pip-install it. Artifacts are producer-pinned:
# only artifacts uploaded by a successful pull_request or push run of ci.yml at
# the exact commit under test are accepted. Failures at this step mean either
# the wheel-build matrix does not cover this (CUDA, Python) tuple (by design,
# until the matrix is expanded), or a fresh /test needs to be issued.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."; pwd)"

# The token is delivered by run.sh as a mounted file (never an env var), so it
# stays out of the container's `env` dump. Read it with xtrace off so the value
# never reaches the trace log, then delete the file and (later) unset GH_TOKEN
# so the token is gone before the PR-controlled test scripts run.
if [[ -z "${CUPY_CI_GITHUB_TOKEN_FILE:-}" || ! -r "${CUPY_CI_GITHUB_TOKEN_FILE}" ]]; then
    echo "Error: CUPY_CI_GITHUB_TOKEN_FILE must point to a readable token file (provisioned on the FlexCI side) to fetch wheel artifacts from cupy/cupy CI" >&2
    exit 1
fi
set +x
GH_TOKEN="$(cat "${CUPY_CI_GITHUB_TOKEN_FILE}")"
export GH_TOKEN
# Delete the token file now that it is read, so PR-controlled test code running
# later in this container cannot read it back. (The /test trust model still
# exposes the token to this vouched PR code during the fetch itself.)
rm -f "${CUPY_CI_GITHUB_TOKEN_FILE}"
set -x

# The free-threaded ABI ('t') is part of the artifact name build-wheel.yml
# uploads (py3.14t, not py3.14); sys.version_info alone would drop it.
PY_VER=$(python3 -c 'import sys, sysconfig; print(f"{sys.version_info.major}.{sys.version_info.minor}" + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""))')
CUDA_VER=$(python3 -c "import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'])")
CUDA_MAJOR="${CUDA_VER%%.*}"

# Derive the artifact suffix ci-guard.sh emits:
#   PR (pull_request labeled):  pr<N>-<head sha>
#   push (post-merge):          <sha>
# PULL_REQUEST is forwarded by run.sh from FlexCI; 0 (or unset) means non-PR.
if [[ "${PULL_REQUEST:-0}" != "0" ]]; then
    SUFFIX_PREFIX="pr${PULL_REQUEST}-"
    EVENT="pull_request"
else
    SUFFIX_PREFIX=""
    EVENT="push"
fi

# FlexCI's checkout may be the tested commit directly, or a merge of that
# commit into its base -- in the merge case the tested commit is the second
# parent.
CANDIDATE_SHAS=("$(git -C "${REPO_ROOT}" rev-parse HEAD)")
if SECOND_PARENT="$(git -C "${REPO_ROOT}" rev-parse --verify --quiet "HEAD^2")"; then
    CANDIDATE_SHAS+=("${SECOND_PARENT}")
fi

ARTIFACT_ID=""
RUN_ID=""
for sha in "${CANDIDATE_SHAS[@]}"; do
    expected_name="cupy-cuda${CUDA_MAJOR}x-py${PY_VER}-linux-64-${SUFFIX_PREFIX}${sha}"

    # The artifact name is unique per (PR, commit, platform, Python), so query
    # it directly -- each match carries its producing run id, which is our Run
    # ID (no run enumeration). A gh failure here is fatal (never "no artifact").
    if ! candidates="$(gh api --paginate \
        "repos/cupy/cupy/actions/artifacts?name=${expected_name}&per_page=100" \
        --jq ".artifacts[]
              | select(.expired == false)
              | select(.workflow_run.head_sha == \"${sha}\")
              | \"\(.id) \(.workflow_run.id)\"")"; then
        echo "Error: failed to query wheel artifacts for ${sha}" >&2
        exit 1
    fi

    # Producer pin: the artifact name is a free string any run could pick, so
    # confirm its producing run is a *successful ci.yml run* before trusting the
    # wheel. This rejects a same-named artifact from a sibling workflow or a
    # failed run at the same commit, and resolves the Run ID with no guessing.
    while read -r cand_artifact cand_run; do
        [[ -z "${cand_run}" ]] && continue
        if ! producer="$(gh api "repos/cupy/cupy/actions/runs/${cand_run}" \
            --jq "select(.path == \".github/workflows/ci.yml\" and .conclusion == \"success\" and .event == \"${EVENT}\") | .id")"; then
            echo "Error: failed to query producing run ${cand_run}" >&2
            exit 1
        fi
        if [[ -n "${producer}" ]]; then
            ARTIFACT_ID="${cand_artifact}"
            RUN_ID="${cand_run}"
            break 2
        fi
    done <<< "${candidates}"
done

if [[ -z "${ARTIFACT_ID}" ]]; then
    echo "Error: no wheel artifact from a successful ci.yml run for candidate SHAs: ${CANDIDATE_SHAS[*]}" >&2
    echo "Expected name: cupy-cuda${CUDA_MAJOR}x-py${PY_VER}-linux-64-${SUFFIX_PREFIX}<sha>." >&2
    echo "Re-issue /test on the PR (or check the ci.yml push run for the merge commit)." >&2
    exit 1
fi
echo "Resolved wheel artifact ${ARTIFACT_ID} from ci.yml run ${RUN_ID}"

WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "${WHEEL_DIR}"' EXIT

# Download by immutable artifact ID -- name-scoped downloads are not
# re-run-attempt-safe if build-wheel.yml's overwrite:true is ever reverted.
if ! gh api "repos/cupy/cupy/actions/artifacts/${ARTIFACT_ID}/zip" > "${WHEEL_DIR}/artifact.zip"; then
    echo "Error: failed to download wheel artifact ${ARTIFACT_ID}" >&2
    exit 1
fi

# The token is no longer needed past this point.
unset GH_TOKEN

python3 -m zipfile -e "${WHEEL_DIR}/artifact.zip" "${WHEEL_DIR}"
rm "${WHEEL_DIR}/artifact.zip"

WHEEL="$(ls "${WHEEL_DIR}"/*.whl | head -n 1)"
time python3 -m pip install --user -v "${WHEEL}[test]"
