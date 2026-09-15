#!/bin/bash

# Front-door guard for .github/workflows/ci.yml (the build workflow).
# Decides what mode this run is in:
#
#   test   : builds and dispatches. Set for pull_request/labeled runs whose
#            vouched artifact records a `/test <target>` comment, or for
#            push runs of merged/reviewed code.
#   skip   : no build; still uploads a dispatch sentinel so flexci.yml fires
#            and the dispatcher posts skip statuses. Set for
#            pull_request/labeled runs whose vouched artifact records a
#            `/test skip` or `/test force-skip` comment.
#   no-op  : nothing runs downstream. Unrelated labels, non-vouched
#            ci:triggered, branch deletions, skip-ci merged PRs, etc.
#
# Inputs (standard GitHub Actions environment): GITHUB_EVENT_NAME,
# GITHUB_EVENT_PATH, GITHUB_SHA, GITHUB_REF_NAME, GITHUB_REPOSITORY,
# GITHUB_OUTPUT, GH_TOKEN.

set -euo pipefail

mode=no-op
ref=""
artifact_suffix=""
head_sha=""
pr_number=""

case "$GITHUB_EVENT_NAME" in
pull_request)
  # Only `labeled` is subscribed. Any label other than ci:triggered (triage,
  # mergify, ...) makes this a no-op run; flexci.yml additionally requires
  # the dispatch sentinel, so no-op runs can never cause a dispatch.
  label="$(jq -r '.label.name // empty' "$GITHUB_EVENT_PATH")"
  sender="$(jq -r '.sender.login' "$GITHUB_EVENT_PATH")"
  if [[ "${label}" != "ci:triggered" ]]; then
    echo "Label '${label}' is not ci:triggered; mode=no-op"
  else
    if [[ "${sender}" != "cupy-ci-trigger[bot]" ]]; then
      perm="$(gh api "repos/${GITHUB_REPOSITORY}/collaborators/${sender}/permission" --jq .permission 2>/dev/null || echo none)"
      if [[ "${perm}" != "admin" && "${perm}" != "write" ]]; then
        echo "::warning::ci:triggered applied by ${sender} (permission=${perm}); mode=no-op"
        label=""
      fi
    fi
    if [[ -n "${label}" ]]; then
      pr_number="$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH")"
      head_sha="$(jq -r '.pull_request.head.sha' "$GITHUB_EVENT_PATH")"
      # GITHUB_SHA is the merge commit pinned at event time -- immune to
      # pushes racing the label bounce.
      ref="${GITHUB_SHA}"
      artifact_suffix="pr${pr_number}-${head_sha}"

      # Distinguish /test skip from /test <target> by reading the vouched
      # artifact ci-trigger.yml uploaded for this (PR, head). The read is
      # a routing hint only -- flexci.yml re-reads the same artifact from a
      # trusted checkout, so a hostile PR that spoofs the guard's decision
      # cannot forge the dispatcher's action (worst case is CI-time
      # misbehavior of its own build/skip).
      #
      # The upload-then-bounce ordering in ci-trigger.yml means the artifact
      # is uploaded before the label event fires here, but the /actions/
      # artifacts listing can lag a few seconds behind the upload. Retry
      # 3 x 10s before treating absence as a stale-head race, so a healthy
      # /test doesn't go red on a listing hiccup.
      artifact_id=""
      for attempt in 1 2 3; do
        artifact_id="$(gh api \
            "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=dispatch-request-${pr_number}-${head_sha}&per_page=100" \
            --jq '[.artifacts[] | select(.expired == false)]
                  | sort_by(.created_at) | last | .id // empty')"
        [[ -n "${artifact_id}" ]] && break
        if [[ ${attempt} -lt 3 ]]; then
          echo "Vouched artifact for PR #${pr_number} at ${head_sha} not yet listed (attempt ${attempt}/3); waiting 10s..."
          sleep 10
        fi
      done
      if [[ -z "${artifact_id}" ]]; then
        # Bot-applied ci:triggered with no vouched artifact means a stale
        # head race: ci-trigger.yml uploads the artifact BEFORE bouncing
        # the label, so absence at label-event time means the artifact was
        # keyed to a superseded SHA (issue #10258). Fail the guard so
        # `Check job status` goes red and loud instead of coasting on a
        # skipped-therefore-passing required check. Non-bot manual
        # ci:triggered (admin/write hand-application) still degrades to
        # mode=no-op below.
        if [[ "${sender}" == "cupy-ci-trigger[bot]" ]]; then
          echo "::error::Bot-applied ci:triggered on PR #${pr_number} at ${head_sha} has no vouched artifact (stale head or upload race); failing loud."
          exit 1
        fi
        echo "::warning::No vouched request for PR #${pr_number} at ${head_sha}; mode=no-op"
        pr_number=""
        head_sha=""
        ref=""
        artifact_suffix=""
      else
        gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts/${artifact_id}/zip" \
            > "${RUNNER_TEMP}/dispatch-request.zip"
        unzip -q -o "${RUNNER_TEMP}/dispatch-request.zip" -d "${RUNNER_TEMP}/dispatch-request"
        body="$(jq -r '.comment.body' "${RUNNER_TEMP}/dispatch-request/event.json")"
        # Match the first-line convention used by the dispatcher's
        # extract_requested_tags: a comment mixing "/test cuda120" and
        # "/test skip" is treated as the first line's intent.
        first_test="$(printf '%s\n' "${body}" | grep -m1 -E '^/test[[:space:]]+[^[:space:]]' || true)"
        if [[ "${first_test}" =~ ^/test[[:space:]]+(skip|force-skip)[[:space:]]*$ ]]; then
          mode=skip
        else
          mode=test
        fi
      fi
    fi
  fi
  ;;
push)
  # Branch deletions also deliver push events; nothing to build.
  if [[ "$(jq -r '.deleted' "$GITHUB_EVENT_PATH")" == "true" ]] || \
     [[ "$(jq -r '.after' "$GITHUB_EVENT_PATH")" =~ ^0+$ ]]; then
    echo "Branch deletion; mode=no-op"
  else
    # Honor skip-ci on the merged PR. The commit->PRs endpoint can return
    # open/unrelated associations, so require a PR that was really merged
    # into this branch, preferring an exact merge-commit match.
    pr_json="$(gh api "repos/${GITHUB_REPOSITORY}/commits/${GITHUB_SHA}/pulls" \
      --jq "[.[] | select(.merged_at != null and .base.ref == \"${GITHUB_REF_NAME}\")]
            | (map(select(.merge_commit_sha == \"${GITHUB_SHA}\")) + .) | .[0] // empty")"
    if [[ -n "${pr_json}" ]]; then
      pr_number="$(jq -r '.number' <<< "${pr_json}")"
      if jq -e '.labels[] | select(.name == "skip-ci")' <<< "${pr_json}" >/dev/null; then
        echo "PR #${pr_number} carries skip-ci; mode=no-op"
        pr_number=""
      else
        mode=test
      fi
    else
      # Direct push with no associated merged PR: build + dispatch anyway
      # (superset of the old CI's behavior; rare by repo policy).
      mode=test
    fi
    if [[ "${mode}" == "test" ]]; then
      ref="${GITHUB_SHA}"
      artifact_suffix="${GITHUB_SHA}"
      head_sha="${GITHUB_SHA}"
    fi
  fi
  ;;
*)
  echo "Unsupported event: $GITHUB_EVENT_NAME"
  ;;
esac

{
  echo "mode=${mode}"
  echo "ref=${ref}"
  echo "artifact_suffix=${artifact_suffix}"
  echo "head_sha=${head_sha}"
  echo "pr_number=${pr_number}"
} >> "$GITHUB_OUTPUT"
