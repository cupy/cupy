#!/bin/bash

# Front-door guard for .github/workflows/ci.yml (the build workflow).
# Decides whether this run builds, and computes what to build:
#
#   pull_request/labeled : build leg for /test. Runs only for the
#                          ci:triggered label applied by the trigger bot (or
#                          a user with write access). Builds the merge commit
#                          pinned in GITHUB_SHA at event time; wheel
#                          artifacts are tagged with the PR number so that
#                          same-head PRs against different bases never share
#                          wheels.
#   push                 : post-merge leg. Skips branch deletions and merges
#                          of PRs labeled skip-ci; otherwise builds the
#                          pushed commit (refreshing this branch's cache
#                          scope with merged code). The workflow dispatches
#                          FlexCI in-run after the builds.
#
# Inputs (standard GitHub Actions environment): GITHUB_EVENT_NAME,
# GITHUB_EVENT_PATH, GITHUB_SHA, GITHUB_REF_NAME, GITHUB_REPOSITORY,
# GITHUB_OUTPUT, GH_TOKEN.

set -euo pipefail

should_run=false
ref=""
artifact_suffix=""
head_sha=""
pr_number=""

case "$GITHUB_EVENT_NAME" in
pull_request)
  # Only `labeled` is subscribed. Any label other than ci:triggered (triage,
  # mergify, ...) makes this a no-op run; flexci.yml additionally requires
  # the build sentinel, so no-op runs can never cause a dispatch.
  label="$(jq -r '.label.name // empty' "$GITHUB_EVENT_PATH")"
  sender="$(jq -r '.sender.login' "$GITHUB_EVENT_PATH")"
  if [[ "${label}" != "ci:triggered" ]]; then
    echo "Label '${label}' is not ci:triggered; skipping"
  else
    if [[ "${sender}" != "cupy-ci-trigger[bot]" ]]; then
      perm="$(gh api "repos/${GITHUB_REPOSITORY}/collaborators/${sender}/permission" --jq .permission 2>/dev/null || echo none)"
      if [[ "${perm}" != "admin" && "${perm}" != "write" ]]; then
        echo "::warning::ci:triggered applied by ${sender} (permission=${perm}); skipping"
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
      should_run=true
    fi
  fi
  ;;
push)
  # Branch deletions also deliver push events; nothing to build.
  if [[ "$(jq -r '.deleted' "$GITHUB_EVENT_PATH")" == "true" ]] || \
     [[ "$(jq -r '.after' "$GITHUB_EVENT_PATH")" =~ ^0+$ ]]; then
    echo "Branch deletion; skipping"
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
        echo "PR #${pr_number} carries skip-ci; skipping"
        pr_number=""
      else
        should_run=true
      fi
    else
      # Direct push with no associated merged PR: build + dispatch anyway
      # (superset of the old CI's behavior; rare by repo policy).
      should_run=true
    fi
    if [[ "${should_run}" == "true" ]]; then
      ref="${GITHUB_SHA}"
      artifact_suffix="${GITHUB_SHA}"
    fi
  fi
  ;;
*)
  echo "Unsupported event: $GITHUB_EVENT_NAME"
  ;;
esac

{
  echo "should_run=${should_run}"
  echo "ref=${ref}"
  echo "artifact_suffix=${artifact_suffix}"
  echo "head_sha=${head_sha}"
  echo "pr_number=${pr_number}"
} >> "$GITHUB_OUTPUT"
