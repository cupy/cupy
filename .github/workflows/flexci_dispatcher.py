#!/usr/bin/env python3

#
# FlexCI Dispatcher: Trigger FlexCI based on comments.
#

import argparse
import hmac
import json
import os
import re
import sys
from typing import Any, Optional, Set
import urllib.request

import github


def _log(msg: str) -> None:
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()


def _forward_to_flexci(
        payload: bytes, secret: str, projects: Set[str],
        base_url: str) -> bool:
    """
    Submits the GitHub webhook payload to FlexCI.
    """
    project_list = ','.join(projects)
    url = f'{base_url}/x/github_webhook?project={project_list}&rule=^issue_comment:test$&quiet=true'  # NOQA
    _log(f'Request URI: {url}')
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'User-Agent': 'FlexCI-Dispatcher',
            'Content-Type': 'application/json',
            'X-GitHub-Event': 'issue_comment',
            'X-Hub-Signature': 'sha1={}'.format(
                hmac.new(secret.encode(), payload, 'sha1').hexdigest()),
            'X-Hub-Signature-256': 'sha256={}'.format(
                hmac.new(secret.encode(), payload, 'sha256').hexdigest()),
        },
    )
    with urllib.request.urlopen(req) as res:
        response = json.loads(res.read())
    if 'job_ids' in response:
        for job in response['job_ids']:
            _log(f'Triggered: {base_url}/r/job/{job["id"]}')
        return True
    elif 'message' in response:
        _log(f'Failed to submit webhook payload: {response["message"]}')
        return False
    raise RuntimeError('unexpected response: {response}')


def _complement_commit_status(
        repo: str, pull_req: int, token: str,
        projects: Set[str], context_prefix: str) -> None:
    gh_repo = github.Github(token).get_repo(repo)
    gh_commit = gh_repo.get_commit(gh_repo.get_pull(pull_req).head.sha)
    _log(f'Checking statuses: {repo}, PR #{pull_req}, commit {gh_commit.sha}')
    contexts = [s.context for s in gh_commit.get_statuses()]
    for prj in projects:
        context = f'{context_prefix}/{prj}'
        if context in contexts:
            # Preserve status set via previous (real) CI run.
            continue
        _log(f'Setting status as skipped: {context}')
        gh_commit.create_status(
            state='success', description='Skipped', context=context)


def extract_requested_tags(comment: str) -> Optional[Set[str]]:
    """
    Returns the list of test tags requested in the comment.
    """
    for line in comment.splitlines():
        match = re.fullmatch(r'/test ([\w,\- ]+)', line)
        if match is not None:
            return set([x.strip() for x in match.group(1).split(',')])
    return None


def parse_args(argv: Any) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--webhook', type=str, required=True,
        help='Path to the JSON file containing issue_comment webhook payload')
    parser.add_argument(
        '--projects', type=str, required=True,
        help='Path to the JSON file containing map from FlexCI project to '
             'list of tags')
    parser.add_argument(
        '--flexci-uri', type=str, default='https://ci.preferred.jp',
        help='Base URI of the FlexCI server (default: %(default)s)')
    parser.add_argument(
        '--flexci-context', type=str, default='pfn-public-ci',
        help='Context prefix of the FlexCI server (default: %(default)s)')
    return parser.parse_args(argv[1:])


def main(argv: Any) -> int:
    options = parse_args(argv)
    webhook_secret = str(os.environ['FLEXCI_WEBHOOK_SECRET'])
    github_token = str(os.environ['GITHUB_TOKEN'])

    with open(options.webhook, 'rb') as f:
        payload = f.read()
    with open(options.projects) as f:  # type: ignore
        project_tags = json.load(f)

    payload_obj = json.loads(payload)
    if payload_obj['action'] != 'created':
        _log('Invalid action')
        return 1

    requested_tags = extract_requested_tags(payload_obj['comment']['body'])
    if requested_tags is None:
        _log('No test requested in comment.')
        return 0
    _log(f'Test tags requested: {requested_tags}')

    association = payload_obj['comment']['author_association']
    if association not in ('OWNER', 'MEMBER'):
        _log(f'Tests cannot be triggered by {association}')
        return 1

    projects_dispatch: Set[str] = set()
    projects_skipped: Set[str] = set()
    for project, tags in project_tags.items():
        _log(f'Project: {project} (tags: {tags})')
        if len(set(tags) & requested_tags) != 0:
            projects_dispatch.add(project)
        else:
            projects_skipped.add(project)

    if len(projects_dispatch) == 0:
        _log('No projects matched with the requested tag')
        return 1

    _log(f'Dispatching projects: {projects_dispatch}')
    success = _forward_to_flexci(
        payload, webhook_secret, projects_dispatch, options.flexci_uri)
    if not success:
        _log('Failed to dispatch')
        return 1

    if len(projects_skipped) != 0:
        _complement_commit_status(
            payload_obj['repository']['full_name'],
            payload_obj['issue']['number'],
            github_token,
            projects_skipped,
            options.flexci_context)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
