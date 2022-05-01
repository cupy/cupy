#!/usr/bin/env python3

#
# FlexCI Dispatcher: Trigger FlexCI based on webhooks.
#

import argparse
import hmac
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Set
import urllib.request

import github


def _log(msg: str) -> None:
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()


def _forward_to_flexci(
        event_name: str, payload: Dict[str, Any], secret: str,
        projects: Set[str], base_url: str) -> bool:
    """
    Submits the GitHub webhook payload to FlexCI.
    """
    payload_enc = json.dumps(payload).encode('utf-8')
    project_list = ','.join(projects)
    url = f'{base_url}/x/github_webhook?project={project_list}&rule={event_name}:.%2B&quiet=true'  # NOQA
    _log(f'Request URI: {url}')
    req = urllib.request.Request(
        url,
        data=payload_enc,
        headers={
            'User-Agent': 'FlexCI-Dispatcher',
            'Content-Type': 'application/json',
            'X-GitHub-Event': event_name,
            'X-Hub-Signature': 'sha1={}'.format(
                hmac.new(secret.encode(), payload_enc, 'sha1').hexdigest()),
            'X-Hub-Signature-256': 'sha256={}'.format(
                hmac.new(secret.encode(), payload_enc, 'sha256').hexdigest()),
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
    raise RuntimeError(f'unexpected response: {response}')


def _complement_commit_status(
        event_name: str, payload: Dict[str, Any], token: str,
        projects: Set[str], context_prefix: str) -> None:
    gh_repo = github.Github(token).get_repo(payload['repository']['full_name'])
    if event_name == 'push':
        sha = payload['after']
    elif event_name == 'issue_comment':
        sha = gh_repo.get_pull(payload['issue']['number']).head.sha
    else:
        assert False

    _log(f'Checking statuses for commit {sha}')
    gh_commit = gh_repo.get_commit(sha)
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
    Returns the set of test tags requested in the comment.
    """
    for line in comment.splitlines():
        match = re.fullmatch(r'/test ([\w,\- ]+)', line)
        if match is not None:
            return set([x.strip() for x in match.group(1).split(',')])
    return None


def parse_args(argv: Any) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--event', type=str, required=True, choices=['issue_comment', 'push'],
        help='The name of the event')
    parser.add_argument(
        '--webhook', type=str, required=True,
        help='Path to the JSON file containing the webhook payload')
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

    event_name = options.event
    with open(options.webhook, 'rb') as f:
        payload = json.load(f)
    with open(options.projects) as f2:
        project_tags = json.load(f2)

    requested_tags = None
    if event_name == 'push':
        requested_tags = {'@push'}
        _log('Requesting tests with @push tag')
    elif event_name == 'issue_comment':
        action = payload['action']
        if action != 'created':
            _log(f'Invalid issue_comment action: {action}')
            return 1

        requested_tags = extract_requested_tags(payload['comment']['body'])
        if requested_tags is None:
            _log('No test requested in comment.')
            return 0

        association = payload['comment']['author_association']
        if association not in ('OWNER', 'MEMBER'):
            _log(f'Tests cannot be triggered by {association}')
            return 1

        _log(f'Requesting tests with tags: {requested_tags}')
    else:
        _log(f'Invalid event name: {event_name}')
        return 1

    projects_dispatch: Set[str] = set()
    projects_skip: Set[str] = set()
    for project, tags in project_tags.items():
        _log(f'Project: {project} (tags: {tags})')
        if len(set(tags) & requested_tags) != 0:
            projects_dispatch.add(project)
        else:
            projects_skip.add(project)

    if len(projects_dispatch) == 0:
        if requested_tags == {'skip'}:
            _log('Skipping all projects as requested')
        else:
            _log('No projects matched with the requested tag')
            return 1
    else:
        _log(f'Dispatching projects: {projects_dispatch}')
        success = _forward_to_flexci(
            event_name, payload, webhook_secret, projects_dispatch,
            options.flexci_uri)
        if not success:
            _log('Failed to dispatch')
            return 1

    if len(projects_skip) != 0:
        _complement_commit_status(
            event_name, payload, github_token, projects_skip,
            options.flexci_context)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
