#!/usr/bin/env python3

#
# FlexCI Dispatcher: Trigger FlexCI based on webhooks.
#
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import sys
from typing import Any
import urllib.request

import github


def _log(msg: str) -> None:
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()


def _forward_to_flexci(
        event_name: str, payload: dict[str, Any], secret: str,
        projects: set[str], base_url: str) -> bool:
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


def _fill_commit_status(
        event_name: str, payload: dict[str, Any], token: str,
        projects: set[str], force_skip: bool, context_prefix: str,
        base_url: str) -> None:
    gh = github.Github(auth=github.Auth.Token(token))
    gh_repo = gh.get_repo(payload['repository']['full_name'])
    if event_name == 'push':
        sha = payload['after']
    elif event_name == 'issue_comment':
        gh_pr = gh_repo.get_pull(payload['issue']['number'])
        sha = gh_pr.head.sha
    else:
        assert False

    _log(f'Retrieving commit {sha}')
    gh_commit = gh_repo.get_commit(sha)

    _log('Setting dashboard url to commit status')
    gh_commit.create_status(
        state='success',
        context=f'{context_prefix} (dashboard)',
        target_url=f'{base_url}/p/dashboard_by_commit_id?commit_id={sha}',
    )

    if len(projects) == 0:
        _log('No projects to complement commit status')
        return

    # Get the latest commit status for each context.
    _log(f'Checking statuses for commit {sha}')
    contexts = {
        s.context: s
        for s in sorted(gh_commit.get_statuses(), key=lambda s: s.updated_at)
    }
    force_skip_contexts: list[str] = []
    for prj in projects:
        context = f'{context_prefix}/{prj}'
        if context in contexts:
            # If force-skip is requested, overwrite status later.
            # Otherwise preserve status set via previous (real) CI run.
            if contexts[context].state != 'success' and force_skip:
                force_skip_contexts.append(context)
            continue
        _log(f'Setting status as skipped: {context}')
        gh_commit.create_status(
            state='success', description='Skipped', context=context)

    if len(force_skip_contexts) != 0:
        assert gh_pr is not None
        force_skip_comment = gh_pr.create_issue_comment(
            'The following tests were force-skipped:\n\n' +
            '\n'.join(
                f'- [`{c}`]({contexts[c].target_url}): '
                f'{contexts[c].state} - {contexts[c].description}'
                for c in sorted(force_skip_contexts)
            )
        )
        gh_commit.create_status(
            state='failure',
            context=f'{context_prefix} (force-skip)',
            target_url=force_skip_comment.html_url,
        )
        for context in force_skip_contexts:
            _log(f'FORCE SKIP: overwriting status for {context}')
            gh_commit.create_status(
                state='success',
                context=context,
                description=f'❌ {contexts[context].description} (force-skip)',
                target_url=contexts[context].target_url,
            )


def extract_requested_tags(comment: str) -> set[str] | None:
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
        '--event', type=str, required=True,
        choices=[
            'issue_comment', 'push', 'pull_request', 'pull_request_target'],
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
    parser.add_argument(
        '--external-tag', action='append', default=[],
        help='Test tags to be ignored by FlexCI Dispatcher')
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

    # Transform events to ones that FlexCI Legacy supports.
    if event_name in ('pull_request', 'pull_request_target'):
        payload_pr = payload['pull_request']
        if payload['action'] == 'closed' and payload_pr['merged']:
            merge_commit_sha = payload_pr['merge_commit_sha']
            event_name = 'push'
            payload = {
                'ref': f'refs/heads/{payload_pr["base"]["ref"]}',
                'after': merge_commit_sha,
                'head_commit': {
                    'id': merge_commit_sha,
                    'message': f'Merge pull request #{payload["number"]}',
                    'url': f'https://github.com/{payload_pr["base"]["repo"]["full_name"]}/commit/{merge_commit_sha}',
                    'timestamp': payload_pr['merged_at'],
                },
                'repository': payload['repository'],
                'sender': payload['sender'],
            }
            _log(
                'Transformed the pull_request event as push event: ' +
                json.dumps(payload, indent=2))
        else:
            _log('Only merged pull-request events can be handled')
            return 1

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
        if len(requested_tags - set(options.external_tag)) == 0:
            _log('All tests requested are not for FlexCI')
            requested_tags = {'skip'}

        # Note: this is not for security but to show a friendly message.
        # FlexCI server also validates the membership of the user triggered.
        association = payload['comment']['author_association']
        if association not in ('OWNER', 'MEMBER'):
            _log(f'Tests cannot be triggered by {association}')
            return 1

        _log(f'Requesting tests with tags: {requested_tags}')
    else:
        _log(f'Invalid event name: {event_name}')
        return 1

    projects_dispatch: set[str] = set()
    projects_skip: set[str] = set()
    for project, tags in project_tags.items():
        dispatch = (len(set(tags) & requested_tags) != 0)
        if dispatch:
            projects_dispatch.add(project)
        else:
            projects_skip.add(project)
        _log(f'Project: {"✅" if dispatch else "🚫"} {project} (tags: {tags})')

    force_skip = False
    if len(projects_dispatch) == 0:
        if requested_tags == {'skip'}:
            _log('Skipping all projects as requested')
        elif requested_tags == {'force-skip'}:
            _log('Force skipping all projects as requested')
            force_skip = True
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

    _fill_commit_status(
        event_name, payload, github_token, projects_skip, force_skip,
        options.flexci_context, options.flexci_uri)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
