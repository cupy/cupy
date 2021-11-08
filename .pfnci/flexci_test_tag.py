#!/usr/bin/env python

# Note: keep this script runnable Python 3.6 until FlexCI Python update

import os
import re
import sys

import github


GITHUB_REPOSITORY = 'cupy/cupy'


def _log(msg):
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()


def get_requested_tags(github_token, description):
    # Returns a set of test tags requested in the pull-request comment.
    # The comment format is the test phrase followed by comma-searated tags.
    # e.g., "/test mini,doctest"
    match = re.search(r'/pull/(\d+)#issuecomment-(\d+)', description)
    if match is None:
        raise RuntimeError(
            'Cannot detect information from FLEXCI_DESCRIPTION: {}'.format(
                description))
    pr_id = int(match.group(1))
    comment_id = int(match.group(2))
    _log('Triggered by pull-request: #{} (comment {})'.format(
        pr_id, comment_id))

    repo = github.Github(github_token).get_repo(GITHUB_REPOSITORY)
    comment = repo.get_issue(pr_id).get_comment(comment_id)
    for line in comment.body.splitlines():
        match = re.fullmatch(r'/test\s+([\w,\-]+)', line)
        if match is not None:
            return set(match.group(1).split(','))
    _log('No test tags specified in comment')
    return None


def main(argv):
    # Prints "yes" if the current test is requested to run.
    # Prints "no" otherwise.

    # Comma-separated list of tags of the current test.
    tags = set(argv[1].split(','))

    github_token = os.environ.get('GITHUB_TOKEN', None)
    description = os.environ.get('FLEXCI_DESCRIPTION', '')
    req_tags = get_requested_tags(github_token, description)

    _log('Test tags: {}'.format(tags))
    _log('Requested tags: {}'.format(req_tags))

    if req_tags is None:
        # No tags requested; run all tests.
        print('yes')
        return

    for req_tag in req_tags:
        if req_tag in tags:
            print('yes')
            return
    print('no')


if __name__ == '__main__':
    main(sys.argv)
