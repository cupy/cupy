#!/usr/bin/env python

from __future__ import annotations


import argparse
from collections import defaultdict
import os
import sys
import traceback

from github import Auth, Github


CATEGORY_LABEL_TO_SECTION = {
    'cat:feature': '### New Features',
    'cat:enhancement': '### Enhancements',
    'cat:numpy-compat': '### NumPy/SciPy Compatibility',
    'cat:performance': '### Performance Improvements',
    'cat:bug': '### Bug Fixes',
    'cat:code-fix': '### Code Fixes',
    'cat:document': '### Documentation',
    'cat:install': '### Installation',
    'cat:example': '### Examples',
    'cat:test': '### Tests',
    'cat:other': '### Others',
}


def find_milestone(repo, milestone):
    for ms in repo.get_milestones(state='all'):
        if milestone.strip() == ms.title.strip():
            return ms
    print('milestone is not found: {}'.format(milestone))
    sys.exit(1)


def get_issues(milestone, repo):
    milestone = find_milestone(repo, milestone)
    issues = repo.get_issues(milestone=milestone, state='closed')
    return issues


def check_corresponding_backport_pr(title, issue, backport_issues):
    for bp_issue in backport_issues:
        if title in bp_issue.title.strip():
            return True
    return False


def create_msg(issue, repo, tbp_issues, bp_issues, verbose):
    if verbose:
        print(issue)

    try:
        if issue.pull_request is None:
            return None
        if not repo.get_pull(issue.number).merged:
            return None
        title = issue.title.strip().replace('[backport]', '').strip()
        title = title.replace('[Backport]', '').strip()

        # Contributor name
        author = issue.user.login.strip()

        # Create the base message
        msg = '- {} (#{})'.format(title, issue.number)

        # List up all given labels
        label_names = [label.name for label in issue.get_labels()]

        err_msgs = []

        # If it's "backport" PR, retrieve the original author's name from the
        # corresponding "to-be-backported" PR
        if 'backport' in label_names:
            for tbp_issue in tbp_issues:
                if title in tbp_issue.title.strip():
                    author = tbp_issue.user.login.strip()
                    break
            else:
                err_msg = (
                    'Couldn\'t find the corresponding "to-be-backported" PR '
                    'for {}.\n'.format(issue.html_url) + 'This title may not'
                    ' be included in the original PR\'s title:{}\n'.format(
                        title))
                err_msgs.append(err_msg)

        # Use an invalid username for takeover PRs to avoid mentioning anyone.
        if 'takeover' in label_names:
            author = '??? (manually check original author: {})'.format(
                issue.html_url)

        # If it's "to-be-backported" PR, check the corresponding "backport" PR
        if 'to-be-backported' in label_names:
            if not check_corresponding_backport_pr(title, issue, bp_issues):
                err_msg = ('The PR: {} (#{}, {}) is marked as '
                           '"to-be-backported", but there seems no '
                           'corresponding "backport" PR or that is still '
                           'open.\n'.format(
                               issue.title.strip(), issue.number,
                               issue.html_url.strip()))
                err_msgs.append(err_msg)

        # Check if the category label is assigned correctly.
        cat_label_count = len([x for x in label_names if x.startswith('cat:')])
        if cat_label_count != 1:
            err_msg = ('The PR: {} (#{}, {}) must have exactly one category '
                       'label ("cat:*"), but found {}.\n'.format(
                           issue.title.strip(), issue.number,
                           issue.html_url.strip(), cat_label_count))
            err_msgs.append(err_msg)

        return label_names, issue.number, msg, author, err_msgs

    except Exception as e:
        print(str(type(e)), e, issue.user.login,
              issue.number, issue.html_url, traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--milestone', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--owner', type=str, default='cupy')
    parser.add_argument('--repo', type=str, default='cupy')
    parser.add_argument('--internal-members', type=str, default=None)
    parser.add_argument('--token', type=str,
                        default=None,
                        help='Override GITHUB_TOKEN env var')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # List of internal members, who is excluded from the list of contributors.
    internal_members = []
    if args.internal_members:
        internal_members = [
            x for x in open(args.internal_members).read().splitlines() if x]

    g = Github(auth=Auth.Token(args.token or os.environ['GITHUB_TOKEN']))
    org = g.get_organization(args.owner)
    repo = org.get_repo(args.repo)
    issues = get_issues(args.milestone, repo)

    tbp_issues = repo.get_issues(labels=[repo.get_label('to-be-backported')],
                                 state='closed')
    bp_issues = repo.get_issues(labels=[repo.get_label('backport')],
                                state='closed')

    labels = defaultdict(list)
    numbers = set()
    contributors = set()

    common_args = (repo, tbp_issues, bp_issues, args.verbose)
    ret = [create_msg(i, *common_args) for i in issues]

    all_err_msgs = []
    no_tags = []
    for r in ret:
        if r is None:
            continue
        label_names, number, msg, author, err_msgs = r
        for label in label_names:
            labels[label].append((number, msg))
        if not label_names:
            no_tags.append((number, msg))
        numbers.add(number)
        if author not in internal_members:
            contributors.add(author)
        all_err_msgs += err_msgs

    if args.out is None:
        out_fn = '{}-{}_{}.txt'.format(args.owner, args.repo, args.milestone)
    else:
        out_fn = args.out

    with open(out_fn, 'w') as fp:
        rendered_sections = defaultdict(list)
        for label, msgs in labels.items():
            section_name = CATEGORY_LABEL_TO_SECTION.get(
                label, f'### LABEL: {label}')
            rendered_sections[section_name] += msgs

        changelog_sections = list(CATEGORY_LABEL_TO_SECTION.values())
        section_names = changelog_sections + sorted(
            set(rendered_sections.keys()) - set(changelog_sections)
        )

        for section_name in section_names:
            if section_name not in rendered_sections:
                continue
            print(section_name, file=fp)
            for _, msg in sorted(rendered_sections[section_name],
                                 key=lambda x: x[0]):
                print(msg, file=fp)
            print('', file=fp)

        if no_tags:
            print('PRs without tags', file=fp)
        for (num, msg) in no_tags:
            print(msg, file=fp)

        print('{} unique PR IDs found.'.format(len(numbers)), file=fp)
        print('Contributors:\n{}'.format(
            ' '.join([f'@{u}' for u in sorted(contributors, key=str.lower)])),
            file=fp)

        if len(all_err_msgs) > 0:
            print('Found some errors:', file=fp)
            for err in all_err_msgs:
                print(err, file=fp)

    if len(all_err_msgs) > 0:
        print('Found some errors:')
        for err in all_err_msgs:
            print(err)
