#!/usr/bin/env python

import argparse
import datetime
from multiprocessing import Pool
import os
import traceback

from github import Github


def check_tbp_issue(
        issue, repo, bp_issues, grace_dt, interactive, verbose):
    # Check each closed issues labeled with "to-be-backported"
    # bp_issues MUST be desc-sorted by creation date.

    if verbose:
        print(issue)

    try:
        if issue.pull_request is None:
            # No check needed for issues.
            return

        pr = repo.get_pull(issue.number)
        if not pr.merged:
            # No check needed for unmerged PRs.
            return

        found_backport = False
        needs_attention = False
        for bp_issue in bp_issues:
            if bp_issue.created_at < issue.created_at:
                # As the creation date of backport PR should always be newer
                # than that of the original PR, terminate iteration.
                break
            if issue.title.strip() in bp_issue.title.strip():
                found_backport = True
                break
        if not found_backport and (issue.updated_at < grace_dt):
            # The grace period has passed, ping the assignee.
            mention_to = pr.merged_by.login
            if mention_to.endswith('[bot]'):
                # Avoid notifying mergify.
                if pr.assignee is not None:
                    mention_to = pr.assignee.login
                else:
                    # No assignee set to PR, mention to the original author.
                    mention_to = pr.user.login

            if interactive:
                print('PR #{} (@{}): {} (last update: {})'.format(
                    issue.number, mention_to, issue.title, issue.updated_at))
                return
            msg = ('@{} This pull-request is marked as `to-be-backported`, '
                   'but the corresponding backport PR could not be found. '
                   'Could you check?'.format(mention_to))
            # N.B. leaving the comment will "update" the pull request, which
            # in result pause another notification for `inactive_days`.
            issue.create_comment(msg)

    except Exception as e:
        print(str(type(e)), e, issue.user.login,
              issue.number, issue.html_url, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', type=str, default='cupy')
    parser.add_argument('--repo', type=str, default='cupy')
    parser.add_argument('--token', type=str,
                        default=os.environ.get('GITHUB_TOKEN', None))
    parser.add_argument('--within-days', type=int, default=30,
                        help='Pull requests updated within the '
                             'specified days will be checked')
    parser.add_argument('--inactive-days', type=int, default=7,
                        help='Pull requests (with to-be-backported label) '
                             'inactive for the specified days will be '
                             'notified')
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--interactive', action='store_true',
                        help='Do not post a comment (for debugging)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    g = Github(args.token)
    org = g.get_organization(args.owner)
    repo = org.get_repo(args.repo)

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    after_dt = now - datetime.timedelta(args.within_days)
    grace_dt = now - datetime.timedelta(args.inactive_days)
    print(f'"to-be-backported" pull-requests updated after {after_dt} will be '
          f'checked, and those inactive since {grace_dt} will be notified.')
    tbp_issues = repo.get_issues(
        labels=[repo.get_label('to-be-backported')],
        state='closed',
        sort='updated',
        since=after_dt,
    )
    bp_issues = repo.get_issues(
        labels=[repo.get_label('backport')],
        state='all',
        sort='created',
        direction='desc',
    )

    count = 0
    p = Pool(processes=args.processes)
    for issue in tbp_issues:
        p.apply_async(check_tbp_issue, (issue, repo, bp_issues, grace_dt, args.interactive, args.verbose))
        count += 1
    p.close()
    print('Found {} issues to check...'.format(count))
    p.join()


if __name__ == '__main__':
    main()
