#!/usr/bin/env python3

# Note: keep this script runnable Python 3.6 until FlexCI Python update
from __future__ import annotations

import json
import sys
import os
import urllib.error
import urllib.request


# Standard library only: installing packages here would expose the
# credentials in the environment to whatever pip downloads.
def send_slack(hook_url, text):
    req = urllib.request.Request(
        hook_url,
        data=json.dumps({'text': text}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as e:
        print('Slack notification failed: HTTP {}'.format(e.code),
              file=sys.stderr)


def main(argv):
    # Slack config: "HOOK_URL1,HOOK_URL2,HOOK_URL3,..."
    slack_config = os.environ.get('CUPY_CI_SLACK_CONFIG', None)

    subdesc = os.environ.get('FLEXCI_SUB_DESCRIPTION', '')
    if subdesc == '':
        subdesc = os.environ.get('FLEXCI_DESCRIPTION', '(no description)')
    url = os.environ.get('FLEXCI_JOB_URL', '<no url>')
    msg = argv[1]
    body = '''\
*{}:* {}

```
{}
```
'''.format(msg, url, subdesc)

    if slack_config is not None:
        for hook_url in slack_config.split(','):
            send_slack(hook_url, body)


if __name__ == '__main__':
    main(sys.argv)
