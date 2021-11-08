#!/usr/bin/env python

import sys
import os


def main(argv):
    # Slack config: "HOOK_URL1,HOOK_URL2,HOOK_URL3,..."
    slack_config = os.environ.get('CUPY_CI_SLACK_CONFIG', None)

    # Gitter config: "TOKEN:ROOM1,ROOM2,ROOM3,..."
    gitter_config = os.environ.get('CUPY_CI_GITTER_CONFIG', None)

    desc = os.environ.get('FLEXCI_DESCRIPTION', '<no description>')
    subdesc = os.environ.get('FLEXCI_SUB_DESCRIPTION', '')
    url = os.environ.get('FLEXCI_JOB_URL', '')
    msg = argv[1]
    body = f'{desc}\n{subdesc}\n{msg}\n{url}'

    if slack_config is not None:
        from slack_sdk.webhook import WebhookClient
        for hook_url in slack_config.split(','):
            slack = WebhookClient(hook_url)
            slack.send(text=body)

    if gitter_config is not None:
        from gitterpy.client import GitterClient
        token, rooms = gitter_config.split(':')
        gitter = GitterClient(token)
        for room in rooms.split(','):
            gitter.messages.send(room, body)


if __name__ == '__main__':
    main(sys.argv)
