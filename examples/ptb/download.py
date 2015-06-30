#!/usr/bin/env python
import urllib

host = 'https://raw.githubusercontent.com'
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.train.txt' % host,
    'ptb.train.txt')
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.valid.txt' % host,
    'ptb.valid.txt')
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.test.txt' % host,
    'ptb.test.txt')
