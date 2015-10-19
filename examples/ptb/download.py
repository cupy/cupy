#!/usr/bin/env python
from six.moves.urllib import request

host = 'https://raw.githubusercontent.com'
request.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.train.txt' % host,
    'ptb.train.txt')
request.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.valid.txt' % host,
    'ptb.valid.txt')
request.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.test.txt' % host,
    'ptb.test.txt')
