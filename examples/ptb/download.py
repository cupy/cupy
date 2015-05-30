#!/usr/bin/env python
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt', 'ptb.train.txt')
urllib.urlretrieve('https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt', 'ptb.valid.txt')
urllib.urlretrieve('https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt',  'ptb.test.txt')
