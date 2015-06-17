#!/usr/bin/env python
import urllib, zipfile, os, os.path
urllib.urlretrieve('http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip', 'trainDevTestTrees_PTB.zip')
zf = zipfile.ZipFile('trainDevTestTrees_PTB.zip')
for name in zf.namelist():
  (dirname, filename) = os.path.split(name)
  if not filename == '':
    zf.extract(name, ".")
