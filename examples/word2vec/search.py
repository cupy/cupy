#!/usr/bin/env python
import numpy
import six

n_result = 5  # number of search result to show


with open('word2vec.model', 'r') as f:
    ss = f.readline().split()
    n_vocab, n_units = int(ss[0]), int(ss[1])
    word2index = {}
    index2word = {}
    w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
    for i, line in enumerate(f):
        ss = line.split()
        assert len(ss) == n_units + 1
        word = ss[0]
        word2index[word] = i
        index2word[i] = word
        w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)


s = numpy.sqrt((w * w).sum(1))
w /= s.reshape((s.shape[0], 1))  # normalize

try:
    while True:
        q = six.moves.input('>> ')
        if q not in word2index:
            print('"{0}" is not found'.format(q))
            continue
        v = w[word2index[q]]
        similarity = w.dot(v)
        print('query: {}'.format(q))
        count = 0
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):
                continue
            if index2word[i] == q:
                continue
            print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break

except EOFError:
    pass
