# Word Embedding

This is an example of word embedding.
We impelmented Mikolov's Skip-gram model and Continuous-BoW model with Hierarhcical softmax and Negative sampling.

First use `../ptb/download.py` to download `ptb.train.txt`.
And then, run `train_word2vec.py` to train and get `model.pickle` which includes embedding data.
You can find top-5 nearest embedding vectors using `search.py`.

This example is based on the following word embedding implementation in C++.
https://code.google.com/p/word2vec/
