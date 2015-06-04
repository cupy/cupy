# Recurrent Net Language Model

This is an example of a recurrent net for language modeling.
The network is trained to predict the word given the preceding word sequence.

This example is based on the following RNNLM implementation written in Torch7.
https://github.com/tomsercu/lstm

This example requires the dataset to be downloaded by the script `download.py`.
If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.
