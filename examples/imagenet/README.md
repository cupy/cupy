# Large Scale ConvNets

## Requirements

- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an experimental example of learning from the ILSVRC2012 classification dataset.
It requires the training and validation dataset of following format:

* Each line contains one training example.
* Each line consists of two elements separated by space(s).
* The first element is a path to 256x256 RGB image.
* The second element is its ground truth label from 0 to 999.

The text format is equivalent to what Caffe uses for ImageDataLayer.
This example currently does not include dataset preparation script.

This example requires "mean file" which is computed by `compute_mean.py`.
