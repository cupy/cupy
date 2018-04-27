# GMM example

This example contains implementation of Gaussian Mixture Model (GMM).


### How to demo
The demo contains a script that partitions data into groups using Gaussian Mixture Model.
The demo can be run by the following command.

```
python gmm.py [--gpu-id GPU_ID] [--num NUM] [--dim DIM]
              [--max-iter MAX_ITER] [--tol TOL] [--output-image OUTPUT]
```

If you run this script on environment without matplotlib renderers (e.g., non-GUI environment), setting the environmental variable `MPLBACKEND` to `Agg` may be required to use `matplotlib`. For example,

```
MPLBACKEND=Agg python gmm.py ...
```
