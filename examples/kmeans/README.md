# kmeans example

This example contains implementation of K-means clustering.


### How to demo
The demo contains a script that partitions data into groups using K-means clustering.
The demo can be run by the following command.

```
python kmeans.py [--gpu-id GPU_ID] [--n-clusters N_CLUSTERS] [--num NUM]
                 [--max-iter MAX_ITER] [--use-custom-kernel]
                 [--output-image OUTPUT_IMAGE]
```

If you run this script on environment without matplotlib renderers (e.g., non-GUI environment), setting the environmental variable `MPLBACKEND` to `Agg` may be required to use `matplotlib`. For example,

```
MPLBACKEND=Agg python kmeans.py ...
```
