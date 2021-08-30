import cupy.cuda.nccl

if cupy.cuda.nccl.available:
    from cupyx.distributed._comm import NCCLBackend  # NOQA
