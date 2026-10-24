"""Pipelined host <-> device transfers for large NumPy arrays.

The default path for a pageable NumPy array is: allocate one pinned buffer of
the full size (``cudaHostAlloc`` on a pool miss), fill it with a single-
threaded NumPy copy, then issue one DMA.  For large arrays this leaves the
PCIe link idle while the host copies, runs the copy on one core, and on a
pool miss pays for pinning a fresh buffer of the full size.

Here the array is split into chunks that are staged through a small ring of
fixed-size pinned slots (pool hits after the first use, independent of the
array size).  Worker threads fill slots in parallel (NumPy releases the GIL
for the copy) and the calling thread enqueues each chunk's DMA as soon as its
slot is filled, so host copies overlap with the transfer.  All CUDA calls are
made from the calling thread, on the caller's stream.

When the requested device layout differs from the host layout (e.g. a
C-contiguous input with ``order='F'``), the bytes are transferred in the
host's own layout and reordered on the device, one chunk at a time, instead
of transposing on the host.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy

import cupy
from cupy.cuda import pinned_memory


# Arrays at least this large use the pipelined paths. The defaults come from
# a sweep on an RTX 5090 / PCIe 5 / dual-channel DDR5 machine; below these
# sizes the single-buffer paths are as fast or faster.
H2D_THRESHOLD = 16 << 20
D2H_THRESHOLD = 32 << 20
CHUNK_BYTES = 2 << 20
NUM_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))
# Pinned slots in flight per transfer (pinned memory: NUM_SLOTS * CHUNK_BYTES).
NUM_SLOTS = 2 * NUM_THREADS

_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=NUM_THREADS,
                    thread_name_prefix='cupy_transfer')
    return _executor


def _reset_after_fork():
    global _executor, _executor_lock
    _executor = None
    _executor_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _memory_order(a):
    """'C' or 'F' for a contiguous NumPy/CuPy array (C wins for 1-D)."""
    return 'C' if a.flags.c_contiguous else 'F'


def _row_chunks(shape, itemsize, order, chunk_bytes):
    """Split along the outermost axis in memory (axis 0 for C, -1 for F).

    Yields ``(start, stop, byte_offset, nbytes)``; every chunk is contiguous
    in host memory.
    """
    if len(shape) == 0:
        yield 0, 1, 0, itemsize
        return
    axis = 0 if order == 'C' else len(shape) - 1
    n = shape[axis]
    inner = itemsize
    for k, s in enumerate(shape):
        if k != axis:
            inner *= s
    step = max(1, chunk_bytes // max(inner, 1))
    for start in range(0, n, step):
        stop = min(n, start + step)
        yield start, stop, start * inner, (stop - start) * inner


def copy_host_to_device(a_gpu, a_cpu, stream):
    """Copy contiguous ``a_cpu`` into ``a_gpu`` (same shape/dtype).

    ``a_gpu`` must be C- or F-contiguous; its layout may differ from
    ``a_cpu``'s, in which case the reorder happens on the device.
    """
    src_order = _memory_order(a_cpu)
    dst_order = 'C' if a_gpu._c_contiguous else 'F'
    reorder = src_order != dst_order and a_cpu.ndim > 1
    src_u8 = a_cpu.ravel(order='K').view(numpy.uint8)
    nbytes = src_u8.size

    if reorder:
        chunks = list(_row_chunks(a_cpu.shape, a_cpu.itemsize, src_order,
                                  CHUNK_BYTES))
    else:
        chunks = [(None, None, off, min(CHUNK_BYTES, nbytes - off))
                  for off in range(0, nbytes, CHUNK_BYTES)]
    slot_bytes = max(c[3] for c in chunks)
    nslots = min(NUM_SLOTS, len(chunks))
    slots = [pinned_memory.alloc_pinned_memory(slot_bytes)
             for _ in range(nslots)]
    views = [numpy.frombuffer(s, numpy.uint8, slot_bytes) for s in slots]

    def fill(i, wait_event):
        if wait_event is not None:
            wait_event.synchronize()
        off, n = chunks[i][2], chunks[i][3]
        views[i % nslots][:n] = src_u8[off:off + n]

    def issue(i):
        start, stop, off, n = chunks[i]
        slot = slots[i % nslots]
        if not reorder:
            (a_gpu.data + off).copy_from_host_async(slot.ptr, n, stream)
            return
        if src_order == 'C':
            dst_view = a_gpu[start:stop]
        else:
            dst_view = a_gpu[..., start:stop]
        tmp = cupy.empty(dst_view.shape, a_gpu.dtype, order=src_order)
        tmp.data.copy_from_host_async(slot.ptr, n, stream)
        cupy._core.elementwise_copy(tmp, dst_view)

    ex = _get_executor()
    futures = []
    slot_events = [None] * nslots
    next_issue = 0
    with stream:
        try:
            for i in range(len(chunks)):
                if i >= nslots:
                    # Slot i % nslots is free once chunk i - nslots' DMA
                    # completes; make sure that DMA has been enqueued.
                    while next_issue <= i - nslots:
                        futures[next_issue].result()
                        issue(next_issue)
                        slot_events[next_issue % nslots] = stream.record()
                        next_issue += 1
                futures.append(
                    ex.submit(fill, i, slot_events[i % nslots]))
            while next_issue < len(chunks):
                futures[next_issue].result()
                issue(next_issue)
                next_issue += 1
        finally:
            for f in futures:
                f.cancel()
            for f in futures:
                if not f.cancelled():
                    f.exception()  # wait; workers reference the slots
    # Keep the slots alive until the last DMA has consumed them.
    pinned_memory._add_to_watch_list(stream.record(), slots)


def copy_device_to_host(a_gpu, a_cpu, stream):
    """Blocking copy of contiguous ``a_gpu`` into ``a_cpu`` (same layout)."""
    dst_u8 = a_cpu.ravel(order='K').view(numpy.uint8)
    nbytes = dst_u8.size
    chunks = [(off, min(CHUNK_BYTES, nbytes - off))
              for off in range(0, nbytes, CHUNK_BYTES)]
    nslots = min(NUM_SLOTS, len(chunks))
    slots = [pinned_memory.alloc_pinned_memory(CHUNK_BYTES)
             for _ in range(nslots)]
    views = [numpy.frombuffer(s, numpy.uint8, CHUNK_BYTES) for s in slots]

    def drain(i, event):
        event.synchronize()
        off, n = chunks[i]
        dst_u8[off:off + n] = views[i % nslots][:n]

    ex = _get_executor()
    futures = []
    src = a_gpu.data
    try:
        for i, (off, n) in enumerate(chunks):
            if i >= nslots:
                # slot reuse: its previous chunk must have been drained
                futures[i - nslots].result()
            (src + off).copy_to_host_async(slots[i % nslots].ptr, n, stream)
            futures.append(ex.submit(drain, i, stream.record()))
        for f in futures:
            f.result()
    finally:
        for f in futures:
            f.cancel()
        for f in futures:
            if not f.cancelled():
                f.exception()
