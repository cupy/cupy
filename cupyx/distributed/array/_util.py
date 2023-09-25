from typing import Iterable

import cupyx.distributed.array as darray


def all_chunks(
    chunks_map: dict[int, list['darray.Chunk']],
) -> Iterable[tuple[int, 'darray.Chunk']]:
    for dev, chunks in chunks_map.items():
        for chunk in chunks:
            yield dev, chunk

def all_indices(
    index_map: dict[int, list[tuple[slice, ...]]],
) -> Iterable[tuple[int, tuple[slice, ...]]]:
    for dev, idxs in index_map.items():
        for idx in idxs:
            yield dev, idx
