from typing import Iterable

import cupyx.distributed.array as darray


def all_chunks(
    chunks_map: dict[int, list['darray._Chunk']],
) -> Iterable[tuple[int, 'darray._Chunk']]:
    for dev, chunks in chunks_map.items():
        for chunk in chunks:
            yield dev, chunk
