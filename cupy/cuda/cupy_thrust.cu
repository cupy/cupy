#include <cupy/type_dispatcher.cuh>
#include "cupy_thrust.h"


//
// APIs exposed to CuPy
//

/* -------- sort -------- */

void thrust_sort(int dtype_id, void *data_start, size_t *keys_start,
    const std::vector<ptrdiff_t>& shape, intptr_t stream, void* memory) {

    return;
}
