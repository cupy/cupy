// Stub header file of cuGraph

#ifndef INCLUDE_GUARD_STUB_CUPY_CUGRAPH_H
#define INCLUDE_GUARD_STUB_CUPY_CUGRAPH_H

#include "cupy_cuda_common.h"

extern "C" {

typedef enum {
    CUSPARSE_STATUS_SUCCESS=0,
} cusparseStatus_t;

}

namespace cugraph {

typedef enum {
    CUGRAPH_WEAK,
    CUGRAPH_STRONG
} cugraph_cc_t;

template <typename VT, typename ET, typename WT>
class GraphViewBase {
};

template <typename VT, typename ET, typename WT>
class GraphCompressedSparseBaseView : public GraphViewBase<VT, ET, WT> {
};

template <typename VT, typename ET, typename WT>
class GraphCSRView : public GraphCompressedSparseBaseView<VT, ET, WT> {
public:
    GraphCSRView() {}
    GraphCSRView(ET *, VT *, WT *, VT, ET) {}
};

template <typename VT, typename ET, typename WT>
void connected_components(GraphCSRView<VT, ET, WT> const &graph,
                          cugraph_cc_t connect_type, VT *labels);

}

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUGRAPH_H
