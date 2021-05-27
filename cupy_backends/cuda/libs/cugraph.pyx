# distutils: language = c++

from libc.stdint cimport uintptr_t

cdef extern from '../../cupy_cugraph.h' namespace 'cugraph':

    ctypedef enum cugraph_cc_t:
        CUGRAPH_WEAK "cugraph::cugraph_cc_t::CUGRAPH_WEAK"
        CUGRAPH_STRONG "cugraph::cugraph_cc_t::CUGRAPH_STRONG"
        NUM_CONNECTIVITY_TYPES "cugraph::cugraph_cc_t::NUM_CONNECTIVITY_TYPES"

    cdef cppclass GraphViewBase[VT, ET, WT]:
        GraphViewBase(WT*, VT, ET)

    cdef cppclass GraphCompressedSparseBaseView[VT, ET, WT](
            GraphViewBase[VT, ET, WT]):
        GraphCompressedSparseBaseView(const VT *, const ET *, const WT *,
                                      size_t, size_t)

    cdef cppclass GraphCSRView[VT, ET, WT](
            GraphCompressedSparseBaseView[VT, ET, WT]):
        GraphCSRView()
        GraphCSRView(const ET *, const VT *, const WT *, size_t, size_t)

    cdef void connected_components[VT, ET, WT](
        const GraphCSRView[VT, ET, WT] &graph, cugraph_cc_t connect_type,
        VT *labels) except +

    # Built time version
    int CUGRAPH_VERSION_MAJOR
    int CUGRAPH_VERSION_MINOR
    int CUGRAPH_VERSION_PATCH


def weakly_connected_components(csr, labels):
    cdef uintptr_t p_indptr = csr.indptr.data.ptr
    cdef uintptr_t p_indices = csr.indices.data.ptr
    num_verts = csr.shape[0]
    num_edges = csr.nnz
    cdef uintptr_t p_labels = labels.data.ptr
    cdef GraphCSRView[int, int, float] g
    g = GraphCSRView[int, int, float](<int*>p_indptr, <int*>p_indices,
                                      <float*>NULL, num_verts, num_edges)
    connected_components(g, <cugraph_cc_t>CUGRAPH_WEAK, <int*>p_labels)


def strongly_connected_components(csr, labels):
    cdef uintptr_t p_indptr = csr.indptr.data.ptr
    cdef uintptr_t p_indices = csr.indices.data.ptr
    num_verts = csr.shape[0]
    num_edges = csr.nnz
    cdef uintptr_t p_labels = labels.data.ptr
    cdef GraphCSRView[int, int, float] g
    g = GraphCSRView[int, int, float](<int*>p_indptr, <int*>p_indices,
                                      <float*>NULL, num_verts, num_edges)
    connected_components(g, <cugraph_cc_t>CUGRAPH_STRONG, <int*>p_labels)
