#ifndef INCLUDE_GUARD_ASCEND_CUPY_HCCL_H
#define INCLUDE_GUARD_ASCEND_CUPY_HCCL_H

// NCCL drop-in compatibility layer over HCCL (Huawei Collective Communication
// Library).
//
// Mirrors the role of ``hip/cupy_rccl.h``: CuPy's collective binding
// (``cupy/backends/cuda/libs/nccl.pyx``) and its backend-agnostic wrapper
// (``cupy/backends/cupy_xccl.h``) are written against the *NCCL* API, so the
// Ascend backend has to provide that API on top of HCCL instead of RCCL.
//
// The two libraries are similar in shape (init / rank query / collective /
// point-to-point) but differ in the details that this header absorbs:
//
//   NCCL                                   HCCL
//   ------------------------------------   ------------------------------------
//   ncclComm_t              (void*)         HcclComm               (void*)
//   ncclUniqueId            128 B           HcclRootInfo           4108 B
//   ncclCommInitRank(...)                   HcclCommInitRootInfo(...)
//   ncclGetUniqueId(...)                    HcclGetRootInfo(...)
//   ncclCommCount/UserRank/CuDevice         HcclGetRankSize/RankId + aclrtGetDevice
//   ncclCommGetAsyncError                   HcclGetCommAsyncError
//   ncclCommAbort                           (no equivalent; suspend+destroy)
//   ncclConfig_t / ncclCommSplit            HcclCommConfig / HcclCreateSubCommConfig
//   ncclGroupStart/End (async, on stream)   HcclBatchSendRecv (one batched call)
//   ncclDataType_t (10 values, 0..9)        HcclDataType (different numbering)
//   ncclRedOp_t (sum/prod/max/min/avg)      HcclReduceOp (no avg)
//   ncclResult_t (0..6 used by cupy)        HcclResult (different numbering)
//
// Everything ``nccl.pyx``/``cupy_xccl.h`` reference must therefore be defined
// *here*, with NCCL's spelling and NCCL's numeric conventions, because the
// Python side (``cupy/backends/cuda/libs/nccl.pxd``) hardcodes NCCL's enum
// values (``NCCL_INT8 = 0`` … ``NCCL_BFLOAT16 = 9``).
//
// Emulated NCCL compatibility level
// ---------------------------------
// ``NCCL_VERSION_CODE`` is reported as 2.17.0.  That is deliberate and needs to
// stay below 2.18.0:
//
// * ``cupy/backends/cuda/libs/nccl.pyx`` guards every API on this macro.  With
//   >= 2.2 the separate-buffer ``ncclBroadcast`` path is taken (we implement
//   it), >= 2.4 enables ``abort``/``checkAsyncError`` (we implement them) and
//   >= 2.7 enables ``send``/``recv`` (we implement them).
// * At < 2.18 ``cupy_xccl.h`` mocks ``_ncclConfig_t`` with a ``splitShare``
//   field, which is exactly what ``NcclConfig`` in ``nccl.pyx`` pokes at.  At
//   >= 2.18 that type is aliased to the real ``ncclConfig_t``, which has no
//   ``splitShare`` member -> the Cython module would no longer compile.
//   Consequently ``NcclConfig``/``commSplit`` stay disabled on Ascend until the
//   Cython layer grows an Ascend branch (tracked in docs, not a silent gap).
//
// Known functional gaps (see docs/ascend/HcclBinding.md):
//   * fp64 / ncclAvg reductions: not expressible in HCCL's reduce ops.
//   * ``ncclCommSplit``: needs HcclCreateSubCommConfig + a real config struct.
//   * P2P to self (rank i -> rank i), which ``cupyx.distributed``'s
//     scatter/gather/all_to_all helpers do issue: HCCL does not guarantee it.

#include <hccl/hccl.h>

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// ``cupy/backends/cupy_xccl.h`` and ``nccl.pyx`` spell the stream parameter
// ``cudaStream_t``; on Ascend that is the ACL stream (``void*``), exactly like
// ``hip_rccl.h`` maps it to ``hipStream_t``.
typedef aclrtStream cudaStream_t;

// ---------------------------------------------------------------------------
// Emulated NCCL version
// ---------------------------------------------------------------------------
// Defined before ``cupy_xccl.h`` computes ``NCCL_VERSION_CODE`` from
// ``NCCL_MAJOR``/``NCCL_MINOR``/``NCCL_PATCH``; keep the literal in sync with
// ``NCCL_VERSION(2, 17, 0) == 21700``.
#define NCCL_MAJOR 2
#define NCCL_MINOR 17
#define NCCL_PATCH 0
#define NCCL_VERSION_CODE 21700

// ---------------------------------------------------------------------------
// Types shared with ``nccl.pyx``
// ---------------------------------------------------------------------------
// The enumerator *values* are NCCL's, not HCCL's: ``nccl.pxd`` exposes them to
// Python as ints and ``nccl.pyx`` forwards those ints unchanged, so the
// translation to ``HcclDataType``/``HcclReduceOp`` happens at call time in the
// helpers below.
typedef enum {
    ncclInt8 = 0, /* ncclChar */
    ncclChar = ncclInt8,
    ncclUint8 = 1,
    ncclInt32 = 2, /* ncclInt */
    ncclInt = ncclInt32,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, /* ncclHalf */
    ncclHalf = ncclFloat16,
    ncclFloat32 = 7, /* ncclFloat */
    ncclFloat = ncclFloat32,
    ncclFloat64 = 8, /* ncclDouble */
    ncclDouble = ncclFloat64,
    ncclBfloat16 = 9,
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4, /* not supported by HCCL */
} ncclRedOp_t;

// Only 0..6 may be produced: ``nccl.pyx``'s ``ERROR2`` table is indexed by the
// status, so anything outside that range would raise ``KeyError`` instead of a
// readable NCCL error.
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
} ncclResult_t;

typedef HcclComm ncclComm_t;

// ``HcclRootInfo`` is 4108 B while ``ncclUniqueId`` is 128 B in real NCCL.  The
// constant is taken from the HCCL header so the Cython side (which only sees
// the *name* ``NCCL_UNIQUE_ID_BYTES`` -- `cdef enum` inside a ``cdef extern``
// block is never emitted) always agrees with the struct it memcpy()s into.
// The exchange through ``cupyx.distributed``'s TCP store must accommodate this
// size, see ``cupyx/distributed/_klv_utils.py``.
#define NCCL_UNIQUE_ID_BYTES HCCL_ROOT_INFO_BYTES

typedef HcclRootInfo ncclUniqueId;

static_assert(sizeof(HcclRootInfo) == 4108,
              "unexpected HcclRootInfo size; NCCL_UNIQUE_ID_BYTES is stale");
static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
              "ncclUniqueId must be byte-identical to HcclRootInfo");

namespace cupy_hccl {

// ---------------------------------------------------------------------------
// Error / dtype / op translation
// ---------------------------------------------------------------------------

// Collapse ``HcclResult`` into the 0..6 range ``nccl.pyx`` can name.  The
// mapping is lossy by construction (HCCL has far more error codes); the
// original code is still visible through ``HcclGetErrorString`` on the HCCL
// side when debugging.
inline ncclResult_t ToNcclResult(HcclResult status) {
    switch (status) {
        case HCCL_SUCCESS:
            return ncclSuccess;
        case HCCL_E_MEMORY:
        case HCCL_E_OOM:
        case HCCL_E_DRV:
        case HCCL_E_RUNTIME:
            return ncclSystemError;
        case HCCL_E_PARA:
        case HCCL_E_PTR:
            return ncclInvalidArgument;
        case HCCL_E_NOT_SUPPORT:
        case HCCL_E_NOT_FOUND:
        case HCCL_E_UNAVAIL:
            return ncclInvalidUsage;
        case HCCL_E_REMOTE:
        case HCCL_E_NETWORK:
        case HCCL_E_SYSCALL:
        case HCCL_E_TIMEOUT:
        case HCCL_E_OPEN_FILE_FAILURE:
        case HCCL_E_TCP_CONNECT:
        case HCCL_E_ROCE_CONNECT:
        case HCCL_E_TCP_TRANSFER:
        case HCCL_E_ROCE_TRANSFER:
            return ncclRemoteError;
        default:
            return ncclInternalError;
    }
}

// NCCL enum value -> HCCL data type.  ``INT16``/``UINT16`` have no NCCL
// counterpart in ``nccl.pxd`` and are therefore never produced.
inline HcclDataType ToHcclDataType(ncclDataType_t dtype) {
    switch (dtype) {
        case ncclInt8:
            return HCCL_DATA_TYPE_INT8;
        case ncclUint8:
            return HCCL_DATA_TYPE_UINT8;
        case ncclInt32:
            return HCCL_DATA_TYPE_INT32;
        case ncclUint32:
            return HCCL_DATA_TYPE_UINT32;
        case ncclInt64:
            return HCCL_DATA_TYPE_INT64;
        case ncclUint64:
            return HCCL_DATA_TYPE_UINT64;
        case ncclFloat16:
            return HCCL_DATA_TYPE_FP16;
        case ncclFloat32:
            return HCCL_DATA_TYPE_FP32;
        case ncclFloat64:
            return HCCL_DATA_TYPE_FP64;
        case ncclBfloat16:
            return HCCL_DATA_TYPE_BFP16;
        default:
            // HCCL_E_PARA would need a HcclResult; callers check the bool form.
            return HCCL_DATA_TYPE_RESERVED;
    }
}

inline bool IsValidDataType(ncclDataType_t dtype) {
    return ToHcclDataType(dtype) != HCCL_DATA_TYPE_RESERVED;
}

// Reduction op.  ``HCCL_REDUCE_RESERVED`` marks "no equivalent" (NCCL_AVG).
inline HcclReduceOp ToHcclReduceOp(ncclRedOp_t op) {
    switch (op) {
        case ncclSum:
            return HCCL_REDUCE_SUM;
        case ncclProd:
            return HCCL_REDUCE_PROD;
        case ncclMax:
            return HCCL_REDUCE_MAX;
        case ncclMin:
            return HCCL_REDUCE_MIN;
        default:
            return HCCL_REDUCE_RESERVED;
    }
}

inline bool IsValidReduceOp(ncclRedOp_t op) {
    return ToHcclReduceOp(op) != HCCL_REDUCE_RESERVED;
}

// Element size, needed only for the out-of-place ``ncclBroadcast`` emulation.
inline size_t DataTypeSize(ncclDataType_t dtype) {
    switch (dtype) {
        case ncclInt8:
        case ncclUint8:
            return 1;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// Group support  (ncclGroupStart / ncclGroupEnd -> HcclBatchSendRecv)
// ---------------------------------------------------------------------------
// HCCL has no stream-ordered "open a group" primitive: point-to-point ops are
// matched against their peer at issue time, so a naive sequence of
// ``HcclSend``/``HcclRecv`` calls deadlocks exactly the way ungrouped NCCL does.
// ``HcclBatchSendRecv`` is the intended replacement, so a group is buffered
// here and flushed in one call at ``ncclGroupEnd``.
//
// Collectives (AllReduce/…/Broadcast) are *not* buffered: HCCL batches only
// peer-to-peer traffic, and issuing them immediately preserves the program
// order ``cupyx.distributed`` relies on (e.g. the sparse path opens a group and
// broadcasts inside it).
struct Group {
    bool active = false;
    HcclComm comm = nullptr;       // comm of the currently buffered batch
    aclrtStream stream = nullptr;  // stream of the currently buffered batch
    std::vector<HcclSendRecvItem> items;
};

// One group per thread: ``cupyx.distributed`` is single-threaded per process,
// but a user may drive collectives from several Python threads.
inline Group& CurrentGroup() {
    static thread_local Group group;
    return group;
}

inline ncclResult_t FlushGroup(Group& group) {
    if (group.items.empty()) {
        return ncclSuccess;
    }
    HcclResult status = HcclBatchSendRecv(group.items.data(),
                                          static_cast<uint32_t>(
                                              group.items.size()),
                                          group.comm, group.stream);
    group.items.clear();
    return ToNcclResult(status);
}

inline ncclResult_t AppendP2P(HcclSendRecvType type, void* buf, size_t count,
                              HcclDataType dtype, int peer, HcclComm comm,
                              aclrtStream stream) {
    Group& group = CurrentGroup();
    if (!group.active) {
        // Defensive: the public entry points already check `active`, but a
        // request must never be dropped silently.
        if (type == HCCL_SEND) {
            return ToNcclResult(HcclSend(buf, count, dtype,
                                         static_cast<uint32_t>(peer), comm,
                                         stream));
        }
        return ToNcclResult(HcclRecv(buf, count, dtype,
                                     static_cast<uint32_t>(peer), comm,
                                     stream));
    }
    // A batch must be flushed when the communicator changes: one
    // HcclBatchSendRecv call is bound to a single comm/stream pair.
    if (!group.items.empty() && group.comm != comm) {
        ncclResult_t status = FlushGroup(group);
        if (status != ncclSuccess) {
            return status;
        }
    }
    if (group.items.empty()) {
        group.comm = comm;
        group.stream = stream;
        group.items.reserve(8);  // typical: world_size sends + one recv
    }
    HcclSendRecvItem item;
    item.sendRecvType = type;
    item.buf = buf;
    item.count = static_cast<uint64_t>(count);
    item.dataType = dtype;
    item.remoteRank = static_cast<uint32_t>(peer);
    group.items.push_back(item);
    return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Weak-symbol guards
// ---------------------------------------------------------------------------
// Every communicator-management entry point is declared ``HCOMM_WEAK_SYMBOL``
// by CANN (they live in the "framework" library -- ``libhcomm.so`` on 9.x,
// ``libhccl_fwk.so`` on 8.5 -- not in ``libhccl.so``, which only holds the
// collective/P2P ops), so a build against an older/newer toolkit may resolve
// them to null.  Calling one would segfault, hence the checks.
template <typename Fn>
inline bool Available(Fn fn) {
    return fn != nullptr;
}

}  // namespace cupy_hccl

// ---------------------------------------------------------------------------
// Version / error string
// ---------------------------------------------------------------------------
inline const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess:
            return "no error";
        case ncclUnhandledCudaError:
            return "unhandled device error";
        case ncclSystemError:
            return "system error (HCCL: memory/syscall failure)";
        case ncclInternalError:
            return "internal error";
        case ncclInvalidArgument:
            return "invalid argument";
        case ncclInvalidUsage:
            return "invalid usage (unsupported by HCCL)";
        case ncclRemoteError:
            return "remote error (HCCL: network/timeout)";
        default:
            return "unknown error";
    }
}

inline ncclResult_t ncclGetVersion(int* version) {
    if (version == nullptr) {
        return ncclInvalidArgument;
    }
    *version = NCCL_VERSION_CODE;
    return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Communicator lifecycle
// ---------------------------------------------------------------------------
inline ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (uniqueId == nullptr) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclGetRootInfo)) {
        return ncclInvalidUsage;
    }
    return cupy_hccl::ToNcclResult(HcclGetRootInfo(uniqueId));
}

inline ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev,
                                    ncclUniqueId commId, int rank) {
    if (comm == nullptr || ndev <= 0 || rank < 0) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclCommInitRootInfo)) {
        return ncclInvalidUsage;
    }
    return cupy_hccl::ToNcclResult(HcclCommInitRootInfo(
        static_cast<uint32_t>(ndev), &commId, static_cast<uint32_t>(rank),
        comm));
}

inline ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev,
                                    const int* devlist) {
    if (comm == nullptr || ndev <= 0) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclCommInitAll)) {
        return ncclInvalidUsage;
    }
    // HcclCommInitAll takes the device list non-const, but does not modify it.
    return cupy_hccl::ToNcclResult(HcclCommInitAll(
        static_cast<uint32_t>(ndev), const_cast<int32_t*>(devlist), comm));
}

inline ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (comm == nullptr) {
        return ncclSuccess;
    }
    if (!cupy_hccl::Available(HcclCommDestroy)) {
        return ncclInvalidUsage;
    }
    return cupy_hccl::ToNcclResult(HcclCommDestroy(comm));
}

// NCCL's ``ncclCommAbort`` tears the communicator down without waiting for
// outstanding work.  HCCL has no abort: suspend (best effort, stops new work and
// releases in-flight requests) followed by destroy is the closest equivalent,
// and is what makes ``NcclCommunicator.abort()`` release its resources instead
// of leaking the comm.  Both steps are best-effort -- the caller has already
// decided the comm is unusable.
inline void ncclCommAbort(ncclComm_t comm) {
    if (comm == nullptr) {
        return;
    }
    if (cupy_hccl::Available(HcclCommSuspend)) {
        HcclCommSuspend(comm);
    }
    if (cupy_hccl::Available(HcclCommDestroy)) {
        HcclCommDestroy(comm);
    }
}

inline ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (comm == nullptr || count == nullptr) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclGetRankSize)) {
        return ncclInvalidUsage;
    }
    uint32_t size = 0;
    ncclResult_t status =
        cupy_hccl::ToNcclResult(HcclGetRankSize(comm, &size));
    if (status == ncclSuccess) {
        *count = static_cast<int>(size);
    }
    return status;
}

inline ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (comm == nullptr || rank == nullptr) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclGetRankId)) {
        return ncclInvalidUsage;
    }
    uint32_t id = 0;
    ncclResult_t status = cupy_hccl::ToNcclResult(HcclGetRankId(comm, &id));
    if (status == ncclSuccess) {
        *rank = static_cast<int>(id);
    }
    return status;
}

// NCCL reports the device the communicator is bound to; HCCL has no per-comm
// device query, so report the device current on this thread (which is what the
// comm was created on -- ``cupyx.distributed`` requires one device per process).
inline ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    (void)comm;  // no per-comm device query in HCCL
    if (device == nullptr) {
        return ncclInvalidArgument;
    }
    int32_t dev = -1;
    if (aclrtGetDevice(&dev) != ACL_SUCCESS) {
        return ncclSystemError;
    }
    *device = static_cast<int>(dev);
    return ncclSuccess;
}

inline ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,
                                          ncclResult_t* asyncError) {
    if (comm == nullptr || asyncError == nullptr) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::Available(HcclGetCommAsyncError)) {
        // Nothing to poll: report "no error" so callers stay functional.
        *asyncError = ncclSuccess;
        return ncclSuccess;
    }
    HcclResult hcclError = HCCL_SUCCESS;
    ncclResult_t status = cupy_hccl::ToNcclResult(
        HcclGetCommAsyncError(comm, &hcclError));
    *asyncError = cupy_hccl::ToNcclResult(hcclError);
    return status;
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------
inline ncclResult_t ncclGroupStart() {
    cupy_hccl::Group& group = cupy_hccl::CurrentGroup();
    // Drop anything a previous (unclosed) group left behind: nesting is not
    // part of the NCCL contract, and forwarding stale requests would be worse
    // than losing them.
    group.items.clear();
    group.active = true;
    return ncclSuccess;
}

inline ncclResult_t ncclGroupEnd() {
    cupy_hccl::Group& group = cupy_hccl::CurrentGroup();
    group.active = false;
    return cupy_hccl::FlushGroup(group);
}

// ---------------------------------------------------------------------------
// Collectives
// ---------------------------------------------------------------------------
inline ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                                  size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm_t comm,
                                  cudaStream_t stream) {
    if (comm == nullptr || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::IsValidReduceOp(op)) {
        return ncclInvalidUsage;  // e.g. ncclAvg
    }
    return cupy_hccl::ToNcclResult(
        HcclAllReduce(const_cast<void*>(sendbuff), recvbuff,
                      static_cast<uint64_t>(count),
                      cupy_hccl::ToHcclDataType(datatype),
                      cupy_hccl::ToHcclReduceOp(op), comm, stream));
}

inline ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclRedOp_t op, int root, ncclComm_t comm,
                               cudaStream_t stream) {
    if (comm == nullptr || root < 0 || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::IsValidReduceOp(op)) {
        return ncclInvalidUsage;
    }
    return cupy_hccl::ToNcclResult(
        HcclReduce(const_cast<void*>(sendbuff), recvbuff,
                   static_cast<uint64_t>(count),
                   cupy_hccl::ToHcclDataType(datatype),
                   cupy_hccl::ToHcclReduceOp(op),
                   static_cast<uint32_t>(root), comm, stream));
}

inline ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                      size_t recvcount,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    if (!cupy_hccl::IsValidReduceOp(op)) {
        return ncclInvalidUsage;
    }
    return cupy_hccl::ToNcclResult(
        HcclReduceScatter(const_cast<void*>(sendbuff), recvbuff,
                          static_cast<uint64_t>(recvcount),
                          cupy_hccl::ToHcclDataType(datatype),
                          cupy_hccl::ToHcclReduceOp(op), comm, stream));
}

inline ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                                  size_t sendcount, ncclDataType_t datatype,
                                  ncclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    return cupy_hccl::ToNcclResult(
        HcclAllGather(const_cast<void*>(sendbuff), recvbuff,
                      static_cast<uint64_t>(sendcount),
                      cupy_hccl::ToHcclDataType(datatype), comm, stream));
}

// ``HcclBroadcast`` is in-place only.  NCCL allows ``sendbuff != recvbuff``
// (only the root's ``sendbuff`` is read), which is what
// ``NcclCommunicator.broadcast`` in ``nccl.pyx`` passes when the caller supplies
// two distinct buffers -- emulate that by copying on the root first.  In the
// common path ``sendbuff == recvbuff`` and nothing is copied.
inline ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                                  size_t count, ncclDataType_t datatype,
                                  int root, ncclComm_t comm,
                                  cudaStream_t stream) {
    if (comm == nullptr || root < 0 || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    if (sendbuff != recvbuff) {
        if (!cupy_hccl::Available(HcclGetRankId)) {
            return ncclInvalidUsage;
        }
        uint32_t rank = 0;
        ncclResult_t status =
            cupy_hccl::ToNcclResult(HcclGetRankId(comm, &rank));
        if (status != ncclSuccess) {
            return status;
        }
        if (static_cast<int>(rank) == root) {
            size_t bytes = count * cupy_hccl::DataTypeSize(datatype);
            if (bytes != 0 &&
                aclrtMemcpyAsync(recvbuff, bytes, sendbuff, bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE,
                                 stream) != ACL_SUCCESS) {
                return ncclSystemError;
            }
        }
    }
    return cupy_hccl::ToNcclResult(
        HcclBroadcast(recvbuff, static_cast<uint64_t>(count),
                      cupy_hccl::ToHcclDataType(datatype),
                      static_cast<uint32_t>(root), comm, stream));
}

inline ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                             int root, ncclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || root < 0 || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    return cupy_hccl::ToNcclResult(
        HcclBroadcast(buff, static_cast<uint64_t>(count),
                      cupy_hccl::ToHcclDataType(datatype),
                      static_cast<uint32_t>(root), comm, stream));
}

// ---------------------------------------------------------------------------
// Point-to-point (groupable)
// ---------------------------------------------------------------------------
inline ncclResult_t ncclSend(const void* sendbuff, size_t count,
                             ncclDataType_t datatype, int peer, ncclComm_t comm,
                             cudaStream_t stream) {
    if (comm == nullptr || peer < 0 || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    return cupy_hccl::AppendP2P(HCCL_SEND, const_cast<void*>(sendbuff), count,
                                cupy_hccl::ToHcclDataType(datatype), peer, comm,
                                stream);
}

inline ncclResult_t ncclRecv(void* recvbuff, size_t count,
                             ncclDataType_t datatype, int peer, ncclComm_t comm,
                             cudaStream_t stream) {
    if (comm == nullptr || peer < 0 || !cupy_hccl::IsValidDataType(datatype)) {
        return ncclInvalidArgument;
    }
    return cupy_hccl::AppendP2P(HCCL_RECV, recvbuff, count,
                                cupy_hccl::ToHcclDataType(datatype), peer, comm,
                                stream);
}

#endif  // INCLUDE_GUARD_ASCEND_CUPY_HCCL_H
