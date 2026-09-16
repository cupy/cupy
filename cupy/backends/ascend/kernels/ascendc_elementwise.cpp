// AscendC elementwise custom kernels for numpy-ascend.
//
// Calling convention (host side: cupy/backends/ascend/acl_custom_kernels.h):
//   extern "C" __global__ __aicore__ void ascendc_<name>(
//       GM_ADDR out0, GM_ADDR out1,       // out1 = nullptr for single-output ops
//       GM_ADDR in0, GM_ADDR in1,         // in1 = nullptr for unary ops
//       uint64_t n, uint64_t perBlock)
// where `n` is the element count and each AI-core instance handles the slice
// [blockIdx*perBlock, min(n, (blockIdx+1)*perBlock)).
//
// Host guarantees: perBlock is a multiple of 128; n is padded by the host to a
// tile multiple so fixed-size tail copies never read outside the allocation.
//
// NOTE (verification level): these kernels compile (L1) but have NOT been run
// on hardware; numerical correctness must be validated on a 910B device.
// Known simplifications: angle uses atan (no quadrant fix); frexp/shift
// edge cases (0, negative, overflow) are best-effort.
#include "kernel_operator.h"
#include "lib/math/trunc.h"
#include "lib/math/frac.h"
#include "lib/math/floor.h"
#include "lib/math/log.h"
#include "lib/math/atan.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE = 2048;      // elements per inner tile (multiple of 128)
constexpr uint32_t ALIGN = 8;        // 32B / 4B elements
constexpr float LN2 = 0.6931471805599453f;
constexpr uint32_t SCRATCH = 2 * TILE;   // floats per scratch buffer

// three independent UB scratch buffers (LocalTensor offset views are not
// supported by TBuf::Get, so each lane gets its own buffer)
struct Scratch {
    TBuf<TPosition::VECCALC> a, b, c;
    __aicore__ inline void Init(TPipe& pipe) {
        pipe.InitBuffer(a, SCRATCH * sizeof(float));
        pipe.InitBuffer(b, SCRATCH * sizeof(float));
        pipe.InitBuffer(c, SCRATCH * sizeof(float));
    }
};

__aicore__ inline uint32_t tile_count(uint64_t remaining) {
    uint32_t cnt = (remaining < TILE) ? static_cast<uint32_t>(remaining) : TILE;
    return (cnt + ALIGN - 1) & ~(ALIGN - 1);
}
}  // namespace

// ---------------------------------------------------------------------------
// unary float kernel driver: z[i] = op(x[i])
// ---------------------------------------------------------------------------
template <typename F>
__aicore__ inline void UnaryKernelF32(GM_ADDR z, GM_ADDR x, uint64_t myN, F op) {
    GlobalTensor<float> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    zGm.SetGlobalBuffer((__gm__ float*)z);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qx;
    TQue<TPosition::VECOUT, 2> qz;
    Scratch scratch;
    pipe.InitBuffer(qx, 2, TILE * sizeof(float));
    pipe.InitBuffer(qz, 2, TILE * sizeof(float));
    scratch.Init(pipe);

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        LocalTensor<float> lx = qx.AllocTensor<float>();
        DataCopy(lx, xGm[off], cnt);
        qx.EnQue(lx);

        LocalTensor<float> dx = qx.DeQue<float>();
        LocalTensor<float> dz = qz.AllocTensor<float>();
        op(dz, dx, cnt, scratch);
        qz.EnQue(dz);
        qx.FreeTensor(dx);

        LocalTensor<float> oz = qz.DeQue<float>();
        DataCopy(zGm[off], oz, cnt);
        qz.FreeTensor(oz);
    }
}

// ---------------------------------------------------------------------------
// binary kernel driver (same dtype float/int): z[i] = op(x[i], y[i])
// ---------------------------------------------------------------------------
template <typename TX, typename TY, typename TZ, typename F>
__aicore__ inline void BinaryKernel(GM_ADDR z, GM_ADDR x, GM_ADDR y,
                                    uint64_t myN, F op) {
    GlobalTensor<TX> xGm;
    GlobalTensor<TY> yGm;
    GlobalTensor<TZ> zGm;
    xGm.SetGlobalBuffer((__gm__ TX*)x);
    yGm.SetGlobalBuffer((__gm__ TY*)y);
    zGm.SetGlobalBuffer((__gm__ TZ*)z);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qx, qy;
    TQue<TPosition::VECOUT, 2> qz;
    Scratch scratch;
    pipe.InitBuffer(qx, 2, TILE * sizeof(TX));
    pipe.InitBuffer(qy, 2, TILE * sizeof(TY));
    pipe.InitBuffer(qz, 2, TILE * sizeof(TZ));
    scratch.Init(pipe);

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        LocalTensor<TX> lx = qx.AllocTensor<TX>();
        LocalTensor<TY> ly = qy.AllocTensor<TY>();
        DataCopy(lx, xGm[off], cnt);
        DataCopy(ly, yGm[off], cnt);
        qx.EnQue(lx);
        qy.EnQue(ly);

        LocalTensor<TX> dx = qx.DeQue<TX>();
        LocalTensor<TY> dy = qy.DeQue<TY>();
        LocalTensor<TZ> dz = qz.AllocTensor<TZ>();
        op(dz, dx, dy, cnt, scratch);
        qz.EnQue(dz);
        qx.FreeTensor(dx);
        qy.FreeTensor(dy);

        LocalTensor<TZ> oz = qz.DeQue<TZ>();
        DataCopy(zGm[off], oz, cnt);
        qz.FreeTensor(oz);
    }
}

// block-slice guard shared by all kernels
#define CUSTOM_KERNEL_GUARD \
    uint64_t base = GetBlockIdx() * perBlock; \
    uint64_t myN = perBlock; \
    if (base + myN > n) { myN = (n > base) ? (n - base) : 0; } \
    if (myN == 0) { return; } \
    uint64_t padded = myN; \
    padded = ((padded + TILE - 1) / TILE) * TILE; \
    (void)base; (void)padded

// ---------------------------------------------------------------------------
// left_shift / right_shift: int32 x int32 -> int32
//   dav_c220 has no vector tensor-tensor shift primitive, so use
//   x * 2^b (resp. floor(x / 2^b)) via Exp/Mul/Cast. Exact for 0 <= b and
//   results inside int32 range.
// ---------------------------------------------------------------------------
struct LeftShiftI32Op {
    __aicore__ inline void operator()(LocalTensor<int32_t>& dz, LocalTensor<int32_t>& dx,
                                      LocalTensor<int32_t>& dy, uint32_t cnt, Scratch& scratch) {
        LocalTensor<float> xf = scratch.a.Get<float>(SCRATCH);
        LocalTensor<float> pf = scratch.b.Get<float>(SCRATCH);
        LocalTensor<float> yf = scratch.c.Get<float>(SCRATCH);
        Cast(xf, dx, RoundMode::CAST_NONE, cnt);
        Cast(pf, dy, RoundMode::CAST_NONE, cnt);       // b as float
        Muls(pf, pf, LN2, cnt);
        Exp(pf, pf, cnt);                              // 2^b
        Mul(yf, xf, pf, cnt);                          // x * 2^b
        Cast(dz, yf, RoundMode::CAST_RINT, cnt);
    }
};

extern "C" __global__ __aicore__ void ascendc_left_shift_i32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR y, uint64_t n, uint64_t perBlock) {
    (void)o1;
    CUSTOM_KERNEL_GUARD;
    BinaryKernel<int32_t, int32_t, int32_t>(z, x, y, myN, LeftShiftI32Op());
}

struct RightShiftI32Op {
    __aicore__ inline void operator()(LocalTensor<int32_t>& dz, LocalTensor<int32_t>& dx,
                                      LocalTensor<int32_t>& dy, uint32_t cnt, Scratch& scratch) {
        LocalTensor<float> xf = scratch.a.Get<float>(SCRATCH);
        LocalTensor<float> pf = scratch.b.Get<float>(SCRATCH);
        LocalTensor<float> yf = scratch.c.Get<float>(SCRATCH);
        Cast(xf, dx, RoundMode::CAST_NONE, cnt);
        Cast(pf, dy, RoundMode::CAST_NONE, cnt);
        Muls(pf, pf, -LN2, cnt);
        Exp(pf, pf, cnt);                              // 2^-b
        Mul(yf, xf, pf, cnt);                          // x / 2^b
        Cast(dz, yf, RoundMode::CAST_FLOOR, cnt);      // arithmetic shift = floor
    }
};

extern "C" __global__ __aicore__ void ascendc_right_shift_i32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR y, uint64_t n, uint64_t perBlock) {
    (void)o1;
    CUSTOM_KERNEL_GUARD;
    BinaryKernel<int32_t, int32_t, int32_t>(z, x, y, myN, RightShiftI32Op());
}

// ---------------------------------------------------------------------------
// modf: float -> (frac, integral)
// ---------------------------------------------------------------------------
struct ModfF32Op {
    __aicore__ inline void operator()(LocalTensor<float>& dz, LocalTensor<float>& dx,
                                      uint32_t cnt, Scratch& scratch) {
        (void)scratch;
        Trunc(dz, dx, cnt);          // integral part -> out1
        Sub(dz, dx, dz, cnt);        // frac = x - trunc(x) -> out0
    }
};

extern "C" __global__ __aicore__ void ascendc_modf_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR i1, uint64_t n, uint64_t perBlock) {
    (void)i1;
    CUSTOM_KERNEL_GUARD;
    UnaryKernelF32(z, x, myN, ModfF32Op());
}

// ---------------------------------------------------------------------------
// ldexp: (float x, int32 e) -> float    x * 2^e
// ---------------------------------------------------------------------------
struct LdexpF32Op {
    __aicore__ inline void operator()(LocalTensor<float>& dz, LocalTensor<float>& dx,
                                      LocalTensor<int32_t>& dy, uint32_t cnt, Scratch& scratch) {
        (void)scratch;
        Cast(dz, dy, RoundMode::CAST_NONE, cnt);   // float(e)
        Muls(dz, dz, LN2, cnt);                    // e * ln2
        Exp(dz, dz, cnt);                          // 2^e
        Mul(dz, dx, dz, cnt);                      // x * 2^e
    }
};

extern "C" __global__ __aicore__ void ascendc_ldexp_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR e, uint64_t n, uint64_t perBlock) {
    (void)o1;
    CUSTOM_KERNEL_GUARD;
    BinaryKernel<float, int32_t, float>(z, x, e, myN, LdexpF32Op());
}

// ---------------------------------------------------------------------------
// frexp: float -> (mantissa float, exponent int32)
//   e = floor(log2(|x|)) + 1; m = x / 2^e   (x == 0 -> best effort)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void ascendc_frexp_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR i1, uint64_t n, uint64_t perBlock) {
    (void)i1;
    uint64_t base = GetBlockIdx() * perBlock;
    uint64_t myN = perBlock;
    if (base + myN > n) { myN = (n > base) ? (n - base) : 0; }
    if (myN == 0) { return; }

    GlobalTensor<float> xGm, mGm;
    GlobalTensor<int32_t> eGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    mGm.SetGlobalBuffer((__gm__ float*)z);      // out0 = mantissa
    eGm.SetGlobalBuffer((__gm__ int32_t*)o1);   // out1 = exponent

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qx;
    TQue<TPosition::VECOUT, 2> qm, qe;
    Scratch scratch;
    pipe.InitBuffer(qx, 2, TILE * sizeof(float));
    pipe.InitBuffer(qm, 2, TILE * sizeof(float));
    pipe.InitBuffer(qe, 2, TILE * sizeof(int32_t));
    scratch.Init(pipe);

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        LocalTensor<float> lx = qx.AllocTensor<float>();
        DataCopy(lx, xGm[off], cnt);
        qx.EnQue(lx);

        LocalTensor<float> dx = qx.DeQue<float>();
        LocalTensor<float> ax = scratch.a.Get<float>(SCRATCH);
        LocalTensor<float> ef = scratch.b.Get<float>(SCRATCH);
        LocalTensor<float> pw = scratch.c.Get<float>(SCRATCH);
        Abs(ax, dx, cnt);
        Log2(ef, ax, cnt);
        Adds(ef, ef, 1.0f, cnt);                       // e (float)
        Exp(pw, ef, cnt);                              // 2^e
        LocalTensor<float> dm = qm.AllocTensor<float>();
        Div(dm, dx, pw, cnt);                          // mantissa
        qm.EnQue(dm);
        LocalTensor<int32_t> de = qe.AllocTensor<int32_t>();
        Cast(de, ef, RoundMode::CAST_RINT, cnt);
        qe.EnQue(de);
        qx.FreeTensor(dx);

        LocalTensor<float> om = qm.DeQue<float>();
        DataCopy(mGm[off], om, cnt);
        qm.FreeTensor(om);
        LocalTensor<int32_t> oe = qe.DeQue<int32_t>();
        DataCopy(eGm[off], oe, cnt);
        qe.FreeTensor(oe);
    }
}

// ---------------------------------------------------------------------------
// angle: complex64 -> float32
//   atan(imag / real) without quadrant correction (best-effort; see doc).
//   real/imag are extracted with strided DataCopy (DataCopyParams).
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void ascendc_angle_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR i1, uint64_t n, uint64_t perBlock) {
    (void)o1;
    (void)i1;
    uint64_t base = GetBlockIdx() * perBlock;
    uint64_t myN = perBlock;
    if (base + myN > n) { myN = (n > base) ? (n - base) : 0; }
    if (myN == 0) { return; }

    GlobalTensor<float> xGm, zGm;                  // x viewed as 2n floats
    xGm.SetGlobalBuffer((__gm__ float*)x);
    zGm.SetGlobalBuffer((__gm__ float*)z);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qr, qi;
    TQue<TPosition::VECOUT, 2> qz;
    pipe.InitBuffer(qr, 2, TILE * sizeof(float));
    pipe.InitBuffer(qi, 2, TILE * sizeof(float));
    pipe.InitBuffer(qz, 2, TILE * sizeof(float));

    DataCopyParams strided;
    strided.blockCount = TILE / 128;
    strided.blockLen = 128 * sizeof(float);   // 512B per block
    strided.srcStride = 128 * sizeof(float);  // skip the sibling half
    strided.dstStride = 0;

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        uint64_t foff = (base + off) * 2;      // float offset

        LocalTensor<float> lr = qr.AllocTensor<float>();
        LocalTensor<float> li = qi.AllocTensor<float>();
        DataCopy(lr, xGm[foff], strided);          // real parts
        DataCopy(li, xGm[foff + 1], strided);      // imag parts
        qr.EnQue(lr);
        qi.EnQue(li);

        LocalTensor<float> dr = qr.DeQue<float>();
        LocalTensor<float> di = qi.DeQue<float>();
        Div(di, di, dr, cnt);                      // imag / real
        Atan(di, di, cnt);                         // best-effort angle
        qz.EnQue(di);
        qr.FreeTensor(dr);

        LocalTensor<float> oz = qz.DeQue<float>();
        DataCopy(zGm[base + off], oz, cnt);
        qz.FreeTensor(oz);
    }
}

// ---------------------------------------------------------------------------
// imag: complex64 -> float32 (strided extraction of the odd floats)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void ascendc_imag_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR i1, uint64_t n, uint64_t perBlock) {
    (void)o1;
    (void)i1;
    uint64_t base = GetBlockIdx() * perBlock;
    uint64_t myN = perBlock;
    if (base + myN > n) { myN = (n > base) ? (n - base) : 0; }
    if (myN == 0) { return; }

    GlobalTensor<float> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    zGm.SetGlobalBuffer((__gm__ float*)z);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qi;
    TQue<TPosition::VECOUT, 2> qz;
    pipe.InitBuffer(qi, 2, TILE * sizeof(float));
    pipe.InitBuffer(qz, 2, TILE * sizeof(float));

    DataCopyParams strided;
    strided.blockCount = TILE / 128;
    strided.blockLen = 128 * sizeof(float);
    strided.srcStride = 128 * sizeof(float);
    strided.dstStride = 0;

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        uint64_t foff = (base + off) * 2 + 1;      // odd floats = imag

        LocalTensor<float> li = qi.AllocTensor<float>();
        DataCopy(li, xGm[foff], strided);
        qi.EnQue(li);

        LocalTensor<float> di = qi.DeQue<float>();
        LocalTensor<float> dz = qz.AllocTensor<float>();
        DataCopyParams contig;
        contig.blockCount = 1;
        contig.blockLen = cnt * sizeof(float);
        contig.srcStride = 0;
        contig.dstStride = 0;
        DataCopy(dz, di, contig);
        qz.EnQue(dz);
        qi.FreeTensor(di);

        LocalTensor<float> oz = qz.DeQue<float>();
        DataCopy(zGm[base + off], oz, cnt);
        qz.FreeTensor(oz);
    }
}

// ---------------------------------------------------------------------------
// conj: complex64 -> complex64
//   y = x * [1,-1,1,-1,...]; pattern built once per instance via
//   CreateVecIndex + Frac (parity in float domain, no integer ops needed).
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void ascendc_conj_f32(
        GM_ADDR z, GM_ADDR o1, GM_ADDR x, GM_ADDR i1, uint64_t n, uint64_t perBlock) {
    (void)o1;
    (void)i1;
    uint64_t base = GetBlockIdx() * perBlock;
    uint64_t myN = perBlock;
    if (base + myN > n) { myN = (n > base) ? (n - base) : 0; }
    if (myN == 0) { return; }

    GlobalTensor<float> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    zGm.SetGlobalBuffer((__gm__ float*)z);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qx;
    TQue<TPosition::VECOUT, 2> qz;
    TBuf<TPosition::VECCALC> patBuf, idxBuf;
    pipe.InitBuffer(qx, 2, 2 * TILE * sizeof(float));
    pipe.InitBuffer(qz, 2, 2 * TILE * sizeof(float));
    pipe.InitBuffer(patBuf, 2 * TILE * sizeof(float));
    pipe.InitBuffer(idxBuf, 2 * TILE * sizeof(float));

    // pattern = [1,-1,1,-1,...]: parity of index in the float domain
    LocalTensor<float> pat = patBuf.Get<float>(2 * TILE);
    LocalTensor<float> idx = idxBuf.Get<float>(2 * TILE);
    CreateVecIndex(idx, 0.0f, 2 * TILE);
    Muls(pat, idx, 0.5f, 2 * TILE);
    Frac(pat, pat, 2 * TILE);              // 0 (even) / 0.5 (odd)
    Muls(pat, pat, -2.0f, 2 * TILE);       // 0 / -1
    Adds(pat, pat, 1.0f, 2 * TILE);        // 1 / -1

    for (uint64_t off = 0; off < myN; off += TILE) {
        uint32_t cnt = tile_count(myN - off);
        uint64_t foff = (base + off) * 2;
        uint32_t fcnt = cnt * 2;

        LocalTensor<float> lx = qx.AllocTensor<float>();
        DataCopy(lx, xGm[foff], fcnt);
        qx.EnQue(lx);

        LocalTensor<float> dx = qx.DeQue<float>();
        LocalTensor<float> dz = qz.AllocTensor<float>();
        Mul(dz, dx, pat, fcnt);
        qz.EnQue(dz);
        qx.FreeTensor(dx);

        LocalTensor<float> oz = qz.DeQue<float>();
        DataCopy(zGm[foff], oz, fcnt);
        qz.FreeTensor(oz);
    }
}
