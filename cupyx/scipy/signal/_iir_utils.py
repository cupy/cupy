
import cupy as cp

IIR_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cupy/math_constants.h>

__global__ void compute_correction_factors(
        const int m, const int k, const double* b, double* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= k) {
        return;
    }

    double* out_start = out + idx * (k + m);
    double* out_off = out_start + k;

    for(int i = 0; i < m; i++) {
        double acc = 0.0;
        for(int j = 0; j < k; j++) {
            acc += b[j] * out_off[i - j - 1];

        }
        out_off[i] = acc;
    }
}

__global__ void first_pass_iir(
        const int m, const int k, const int n, const double* factors,
        double* out, double* carries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = 2 * idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    double* group_start = out + m * group_num;
    double* group_carries = carries + k * group_num;

    int pos = group_pos;
    int up_bound = pos;
    int low_bound = pos;
    int rel_pos;

    for(int level = 1, iter = 1; level < m; level *=2, iter++) {
        int sz = min(pow(2.0, ((double) iter)), ((double) m));

        if(level > 1) {
            int factor = ceil(pos / ((double) sz));
            up_bound = sz * factor - 1;
            low_bound = up_bound - level + 1;
        }

        if(level == 1) {
            pos = low_bound;
        }

        if(pos < low_bound) {
            pos += level / 2;
        }

        if(pos + m * group_num >= n) {
            break;
        }

        rel_pos = pos % level;
        double carry = 0.0;
        for(int i = 1; i <= min(k, level); i++) {
            double k_value = group_start[low_bound - i];
            const double* k_factors = factors + (m + k) * (i - 1) + k;
            double factor = k_factors[rel_pos];
            carry += k_value * factor;
        }

        group_start[pos] += carry;
        __syncthreads();
    }

    if(pos >= m - k) {
        group_carries[pos - (m - k)] = group_start[pos];
    }

}

__global__ void copy_carries(
        const int m, const int k, const int n, const int n_group,
        const double* factors, double* carries, double* out,
        cudaStream_t prev_stream, cudaEvent_t prev_event) {

    if(prev_stream != NULL) {
        cudaStreamWaitEvent(prev_stream, prev_event, 0);
    }

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_start = n_group * m;

    if(group_start + m >= n) {
        return;
    }

    double* this_carries = carries + k * n_group;
    double* prev_carries = this_carries - k;

    double carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const double* k_factors = factors + (m + k) * (i - 1) + k;
        double factor = k_factors[m - k + idx];
        double k_value = prev_carries[k - i];
        carry += factor * k_value;
    }

    this_carries[idx] += carry;

}

__global__ void second_pass_iir(
        const int m, const int k, const int n, const int n_group,
        const double* factors, const double* carries, double* out,
        cudaStream_t carry_stream, cudaEvent_t carry_event) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_start = n_group * m;

    if(group_start + idx >= n) {
        return;
    }

    if(carry_stream != NULL) {
        cudaStreamWaitEvent(carry_stream, carry_event, 0);
    }

    const double* prev_carries = carries + (n_group - 1) * k;
    double* this_group = out + group_start;

    double carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const double* k_factors = factors + (m + k) * (i - 1) + k;
        double factor = k_factors[idx];
        double k_value = prev_carries[k - i];
        carry += factor * k_value;
    }

    this_group[idx] += carry;

}

"""

IIR_MODULE = cp.RawModule(
    code=IIR_KERNEL, options=('-std=c++11', '-dc'),
    name_expressions=['compute_correction_factors',
                      'first_pass_iir',
                      'copy_carries',
                      'second_pass_iir'])


def apply_iir(x, b):
    x = x.astype(cp.float64)
    b = b.astype(cp.float64)
    out = x.copy()

    k = b.size
    n = x.size
    block_sz = 32
    n_blocks = (n + block_sz - 1) // block_sz

    correction = cp.eye(k)
    correction = cp.c_[correction[::-1], cp.empty((k, block_sz))]
    carries = cp.empty((n_blocks - 1, k), dtype=cp.float64)

    corr_kernel = IIR_MODULE.get_function('compute_correction_factors')
    first_pass_kernel = IIR_MODULE.get_function('first_pass_iir')
    copy_kernel = IIR_MODULE.get_function('copy_carries')
    second_pass_kernel = IIR_MODULE.get_function('second_pass_iir')

    corr_kernel((k,), (1,), (block_sz, k, b, correction))
    first_pass_kernel((n_blocks,), (block_sz // 2,),
                      (block_sz, k, n, correction, out, carries))

    streams = [None]
    streams += [cp.cuda.Stream(non_blocking=True) for _ in range(n_blocks - 1)]
    events = [cp.cuda.Event() for _ in streams]

    exec_streams = [cp.cuda.Stream() for _ in range(n_blocks - 2)]

    default_stream = cp.cuda.get_current_stream()

    exec_streams += [default_stream]
    exec_events = [cp.cuda.Event(block=True) for _ in exec_streams]

    for i in range(n_blocks - 1):
        exec_stream = exec_streams[i]
        exec_event = exec_events[i]
        cur_stream = streams[i + 1]
        prev_stream = streams[i]
        cur_event = events[i + 1]
        prev_event = events[i]
        prev_stream_ptr = prev_stream.ptr if prev_stream is not None else 0
        prev_event_ptr = prev_event.ptr if prev_event is not None else 0
        with cur_stream:
            copy_kernel((1,), (k,),
                        (block_sz, k, n, i + 1, correction, carries,
                         out, prev_stream_ptr, prev_event_ptr))
            cur_event.record(cur_stream)

        with exec_stream:
            second_pass_kernel((1,), (block_sz,),
                               (block_sz, k, n, i + 1, correction, carries,
                                out, prev_stream_ptr, prev_event_ptr))
            exec_event.record(exec_stream)

    for evt in exec_events:
        default_stream.wait_event(evt)

    return out
