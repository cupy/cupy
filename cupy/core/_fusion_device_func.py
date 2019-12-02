import string

from cupy.core import _fusion_emit_code


_dtype_to_ctype = _fusion_emit_code._dtype_to_ctype


class _SubmoduleBase(object):
    "A Base class of device function"

    def _parse_op_info(self, func, op_info):
        raise NotImplementedError

    def __init__(self, name, func, op_info, in_params, out_params):
        assert isinstance(name, str)
        assert callable(func)
        assert isinstance(in_params, list)
        assert isinstance(out_params, list)

        self.name = name
        self.in_params = in_params
        self.out_params = out_params
        self.preamble = func._preamble
        self._parse_op_info(func, op_info)

    def emit_code(self):
        """Returns a CUDA device function code.
        """
        raise NotImplementedError

    def emit_call_code(self):
        """Returns a CUDA code block which calls the device function.
        """
        raise NotImplementedError


class _SubmoduleUfunc(_SubmoduleBase):
    """A device function for elementwise operations.
    """

    def _parse_op_info(self, ufunc, op_info):
        self.op_expr, self.dtypes = op_info

    def emit_code(self):
        """Returns a CUDA device function code.

        Returns a string like as:
        ```
        __device__ void cupy_add_0(int &in0_, float &in1_, double &out0_) {
            typedef double in0_type;
            typedef double in1_type;
            typedef double out0_type;
            double in0 = (double) in0_;
            double in1 = (double) in1_;
            double out0 = (double) out0_;
            out0 = in0 + in1;
            out0_ = out0;
        }
        ```
        """
        nin = len(self.in_params)
        assert len(self.in_params) == len(self.dtypes[:nin])
        in_params = [
            (_dtype_to_ctype[p.dtype], _dtype_to_ctype[t], 'in{}'.format(i))
            for i, (p, t) in enumerate(zip(self.in_params, self.dtypes[:nin]))
        ]
        out_params = [
            (_dtype_to_ctype[p.dtype], _dtype_to_ctype[t], 'out{}'.format(i))
            for i, (p, t) in enumerate(zip(self.out_params, self.dtypes[nin:]))
        ]
        params = in_params + out_params

        params_code = ', '.join(['{} &{}_'.format(t, s) for t, _, s in params])
        typedef = ['typedef {} {}_type;'.format(t, s) for _, t, s in params]
        read = ['{} {} = ({}) {}_;'.format(t, s, t, s) for _, t, s in params]
        write = ['{}_ = {};'.format(s, s, s) for _, _, s in out_params]

        return _fusion_emit_code._CodeBlock(
            '__device__ void ${name}(${params}) {', [
                *typedef,
                *read,
                '${operation};',
                *write],
            '}',
            name=self.name,
            params=params_code,
            operation=self.op_expr)

    def emit_call_code(self):
        params = self.in_params + self.out_params
        return '{op_name}({params});'.format(
            op_name=self.name,
            params=', '.join([var.lvar_name for var in params]))


class _SubmoduleReduction(_SubmoduleBase):
    """A device function for reduction operations.
    """

    def _parse_op_info(self, reduction_func, op_info):
        if reduction_func.identity is None:
            self.identity = ''
        else:
            self.identity = str(reduction_func.identity)

        _, self.expr, self.postmap_cast_code, self.reduce_ctype = op_info
        if self.reduce_ctype is None:
            out_param, = self.out_params
            self.reduce_ctype = _dtype_to_ctype[out_param.dtype]

    def emit_code(self):
        """Returns a CUDA device function code.

        The emitted code assumes that ``block_stride`` and `blockDim.x` is a
        power of 2.
        """

        in_param, = self.in_params
        out_param, = self.out_params
        op_name = '{}_op'.format(self.name)
        postmap_name = '{}_postmap'.format(self.name)

        return string.Template('''
#define ${op_name}(a, b) (${reduce_expr})
#define ${postmap_name}(a, out0) (${postmap_cast})

template <typename InType, typename OutType, typename InIndexerType, typename OutIndexerType>
__device__ void ${name}(
        InType in_arr, OutType out_arr,
        InIndexerType in_ind, OutIndexerType out_ind, int block_stride) {
    typedef ${in_type} type_in0_raw;
    typedef ${out_type} type_out0_raw;
    typedef ${reduce_ctype} _type_reduce;
    extern __shared__ char _sdata_raw[];
    _type_reduce *sdata = reinterpret_cast<_type_reduce*>(_sdata_raw);
    unsigned int tid = threadIdx.x;
    int _J = tid >> __popc(block_stride - 1);
    ptrdiff_t _j = (ptrdiff_t)_J * out_ind.size();
    int J_stride = blockDim.x >> __popc(block_stride - 1);
    ptrdiff_t j_stride = (ptrdiff_t)J_stride * out_ind.size();

    for (ptrdiff_t _i = (ptrdiff_t)blockIdx.x * block_stride; _i < out_ind.size(); _i += (ptrdiff_t)gridDim.x * block_stride) {
        _type_reduce s = _type_reduce(${identity});
        ptrdiff_t i = _i + (tid & (block_stride - 1));
        for (ptrdiff_t j = i + _j; j < in_ind.size(); j += j_stride) {
            in_ind.set(j);
            s = ${op_name}(s, static_cast<_type_reduce>(in_arr[in_ind.get()]));
        }
        sdata[tid] = s;
        __syncthreads();
        for (unsigned int block = blockDim.x / 2; block >= block_stride; block >>= 1) {
            if (tid < block) {
                sdata[tid] = ${op_name}(sdata[tid], sdata[tid + block]);
            }
            __syncthreads();
        }
        if (tid < block_stride) {
            s = sdata[tid];
        }
        if (tid < block_stride && i < out_ind.size()) {
            out_ind.set(i);
            ${postmap_name}(s, out_arr[out_ind.get()]);
        }
        __syncthreads();
    }
}'''  # NOQA
        ).substitute(
            name=self.name,
            op_name=op_name,
            postmap_name=postmap_name,
            in_type=_dtype_to_ctype[in_param.dtype],
            out_type=_dtype_to_ctype[out_param.dtype],
            reduce_ctype=self.reduce_ctype,
            reduce_expr=self.expr,
            identity=self.identity,
            postmap_cast=self.postmap_cast_code)
