import cupy as cp
from cupy_backends.cuda.libs import nvPTXCompiler


code = """
   .version 7.0                                           \n \
   .target sm_50                                          \n \
   .address_size 64                                       \n \
   .visible .entry simpleVectorAdd(                       \n \
        .param .u64 simpleVectorAdd_param_0,              \n \
        .param .u64 simpleVectorAdd_param_1,              \n \
        .param .u64 simpleVectorAdd_param_2               \n \
   ) {                                                    \n \
        .reg .f32   %f<4>;                                \n \
        .reg .b32   %r<5>;                                \n \
        .reg .b64   %rd<11>;                              \n \
        ld.param.u64    %rd1, [simpleVectorAdd_param_0];  \n \
        ld.param.u64    %rd2, [simpleVectorAdd_param_1];  \n \
        ld.param.u64    %rd3, [simpleVectorAdd_param_2];  \n \
        cvta.to.global.u64      %rd4, %rd3;               \n \
        cvta.to.global.u64      %rd5, %rd2;               \n \
        cvta.to.global.u64      %rd6, %rd1;               \n \
        mov.u32         %r1, %ctaid.x;                    \n \
        mov.u32         %r2, %ntid.x;                     \n \
        mov.u32         %r3, %tid.x;                      \n \
        mad.lo.s32      %r4, %r2, %r1, %r3;               \n \
        mul.wide.u32    %rd7, %r4, 4;                     \n \
        add.s64         %rd8, %rd6, %rd7;                 \n \
        ld.global.f32   %f1, [%rd8];                      \n \
        add.s64         %rd9, %rd5, %rd7;                 \n \
        ld.global.f32   %f2, [%rd9];                      \n \
        add.f32         %f3, %f1, %f2;                    \n \
        add.s64         %rd10, %rd4, %rd7;                \n \
        st.global.f32   [%rd10], %f3;                     \n \
        ret;                                              \n \
   } 
"""
print(nvPTXCompiler.getVersion())
handle = nvPTXCompiler.create(code)
nvPTXCompiler.compile(handle, ['--gpu-name=sm_86', '--verbose'])
cubin = nvPTXCompiler.getCompiledProgram(handle)
info = nvPTXCompiler.getInfoLog(handle)
error = nvPTXCompiler.getErrorLog(handle)
nvPTXCompiler.destroy(handle)
print(info)
assert error == ''

a = cp.arange(10, dtype=cp.float32)
b = a[::-1].copy()
out = cp.empty_like(a)

mod = cp.cuda.function.Module()
mod.load(cubin)
func = mod.get_function('simpleVectorAdd')
func((1,), (10,), (a, b, out))
print(out)
print(a+b)
