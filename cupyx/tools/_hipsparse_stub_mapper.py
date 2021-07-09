import os 
import sys
import urllib.request


# Take cupy_backends/stub/cupy_cusparse.h and generate cupy_backends/hip/cupy_hipsparse.h,
# with all return values replaced by an error if not supprted. This script mainly focuses
# on getting the CUDA -> HIP API mapping done correctly; structs, enums, etc, are handled
# in a naive fasion.
#
# The stub functions, such as this,
#
# cusparseStatus_t cusparseDestroyMatDescr(...) {
#   return HIPSPARSE_STATUS_INTERNAL_ERROR;
# }
# 
# are mapped to their HIP counterparts, like this
# 
# cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
#   return hipsparseDestroyMatDescr(descrA);
# }

with open('/usr/local/cuda-11.3/include/cusparse.h', 'r') as f:
    cu_h = f.read()


# typedefs
typedefs = ('cusparseIndexBase_t', 'cusparseStatus_t', 'cusparseHandle_t',
            'cusparseMatDescr_t', 'csrsv2Info_t', 'csrsm2Info_t',
            'csric02Info_t', 'bsric02Info_t', 'csrilu02Info_t',
            'bsrilu02Info_t', 'csrgemm2Info_t',
            'cusparseMatrixType_t', 'cusparseFillMode_t', 'cusparseDiagType_t',
            'cusparsePointerMode_t', 'cusparseAction_t', 'cusparseDirection_t',
            'cusparseSolvePolicy_t', 'cusparseOperation_t')


# typedefs for generic API
typedefs += ('cusparseSpVecDescr_t', 'cusparseDnVecDescr_t', 'cusparseSpMatDescr_t',
             'cusparseDnMatDescr_t', 'cusparseIndexType_t', 'cusparseFormat_t',
             'cusparseOrder_t', 'cusparseSpMVAlg_t', 'cusparseSpMMAlg_t',
             'cusparseSparseToDenseAlg_t', 'cusparseDenseToSparseAlg_t',
             'cusparseCsr2CscAlg_t',)


processed_typedefs = set()


def get_idx_to_func(cu_h, cu_func):
    cu_sig = cu_h.find(cu_func)
    # 1. function names are always followed immediately by a "("
    # 2. we need a loop here to find the exact match
    while True:
        #print(cu_func, cu_sig, cu_h[cu_sig+len(cu_func)  ])
        if cu_sig == -1:
            break
        elif cu_h[cu_sig+len(cu_func)] != "(":
            cu_sig = cu_h.find(cu_func, cu_sig+1)
        else:
            break  # match
    return cu_sig


def get_hip_ver_num(hip_version):
    # "3.5.0" -> 305
    hip_version = hip_version.split('.')
    return int(hip_version[0]) * 100 + int(hip_version[1])


def main(hip_h, stubs, hip_version):
    hip_version = get_hip_ver_num(hip_version)

    # output HIP stub
    hip_stub_h = []

    for i, line in enumerate(stubs):
        if i == 3:
            hip_stub_h.append(line)
            if hip_version == 305:
                # insert the include after the include guard
                hip_stub_h.append('#include <hipsparse.h>')
                hip_stub_h.append('#include <hip/hip_version.h>    // for HIP_VERSION')
                hip_stub_h.append('#include <hip/library_types.h>  // for hipDataType')

                cudaDataType_converter = r"""
#if HIP_VERSION >= 402
static hipDataType convert_hipDatatype(cudaDataType type) {
    switch(static_cast<int>(type)) {
        case 2 /* CUDA_R_16F */: return HIP_R_16F;
        case 0 /* CUDA_R_32F */: return HIP_R_32F;
        case 1 /* CUDA_R_64F */: return HIP_R_64F;
        case 6 /* CUDA_C_16F */: return HIP_C_16F;
        case 4 /* CUDA_C_32F */: return HIP_C_32F;
        case 5 /* CUDA_C_64F */: return HIP_C_64F;
        default: throw std::runtime_error("unrecognized type");
    }
}
#endif
"""
                hip_stub_h.append(cudaDataType_converter)
    
        elif line.startswith('typedef'):
            old_line = ''
            typedef_found = False
            typedef_needed = True
            for t in typedefs:
                if t in line and t not in processed_typedefs:
                    hip_t = 'hip' + t[2:] if t.startswith('cu') else t
                    if hip_t in hip_h:
                        old_line = line
                        if t != hip_t:
                            old_line = line
                            line = 'typedef ' + hip_t + ' ' + t + ';'
                        else:
                            if hip_version == 305:
                                line = None
                            typedef_needed = False
                        typedef_found = True
                    else:
                        # new API not supported yet, use typedef from stub
                        pass
                    break
            else:
                t = None

            if line is not None:
                if typedef_found and hip_version > 305:
                    if typedef_needed:
                        hip_stub_h.append(f'#if HIP_VERSION >= {hip_version}')
                    else:
                        hip_stub_h.append(f'#if HIP_VERSION < {hip_version}')
                hip_stub_h.append(line)
                if typedef_found and hip_version > 305:
                    if typedef_needed:
                        hip_stub_h.append('#else')
                        hip_stub_h.append(old_line)
                    hip_stub_h.append('#endif\n')
            if t is not None and typedef_found:
                processed_typedefs.add(t)
                
        elif '...' in line:
            # ex: line = "cusparseStatus_t cusparseDestroyMatDescr(...) {"
            sig = line.split()
            try:
                assert len(sig) == 3
            except AssertionError:
                print(f"sig is {sig}")
                raise
    
            # strip the prefix "cu" and the args "(...)", and assume HIP has the same function
            cu_func = sig[1]
            cu_func = cu_func[:cu_func.find('(')]
            hip_func = 'hip' + cu_func[2:]
    
            # find the full signature from cuSPARSE header
            cu_sig = get_idx_to_func(cu_h, cu_func)
            # check if HIP has the corresponding function
            hip_sig = get_idx_to_func(hip_h, hip_func)
            if cu_sig == -1:
                print(cu_func, "not found in cuSPARSE, maybe removed?", file=sys.stderr)
                can_map = False
            elif hip_sig == -1:
                print(hip_func, "not found in hipSPARSE, maybe not supported?", file=sys.stderr)
                can_map = False
            else:
                end_idx = cu_h[cu_sig:].find(')')
                assert end_idx != -1
                cu_sig = cu_h[cu_sig:cu_sig+end_idx+1]
    
                # pretty print
                cu_sig = cu_sig.split('\n')
                new_cu_sig = cu_sig[0] + '\n'
                for s in cu_sig[1:]:
                    new_cu_sig += (' ' * (len(sig[0]) + 1)) + s + '\n'
                cu_sig = new_cu_sig[:-1]
    
                sig[1] = cu_sig
                can_map = True
            hip_stub_h.append(' '.join(sig))
    
            # now we have the full signature, map the return to HIP's function;
            # note that the "return" line is in the next two lines
            line = stubs[i+1]
            if 'return' not in line:
                line = stubs[i+2]
                assert 'return' in line
            if can_map:
                cu_sig = cu_sig.split('\n')
                cu_sig_processed = []

                # each line ends with an argument, which is followed by a "," or ")"
                # (exceptions: cusparseCbsric02, cusparseCooGet, ...)
                skip_line = None
                for l, s in enumerate(cu_sig):
                    if l != skip_line:
                        if s.endswith(',') or s.endswith(')'):
                            cu_sig_processed.append(s)
                        else:
                            break_idx = s.find(',')
                            if break_idx == -1:
                                break_idx = s.find(')')
                            if break_idx == -1:
                                cu_sig_processed.append(s + cu_sig[l+1])
                                skip_line = l+1
                            else:
                                # argument could be followed by an inline comment
                                cu_sig_processed.append(s[:break_idx+1])
                cu_sig = cu_sig_processed

                if hip_version != 305:
                    hip_stub_h.append(f"#if HIP_VERSION >= {hip_version}")
                hip_sig = '  return ' + hip_func + '('
                decl = ''
                for s in cu_sig:
                    # TODO: prettier print? note that we currently rely on hip_sig being a one-liner...
                    # TODO: I am being silly here; this can probably handled gracefully using regex...
                    if 'const cuComplex*' in s:
                        s = s.split()
                        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
                        cast = 'reinterpret_cast<const hipComplex*>'
                    elif 'const cuDoubleComplex*' in s:
                        s = s.split()
                        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
                        cast = 'reinterpret_cast<const hipDoubleComplex*>'
                    elif 'cuComplex*' in s:
                        s = s.split()
                        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
                        cast = 'reinterpret_cast<hipComplex*>'
                    elif 'cuDoubleComplex*' in s:
                        s = s.split()
                        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
                        cast = 'reinterpret_cast<hipDoubleComplex*>'
                    elif 'cuComplex' in s:
                        s = s.split()
                        decl = '  // This is needed to be safe with -Wstrict-aliasing.\n'
                        decl += f'  hipComplex blah;\n  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
                        arg = 'blah' + s[-1][-1]
                        cast = ''
                    elif 'cuDoubleComplex' in s:
                        s = s.split()
                        decl = '  // This is needed to be safe with -Wstrict-aliasing.\n'
                        decl += f'  hipDoubleComplex blah;\n  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
                        arg = 'blah' + s[-1][-1]
                        cast = ''
                    elif 'cudaDataType*' in s:
                        s = s.split()
                        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
                        cast = 'reinterpret_cast<hipDataType*>'
                    elif 'cudaDataType' in s:
                        s = s.split()
                        decl = '  // This is needed to be safe with -Wstrict-aliasing.\n'
                        decl += f'  hipDataType blah = convert_hipDatatype(' + s[-1][:-1] + ');\n'
                        arg = 'blah' + s[-1][-1]
                        cast = ''
                    else:
                        s = s.split()
                        arg = s[-1]
                        cast = ''
                    hip_sig += (cast + arg + ' ')
                hip_sig = hip_sig[:-1] + ';'
                hip_stub_h.append(decl+hip_sig)
                if hip_version != 305:
                    hip_stub_h.append("#else")
                    hip_stub_h.append('  return HIPSPARSE_STATUS_INTERNAL_ERROR;')
                    hip_stub_h.append("#endif")
            else:
                hip_stub_h.append(line[:line.find('return')+6] + ' HIPSPARSE_STATUS_INTERNAL_ERROR;')
            
        elif 'return' in line:
            if 'CUSPARSE_STATUS_SUCCESS' in line:
                # don't do anything, as we handle the return when parsing "(...)"
                pass
            elif 'HIPSPARSE_STATUS_INTERNAL_ERROR' in line:
                if '#else' in stubs[i-1]:
                    # just copy from the stub
                    hip_stub_h.append(line)
            else:
                # just copy from the stub
                hip_stub_h.append(line)
    
        else:
            # just copy from the stub
            hip_stub_h.append(line)
    
    with open('cupy_backends/hip/cupy_hipsparse.h', 'w') as f:
        f.write('\n'.join(hip_stub_h))
        f.write('\n')


if __name__ == '__main__':
    hipsparse_url = "https://raw.githubusercontent.com/ROCmSoftwarePlatform/hipSPARSE/rocm-{0}/library/include/hipsparse.h"
    versions = ("3.5.0", "3.7.0", "3.8.0", "3.9.0", "4.0.0", "4.2.0")
    #versions = ("3.5.0", "3.7.0")

    with open('cupy_backends/stub/cupy_cusparse.h', 'r') as f:
        stubs = f.read().splitlines()

    for i, ver in enumerate(versions):
        req = urllib.request.urlopen(hipsparse_url.format(ver))
        with req as f:
            hip_h = f.read().decode()

        main(hip_h, stubs, ver)

        with open('cupy_backends/hip/cupy_hipsparse.h', 'r') as f:
            stubs = f.read().splitlines()
