import urllib.request
import sys


# Take cupy_backends/stub/cupy_cusparse.h and generate
# cupy_backends/hip/cupy_hipsparse.h, with all return values replaced by an
# error if not supprted. This script mainly focuses on getting the CUDA ->
# HIP API mapping done correctly; structs, enums, etc, are handled
# automatically to the maximal extent.
#
# The stub functions, such as this,
#
# cusparseStatus_t cusparseDestroyMatDescr(...) {
#   return HIPSPARSE_STATUS_NOT_SUPPORTED;
# }
#
# are mapped to their HIP counterparts, like this
#
# cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
#   return hipsparseDestroyMatDescr(descrA);
# }


# some cuSPARSE APIs are removed in recent CUDA 11.x, so we also need to
# look up older CUDA to fetch the API signatures
cusparse_h = '/usr/local/cuda-{0}/include/cusparse.h'
cu_versions = ('11.3', '11.0', '10.2')


hipsparse_url = ('https://raw.githubusercontent.com/ROCmSoftwarePlatform/'
                 'hipSPARSE/rocm-{0}/library/include/hipsparse.h')
hip_versions = ("3.5.0", "3.7.0", "3.8.0", "3.9.0", "4.0.0", "4.2.0")


# typedefs
typedefs = ('cusparseIndexBase_t', 'cusparseStatus_t', 'cusparseHandle_t',
            'cusparseMatDescr_t', 'csrsv2Info_t', 'csrsm2Info_t',
            'csric02Info_t', 'bsric02Info_t', 'csrilu02Info_t',
            'bsrilu02Info_t', 'csrgemm2Info_t',
            'cusparseMatrixType_t', 'cusparseFillMode_t', 'cusparseDiagType_t',
            'cusparsePointerMode_t', 'cusparseAction_t', 'cusparseDirection_t',
            'cusparseSolvePolicy_t', 'cusparseOperation_t')


# typedefs for generic API
typedefs += ('cusparseSpVecDescr_t', 'cusparseDnVecDescr_t',
             'cusparseSpMatDescr_t', 'cusparseDnMatDescr_t',
             'cusparseIndexType_t', 'cusparseFormat_t', 'cusparseOrder_t',
             'cusparseSpMVAlg_t', 'cusparseSpMMAlg_t',
             'cusparseSparseToDenseAlg_t', 'cusparseDenseToSparseAlg_t',
             'cusparseCsr2CscAlg_t',)


# helpers
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

cusparseOrder_converter = r"""
#if HIP_VERSION >= 402
typedef enum {} cusparseOrder_t;
static hipsparseOrder_t convert_hipsparseOrder_t(cusparseOrder_t type) {
    switch(static_cast<int>(type)) {
        case 1 /* CUSPARSE_ORDER_COL */: return HIPSPARSE_ORDER_COLUMN;
        case 2 /* CUSPARSE_ORDER_ROW */: return HIPSPARSE_ORDER_ROW;
        default: throw std::runtime_error("unrecognized type");
    }
}
"""

default_return_code = r"""
#if HIP_VERSION < 401
#define HIPSPARSE_STATUS_NOT_SUPPORTED (hipsparseStatus_t)10
#endif
"""


# keep track of typedefs that are already handled (as we move from older
# to newer HIP version)
processed_typedefs = set()


def get_idx_to_func(cu_h, cu_func):
    cu_sig = cu_h.find(cu_func)
    # 1. function names are always followed immediately by a "("
    # 2. we need a loop here to find the exact match
    while True:
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


def merge_bad_broken_lines(cu_sig):
    # each line ends with an argument, which is followed by a "," or ")"
    # except for cusparseCbsric02, cusparseCooGet, ...
    cu_sig_processed = []
    skip_line = None
    for line, s in enumerate(cu_sig):
        if line != skip_line:
            if s.endswith(',') or s.endswith(')'):
                cu_sig_processed.append(s)
            else:
                break_idx = s.find(',')
                if break_idx == -1:
                    break_idx = s.find(')')
                if break_idx == -1:
                    cu_sig_processed.append(s + cu_sig[line+1])
                    skip_line = line+1
                else:
                    # argument could be followed by an inline comment
                    cu_sig_processed.append(s[:break_idx+1])
    return cu_sig_processed


def process_func_args(s, hip_sig, decl, hip_func):
    # TODO(leofang): prettier print? note that we currently rely on "hip_sig"
    # being a one-liner...
    # TODO(leofang): I am being silly here; these can probably be handled more
    # elegantly using regex...
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
        decl += '  hipComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cuDoubleComplex' in s:
        s = s.split()
        decl += '  hipDoubleComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cudaDataType*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipDataType*>'
    elif 'cudaDataType' in s:
        s = s.split()
        decl += '  hipDataType blah = convert_hipDatatype('
        decl += s[-1][:-1] + ');\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t*' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(*' + s[-1][:-1] + ');\n'
        arg = '&blah2' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(' + s[-1][:-1] + ');\n'
        arg = 'blah2' + s[-1][-1]
        cast = ''
    elif ('const void*' in s
            and hip_func == 'hipsparseSpVV_bufferSize'):
        # work around HIP's bad typing...
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'const_cast<void*>'
    else:
        s = s.split()
        arg = s[-1]
        cast = ''
    hip_sig += (cast + arg + ' ')
    return hip_sig, decl


def main(hip_h, cu_h, stubs, hip_version, init):
    hip_version = get_hip_ver_num(hip_version)

    # output HIP stub
    hip_stub_h = []

    for i, line in enumerate(stubs):
        if i == 3 and not init:
            hip_stub_h.append(line)
            if hip_version == 305:
                # insert the include after the include guard
                hip_stub_h.append('#include <hipsparse.h>')
                hip_stub_h.append(
                    '#include <hip/hip_version.h>    // for HIP_VERSION')
                hip_stub_h.append(
                    '#include <hip/library_types.h>  // for hipDataType')
                hip_stub_h.append(cudaDataType_converter)
                hip_stub_h.append(default_return_code)

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
                # hack...
                if t == 'cusparseOrder_t' and hip_version == 402:
                    hip_stub_h.append(cusparseOrder_converter)

                elif typedef_found and hip_version > 305:
                    if typedef_needed:
                        hip_stub_h.append(f'#if HIP_VERSION >= {hip_version}')
                    else:
                        hip_stub_h.append(f'#if HIP_VERSION < {hip_version}')

                # hack...
                if not (t == 'cusparseOrder_t' and hip_version == 402):
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

            # strip the prefix "cu" and the args "(...)", and assume HIP has
            # the same function
            cu_func = sig[1]
            cu_func = cu_func[:cu_func.find('(')]
            hip_func = 'hip' + cu_func[2:]

            # find the full signature from cuSPARSE header
            cu_sig = get_idx_to_func(cu_h, cu_func)
            # check if HIP has the corresponding function
            hip_sig = get_idx_to_func(hip_h, hip_func)
            if cu_sig == -1 and hip_sig == -1:
                assert False
            elif cu_sig == -1 and hip_sig != -1:
                print(cu_func, "not found in cuSPARSE, maybe removed?",
                      file=sys.stderr)
                can_map = False
            elif cu_sig != -1 and hip_sig == -1:
                print(hip_func, "not found in hipSPARSE, maybe not supported?",
                      file=sys.stderr)
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
                cu_sig = merge_bad_broken_lines(cu_sig)

                if hip_version != 305:
                    hip_stub_h.append(f"#if HIP_VERSION >= {hip_version}")
                hip_sig = '  return ' + hip_func + '('
                decl = ''
                for s in cu_sig:
                    hip_sig, decl = process_func_args(
                        s, hip_sig, decl, hip_func)
                hip_sig = hip_sig[:-1] + ';'
                hip_stub_h.append(decl+hip_sig)
                if hip_version != 305:
                    hip_stub_h.append("#else")
                    hip_stub_h.append(
                        '  return HIPSPARSE_STATUS_NOT_SUPPORTED;')
                    hip_stub_h.append("#endif")
            else:
                hip_stub_h.append(
                    (line[:line.find('return')+6]
                     + ' HIPSPARSE_STATUS_NOT_SUPPORTED;'))

        elif 'return' in line:
            if 'CUSPARSE_STATUS' in line:
                # don't do anything, as we handle the return when
                # parsing "(...)"
                pass
            elif 'HIPSPARSE_STATUS_NOT_SUPPORTED' in line:
                if '#else' in stubs[i-1]:
                    # just copy from the stub
                    hip_stub_h.append(line)
            else:
                # just copy from the stub
                hip_stub_h.append(line)

        else:
            # just copy from the stub
            hip_stub_h.append(line)

    return ('\n'.join(hip_stub_h)) + '\n'


if __name__ == '__main__':
    with open('cupy_backends/stub/cupy_cusparse.h', 'r') as f:
        stubs = f.read()

    init = False
    for cu_ver in cu_versions:
        with open(cusparse_h.format(cu_ver), 'r') as f:
            cu_h = f.read()

        x = 0
        for hip_ver in hip_versions:
            stubs = stubs.splitlines()

            req = urllib.request.urlopen(hipsparse_url.format(hip_ver))
            with req as f:
                hip_h = f.read().decode()

            stubs = main(hip_h, cu_h, stubs, hip_ver, init)
            init = True

    # more hacks...
    stubs = stubs.replace(
        '#define CUSPARSE_VERSION -1',
        ('#define CUSPARSE_VERSION '
         '(hipsparseVersionMajor*100000+hipsparseVersionMinor*100'
         '+hipsparseVersionPatch)'))
    stubs = stubs.replace(
        'INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H',
        'INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H')
    stubs = stubs[stubs.find('\n'):]

    with open('cupy_backends/hip/cupy_hipsparse.h', 'w') as f:
        f.write(stubs)
