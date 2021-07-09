import os 
import sys


# Take cupy_backends/stub/cupy_cusparse.h and generate cupy_backends/hip/cupy_hipsparse.h,
# with all return values replaced by an error if not supprted. Except for functions, all
# structs, enums, typedefs, etc, are handled manually after cupy_hipsparse.h is generated.
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

with open('cupy_backends/stub/cupy_cusparse.h', 'r') as f:
    stubs = f.read()
    stubs = stubs.splitlines()

with open('/usr/local/cuda-11.3/include/cusparse.h', 'r') as f:
    cu_h = f.read()

with open('/opt/rocm-3.5.0/include/hipsparse.h', 'r') as f:
    hip_h = f.read()

# output HIP stub
hip_stub_h = []

for i, line in enumerate(stubs):
    if '...' in line:
        # ex: line = "cusparseStatus_t cusparseDestroyMatDescr(...) {"
        sig = line.split()
        assert len(sig) == 3

        # strip the prefix "cu" and the args "(...)", and assume HIP has the same function
        cu_func = sig[1]
        cu_func = cu_func[:cu_func.find('(')]
        hip_func = 'hip' + cu_func[2:]

        # find the full signature from cuSPARSE header
        cu_sig = cu_h.find(cu_func)
        # check if HIP has the corresponding function
        hip_sig = hip_h.find(hip_func)
        if cu_sig == -1:
            print(cu_func, "not found in cuSPARSE, maybe removed?", file=sys.stderr)
            can_map = False
            hip_stub_h.append(' '.join(sig))  # TODO: prettier print?
        elif hip_sig == -1:
            print(hip_func, "not found in hipSPARSE, maybe not supported?", file=sys.stderr)
            can_map = False
            hip_stub_h.append(' '.join(sig))  # TODO: prettier print?
        else:
            end_idx = cu_h[cu_sig:].find(')')
            assert end_idx != -1
            cu_sig = cu_h[cu_sig:cu_sig+end_idx+1]
            sig[1] = cu_sig
            can_map = True
            #print('\n'.join(sig))  # TODO: prettier print?
            hip_stub_h.append(' '.join(sig))  # TODO: prettier print?

        # now we have the full signature, map the return to HIP's function;
        # note that the "return" line is the very next line
        line = stubs[i+1]
        assert 'return' in line
        if can_map:
            # the function arguments either end with "," or ")"
            cu_sig = cu_sig.split()
            #print(cu_sig)
            hip_sig = '  return ' + hip_func + '('
            for s in cu_sig:
                # TODO: prettier print?
                if s[-1] in (',', ')'):
                    hip_sig += (s[:-1] + ', ')
            hip_sig = hip_sig[:-2] + ');'
            #print(hip_sig)
            hip_stub_h.append(hip_sig)
        else:
            hip_stub_h.append(line[:line.find('return')+6] + ' HIPSPARSE_STATUS_INTERNAL_ERROR;')
        
    elif 'return' in line:
        # don't do anything, as we handle the return when parsing "(...)"
        pass

    else:
        # just copy from the stub
        hip_stub_h.append(line)

with open('cupy_backends/hip/cupy_hipsparse.h', 'w') as f:
    f.write('\n'.join(hip_stub_h))
    f.write('\n')
