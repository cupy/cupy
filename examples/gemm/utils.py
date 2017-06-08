import cupy as cp


@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, options=()):
    assert isinstance(options, tuple)
    kernel_code = cp.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def read_code(code_filename, params):
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = "#define " + k + " " + str(v) + "\n" + code
    return code


def bencmark(func, args, n_run):
    times = []
    
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        out = func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times
