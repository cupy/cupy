import cupy as cp


def read_code(code_filename, params):
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


def benchmark(func, args, n_run):
    times = []
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times
