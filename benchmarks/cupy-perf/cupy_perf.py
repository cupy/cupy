import argparse
import inspect
import sys
import time

import cupy
import numpy


_prof = None
_line_prof = None


def _init_profiler():
    global _prof
    import cProfile
    if _prof is None:
        _prof = cProfile.Profile()


def _init_line_profiler():
    global _line_prof
    import line_profiler
    if _line_prof is None:
        _line_prof = line_profiler.LineProfiler()


def get_profiler():
    _init_profiler()
    return _prof


def get_line_profiler():
    _init_line_profiler()
    return _line_prof


def _parse_options(cmd_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-gpu', action='store_true')
    args = parser.parse_args(cmd_args)
    return args


class PerfCase:
    def __init__(self, func):
        self.func = func
        self.n = 10000
        self.n_warmup = 10
        self.exclude_others = False
        self.skip = False


def attr(**kwargs):
    def decorator(case):
        if isinstance(case, PerfCase):
            case_ = case
        else:
            case_ = PerfCase(case)

        for key, val in kwargs.items():
            setattr(case_, key, val)
        return case_
    return decorator


class PerfCaseResult(object):
    def __init__(self, name, ts):
        self.name = name
        self.ts = ts

    def cpu_min(self):
        return self.ts[0].min()

    def cpu_mean(self):
        return self.ts[0].mean()

    def cpu_std(self):
        return self.ts[0].std()

    def gpu_min(self):
        return self.ts[1].min()

    def gpu_mean(self):
        return self.ts[1].mean()

    def gpu_std(self):
        return self.ts[1].std()

    def to_str(self, show_gpu=False):
        s = '{:<20s}: {:9.03f} us   +/-{:6.03f} (min:{:9.03f}) us'.format(
            self.name,
            self.cpu_mean() * 1e6,
            self.cpu_std() * 1e6,
            self.cpu_min() * 1e6)
        if show_gpu:
            s += '  {:9.03f} us   +/-{:6.03f} (min:{:9.03f}) us'.format(
                self.gpu_mean() * 1e6,
                self.gpu_std() * 1e6,
                self.gpu_min() * 1e6)
        return s

    def __str__(self):
        return self.to_str(show_gpu=True)


class PerfCases(object):

    enable_profiler = False
    enable_line_profiler = False

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def get_cases(self):
        prefix = 'perf_'
        cases = []
        has_exclude_others = False

        for name in dir(self):
            if name.startswith(prefix):
                obj = getattr(self, name)
                if isinstance(obj, PerfCase):
                    case = obj
                    func = obj.func
                elif callable(obj):
                    case = PerfCase(obj.__func__)
                    func = obj
                else:
                    continue

                if case.skip:
                    continue

                # If this case has `exclude_others` flag, clear previous cases
                # with no `exclude_others` flag. Then after this point, only
                # cases with the flag will be collected.
                if case.exclude_others:
                    if not has_exclude_others:
                        cases = []
                        has_exclude_others = True
                elif has_exclude_others:
                    continue

                name = name[len(prefix):]
                _, linum = inspect.getsourcelines(func)
                cases.append((linum, name, case))

        cases = sorted(cases)
        for linum, name, f in cases:
            yield name, f

    def run(self):
        args = _parse_options(sys.argv[1:])
        if self.enable_profiler:
            _init_profiler()
        if self.enable_line_profiler:
            _init_line_profiler()

        cases = list(self.get_cases())
        for case_name, case in cases:
            self.setUp()

            if isinstance(case, PerfCase):
                pass
            else:
                case = PerfCase(case)
            result = self._run_perf(case_name, case)
            self.tearDown()
            print(result.to_str(show_gpu=args.show_gpu))

    def _run_perf(self, name, case):
        func = case.func
        n = case.n
        n_warmup = case.n_warmup

        ts = numpy.empty((2, n,), dtype=numpy.float64)
        ev1 = cupy.cuda.stream.Event()
        ev2 = cupy.cuda.stream.Event()

        for i in range(n_warmup):
            func(self)

        if self.enable_line_profiler:
            _line_prof.enable()
        if self.enable_profiler:
            _prof.enable()

        for i in range(n):
            ev1.synchronize()
            ev1.record()
            t1 = time.perf_counter()

            func(self)

            t2 = time.perf_counter()
            ev2.record()
            ev2.synchronize()
            cpu_time = t2 - t1
            gpu_time = cupy.cuda.get_elapsed_time(ev1, ev2) * 1e-3
            ts[0, i] = cpu_time
            ts[1, i] = gpu_time

        if self.enable_profiler:
            _prof.disable()
        if self.enable_line_profiler:
            _line_prof.disable()

        return PerfCaseResult(name, ts)


def run(module_name):
    print(cupy)
    mod = sys.modules[module_name]
    classes = []
    for name, cls in inspect.getmembers(mod):
        if (not name.startswith('_')) and inspect.isclass(cls) and issubclass(cls, PerfCases):
            _, linum = inspect.getsourcelines(cls)
            classes.append((linum, cls))

    classes = sorted(classes)
    for linum, cls in classes:
        cases = cls()
        cases.run()
