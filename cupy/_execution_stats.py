import atexit
# import os


class RuntimeStatistics:

    def __init__(self):
        self.compile_cache_hits = 0
        self.compile_cache_miss = 0

    def show_statistics(self):
        print('Runtime Statistics')
        print('==================')
        print(f'Compile Cache hits {self.compile_cache_hits}')
        print(f'Compile Cache misses {self.compile_cache_miss}')
        cc_hr = self.compile_cache_hits / (
            self.compile_cache_hits + self.compile_cache_miss
        )
        print(f'Compile Cache hit ratio {cc_hr:.2f}')


runtime_statistics = RuntimeStatistics()


# if os.environ.get('CUPY_SHOW_CACHE_STATISTICS', False):
atexit.register(runtime_statistics.show_statistics)
