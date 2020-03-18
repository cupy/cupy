from os import path
import sys
import traceback

from cupy.cuda import memory_hook


class LineProfileHook(memory_hook.MemoryHook):
    """Code line CuPy memory profiler.

    This profiler shows line-by-line GPU memory consumption using traceback
    module. But, note that it can trace only CPython level, no Cython level.
    ref. https://github.com/cython/cython/issues/1755

    Example:
        Code example::

            from cupy.cuda import memory_hooks
            hook = memory_hooks.LineProfileHook()
            with hook:
                # some CuPy codes
            hook.print_report()

        Output example::

            _root (4.00KB, 4.00KB)
              lib/python3.6/unittest/__main__.py:18:<module> (4.00KB, 4.00KB)
                lib/python3.6/unittest/main.py:255:runTests (4.00KB, 4.00KB)
                  tests/cupy_tests/test.py:37:test (1.00KB, 1.00KB)
                  tests/cupy_tests/test.py:38:test (1.00KB, 1.00KB)
                  tests/cupy_tests/test.py:39:test (2.00KB, 2.00KB)

        Each line shows::

            {filename}:{lineno}:{func_name} ({used_bytes}, {acquired_bytes})

        where *used_bytes* is the memory bytes used from CuPy memory pool, and
        *acquired_bytes* is the actual memory bytes the CuPy memory pool
        acquired from GPU device.
        *_root* is a root node of the stack trace to show total memory usage.

    Args:
        max_depth (int): maximum depth to follow stack traces.
            Default is 0 (no limit).
    """

    name = 'LineProfileHook'

    def __init__(self, max_depth=0):
        self._memory_frames = {}
        self._root = MemoryFrame(None, None)
        self._filename = path.abspath(__file__)
        self._max_depth = max_depth

    # callback
    def malloc_preprocess(self, device_id, size, mem_size):
        self._cretate_frame_tree(used_bytes=mem_size)

    # callback
    def alloc_preprocess(self, device_id, mem_size):
        self._cretate_frame_tree(acquired_bytes=mem_size)

    def _cretate_frame_tree(self, used_bytes=0, acquired_bytes=0):
        self._root.used_bytes += used_bytes
        self._root.acquired_bytes += acquired_bytes
        parent = self._root
        for depth, stackframe in enumerate(self._extract_stackframes()):
            if 0 < self._max_depth <= depth + 1:
                break
            memory_frame = self._add_frame(parent, stackframe)
            memory_frame.used_bytes += used_bytes
            memory_frame.acquired_bytes += acquired_bytes
            parent = memory_frame

    def _extract_stackframes(self):
        stackframes = traceback.extract_stack()
        stackframes = [StackFrame(st) for st in stackframes]
        stackframes = [
            st for st in stackframes if st.filename != self._filename]
        return stackframes

    def _key_frame(self, parent, stackframe):
        return (parent,
                stackframe.filename,
                stackframe.lineno,
                stackframe.name)

    def _add_frame(self, parent, stackframe):
        key = self._key_frame(parent, stackframe)
        if key in self._memory_frames:
            memory_frame = self._memory_frames[key]
        else:
            memory_frame = MemoryFrame(parent, stackframe)
            self._memory_frames[key] = memory_frame
        return memory_frame

    def print_report(self, file=sys.stdout):
        """Prints a report of line memory profiling."""
        line = '_root (%s, %s)\n' % self._root.humanized_bytes()
        file.write(line)
        for child in self._root.children:
            self._print_frame(child, depth=1, file=file)
        file.flush()

    def _print_frame(self, memory_frame, depth=0, file=sys.stdout):
        indent = ' ' * (depth * 2)
        st = memory_frame.stackframe
        used_bytes, acquired_bytes = memory_frame.humanized_bytes()
        line = '%s%s:%s:%s (%s, %s)\n' % (
            indent, st.filename, st.lineno, st.name,
            used_bytes, acquired_bytes)
        file.write(line)
        for child in memory_frame.children:
            self._print_frame(child, depth=depth + 1, file=file)


class StackFrame(object):
    """Compatibility layer for outputs of traceback.extract_stack().

    Attributes:
        filename (string): filename
        lineno (int): line number
        name (string): function name
    """

    def __init__(self, obj):
        if isinstance(obj, tuple):  # < 3.5
            self.filename = obj[0]
            self.lineno = obj[1]
            self.name = obj[2]
        else:  # >= 3.5 FrameSummary
            self.filename = obj.filename
            self.lineno = obj.lineno
            self.name = obj.name


class MemoryFrame(object):
    """A single stack frame along with sum of memory usage at the frame.

    Attributes:
        stackframe (FrameSummary): stackframe from traceback.extract_stack().
        parent (MemoryFrame): parent frame, that is, caller.
        children (list of MemoryFrame): child frames, that is, callees.
        used_bytes (int): memory bytes that users used from CuPy memory pool.
        acquired_bytes (int): memory bytes that CuPy memory pool acquired
            from GPU device.
    """

    def __init__(self, parent, stackframe):
        self.stackframe = stackframe
        self.children = []
        self._set_parent(parent)
        self.used_bytes = 0
        self.acquired_bytes = 0

    def humanized_bytes(self):
        used_bytes = self._humanized_size(self.used_bytes)
        acquired_bytes = self._humanized_size(self.acquired_bytes)
        return (used_bytes, acquired_bytes)

    def _set_parent(self, parent):
        if parent and parent not in parent.children:
            self.parent = parent
            parent.children.append(self)

    def _humanized_size(self, size):
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
            if size < 1024.0:
                return '%3.2f%sB' % (size, unit)
            size /= 1024.0
        return '%.2f%sB' % (size, 'Z')
