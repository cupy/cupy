"""Source indexing: char-offset computation and line access.

AST byte offsets (utf-8) are converted to char offsets so that the fixer can
do precise source-range replacement without ast.unparse (design doc §17).
"""

from __future__ import annotations

from typing import Tuple


class SourceIndex:
    def __init__(self, text: str, filename: str):
        self.text = text
        self.filename = filename
        self.lines = text.splitlines(keepends=True)
        self.line_starts = [0]
        for ln in self.lines:
            self.line_starts.append(self.line_starts[-1] + len(ln))

    def char_offset(self, lineno: int, col_byte: int) -> int:
        """Convert a 1-based lineno + utf-8 byte column to a char offset."""
        if lineno < 1:
            return 0
        if lineno > len(self.lines):
            return self.line_starts[-1] if self.line_starts else 0
        body = self.lines[lineno - 1].rstrip("\r\n")
        raw = body.encode("utf-8")
        col = max(0, min(col_byte, len(raw)))
        prefix = raw[:col].decode("utf-8", errors="replace")
        return self.line_starts[lineno - 1] + len(prefix)

    def node_range(self, node) -> Tuple[int, int]:
        start = self.char_offset(node.lineno, node.col_offset)
        end = self.char_offset(node.end_lineno, node.end_col_offset)
        return start, end

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].rstrip("\r\n")
        return ""
