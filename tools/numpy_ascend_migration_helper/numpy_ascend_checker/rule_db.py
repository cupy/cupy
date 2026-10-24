"""Rule DB loader.

Rules are data, not code (design doc §1). The DB is versioned and split by
concern so backend capability changes never require analyzer changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class RuleDB:
    def __init__(self, rules_dir):
        self.rules_dir = Path(rules_dir)
        self.schema_version = "1.0"
        self.target = "ascend"
        self.target_version = ""
        self.api_rules: Dict[str, List[dict]] = {}
        self.dtype_rules: Dict[str, dict] = {}
        self.messages: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        api_file = self.rules_dir / "api" / "rules.json"
        if api_file.exists():
            data = self._read(api_file)
            self.schema_version = data.get("schema_version", self.schema_version)
            self.target = data.get("target", self.target)
            self.target_version = data.get("target_version", "")
            for rule in data.get("rules", []):
                op = rule.get("operation")
                if op:
                    self.api_rules.setdefault(op, []).append(rule)

        dtype_file = self.rules_dir / "dtype" / "ascend.json"
        if dtype_file.exists():
            data = self._read(dtype_file)
            self.dtype_rules = data.get("dtypes", {})

        for lang in ("en", "zh"):
            mfile = self.rules_dir / "messages" / f"{lang}.json"
            if mfile.exists():
                self.messages[lang] = self._read(mfile)

    @staticmethod
    def _read(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    def lookup(self, operation: str, backend: str) -> Optional[dict]:
        """Lookup a rule for (operation, backend). Source-specific rules
        (e.g. cupy-only fusion) take precedence over shared rules."""
        candidates = self.api_rules.get(operation, [])
        specific = [r for r in candidates if backend in r.get("source", [])
                    and len(r.get("source", [])) == 1]
        shared = [r for r in candidates if backend in r.get("source", [])]
        return (specific or shared or [None])[0]

    def message(self, lang: str, message_id: str) -> Optional[dict]:
        return self.messages.get(lang, {}).get(message_id)
