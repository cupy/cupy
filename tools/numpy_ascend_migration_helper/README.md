# numpy-ascend-checker

A static migration analyzer that checks NumPy/CuPy Python source against the
**numpy-ascend** (CuPy on Ascend NPU backend) capability rules, and optionally
rewrites the source for known-safe migrations.

> Architecture and rationale: see `numpy_ascend_migration_helper_design.md`
> and `numpy_ascend_migration_helper_prototype.md` in this folder.
> Rule seeds come from `numpy_cupy_pytorch_api_diff.md` and the Ascend
> backend limitation notes.

## Features

- **Two inputs**: NumPy (CPU) and CuPy source; both are resolved through a
  unified abstract namespace (aliases like `np` / `cp`, `from numpy import
  zeros`, assignment aliases such as `xp = np`).
- **Two modes** sharing one pipeline:
  - **dry-run analysis** (default): report only, never touches files.
  - **`--fix`**: source rewrite, restricted to `AUTO_SAFE` replacements.
- **AST + symbol resolution** (not just string matching): dotted APIs
  (`np.linalg.inv`), keywords (`dtype=...`), string dtypes (`"f8"`),
  `np.dtype(...)` calls, and dynamic dtypes.
- **7 diagnostic categories** with priorities (higher priority suppresses
  lower-priority noise on the same line):

  | Severity | Category |
  |---|---|
  | ERROR   | `UNSUPPORTED_API` |
  | ERROR   | `UNSUPPORTED_DTYPE` |
  | ERROR   | `UNSUPPORTED_ARGUMENT` |
  | ERROR   | `SEMANTIC_DIFFERENCE` |
  | WARNING | `CPU_FALLBACK` |
  | WARNING | `DISCOURAGED_DTYPE` |
  | INFO    | `UNKNOWN_DYNAMIC` (dtype not statically determinable) |

- **Data-driven rule DB** (JSON, versioned, separated from the analyzer):
  - `rules/api/rules.json` — API support / argument rules / semantic
    differences / replacements
  - `rules/dtype/ascend.json` — per-dtype capability matrix
  - `rules/messages/{en,zh}.json` — message catalog (`--lang en|zh`)
- **Safe auto-fix**: character-range rewrite on the original source
  (comments/formatting preserved, never `ast.unparse`), overlap conflict
  detection, optional `.bak` backup, diff preview.
- **Reports**: terminal text or machine-readable JSON (consumable by CI /
  IDE / dashboards), plus `--fail-on-error` for CI gating.

## Built-in rule coverage (v0.1 seeds)

| Area | Rule |
|---|---|
| dtype | `uint64` unsupported; `uint8/16/32` discouraged (int-cast overflow); `float64` / `complex128` discouraged (AICPU, slow); `object` / `datetime64` / `timedelta64` unsupported; `str_` CPU fallback |
| matmul / tensordot | only `float16` / `float32` / `bfloat16` (ASCEND TensorCore limitation) |
| power | `bool` operands unsupported (no implicit cast to int) |
| take / put | `axis=` (aclnnTake flatten semantics), `mode=wrap/clip` unsupported |
| pad | `mode=symmetric` etc. unsupported, `mode=wrap` semantic diff (circular, 2d/3d only) |
| sort | `kind=quicksort/heapsort` semantic diff (only stable-sort concept) |
| partition / argpartition / searchsorted | unsupported |
| IO (`load`, `save`, `loadtxt`, ...) | CPU fallback (host/device transfer + sync) |
| cupy.fusion | unsupported — use `triton_bridge` (`cupyx.jit`) |
| cupy.argmax / argmin | extra `dtype` positional argument vs NumPy (upstream incompatibility) |
| NumPy 1.x aliases (`product`, `cumproduct`, `sometrue`, `alltrue`) | unsupported with `AUTO_SAFE` rename to `prod` / `cumprod` / `any` / `all` |

## Usage

Run from this folder (no installation needed, Python >= 3.8, stdlib only):

```bash
cd tools/numpy_ascend_migration_helper
python -m numpy_ascend_checker <file-or-dir> [options]
```

### Analyze a single file (dry-run, default)

```bash
python -m numpy_ascend_checker examples/sample_input.py
```

Example output:

```text
examples/sample_input.py:8:5 ERROR   [UNSUPPORTED_DTYPE] dtype.uint64.unsupported
    | a = mkzeros((100, 100), dtype=np.uint64)
    dtype uint64 is not supported by the Ascend backend (uint64 tensors are not supported by the Ascend backend).
    Suggestion: Consider int64, uint32 if the value range / semantics permit.
    Auto-fix: NO (manual review required)

examples/sample_input.py:11:5 WARN    [DISCOURAGED_DTYPE] dtype.float64.discouraged
    | b = np.ones(1000, dtype=np.float64)
    dtype float64 is discouraged on Ascend (float64 runs on AICPU and is significantly slower (not accelerated)).
    Suggestion: Consider float32 if the value range / semantics permit.
    Auto-fix: YES (--fix --risky)

examples/sample_input.py:37:9 ERROR   [UNSUPPORTED_API] api.unsupported
    | total = np.product(g)
    API numpy.product is not supported by the Ascend backend (removed in NumPy 2.x).
    Suggestion: Use np.prod instead.
    Auto-fix: YES (--fix)

Summary:
  files scanned:  1
  APIs analyzed:  22
  errors:   12
  warnings: 4
  infos:    1
  fixable:  4 (auto_safe: 2, auto_warning: 2, manual: 0)
```

### Preview fixes (diff, no write)

```bash
python -m numpy_ascend_checker examples/sample_input.py --fix --dry-run --risky
```

```diff
-b = np.ones(1000, dtype=np.float64)
+b = np.ones(1000, dtype=np.float32)
-total = np.product(g)
+total = np.prod(g)
```

### Apply fixes

```bash
# only AUTO_SAFE renames (product -> prod, ...); --backup writes <file>.bak
python -m numpy_ascend_checker project/ --fix --backup

# also apply risky fixes (float64 -> float32: changes numerical semantics)
python -m numpy_ascend_checker project/ --fix --risky
```

Safety levels: `AUTO_SAFE` (applied by `--fix`), `AUTO_WARNING` (applied only
with `--risky`), `MANUAL` (never auto-applied, e.g. `uint64 -> int64`).

### JSON report / CI

```bash
python -m numpy_ascend_checker project/ \
    --format json --output report.json

python -m numpy_ascend_checker project/ --fail-on-error   # exit 1 on ERROR
```

### Options

| Option | Description |
|---|---|
| `paths` | one or more `.py` files or directories (scanned recursively) |
| `--fix` | apply `AUTO_SAFE` rewrites (default: analyze only) |
| `--dry-run` | with `--fix`: print diff without writing |
| `--risky` | with `--fix`: also apply `AUTO_WARNING` fixes |
| `--backup` | write `<file>.bak` before rewriting |
| `--format {text,json}` | report format (default `text`) |
| `--output PATH` | write the report to a file |
| `--lang {en,zh}` | message language (default `en`) |
| `--fail-on-error` | exit code 1 when any ERROR is reported |
| `--report-unknown-api` | also report APIs missing from the rule DB (INFO) |
| `--include GLOB` / `--exclude GLOB` / `--exclude-dir NAME` | file filtering (repeatable) |
| `--rules-dir PATH` | alternate rule DB directory |
| `--version` | print version |

Default excluded directories: `.git`, `.venv`, `venv`, `__pycache__`,
`build`, `dist`, `.codebuddy`, `.eggs`, `node_modules`.

## Extending the rule DB

Rules are data — no analyzer changes needed. Add an entry to
`rules/api/rules.json`:

```json
{
  "id": "ascend.my_api",
  "source": ["numpy", "cupy"],
  "operation": "my_api",
  "support": { "status": "unsupported", "reason": "not implemented" },
  "suggest": "Use ... instead.",
  "suggest_zh": "建议改用 ..."
}
```

Rule fields:

- `support.status`: `supported` | `unsupported` | `cpu_fallback` |
  `semantic_difference`
- `dtype_only`: list of allowed dtypes; any other dtype becomes
  `UNSUPPORTED_DTYPE`
- `dtype_rules`: per-dtype overrides, e.g. `"bool": {"status":
  "unsupported", ...}`
- `argument_rules`: per-argument checks (`unsupported_values`,
  `semantic_values`, `presence_unsupported`)
- `semantic`: extra semantic-difference notes attached to a supported API
- `replacement`: `{"kind": "rename"|"dtype"|"code", "target": ...,
  "fix_mode": "auto_safe"|"auto_warning"|"manual"}`

Per-dtype capability lives in `rules/dtype/ascend.json` (`status`,
`reason`, `replacements`, `fix_mode`, `suggestion` / `suggestion_zh`).
Message templates live in `rules/messages/{en,zh}.json` using
`{placeholder}` interpolation.

## Roadmap

- v0.2: runtime trace mode (`--runtime`) merging static + dynamic evidence
- v0.3: SARIF output for GitHub/GitLab code scanning
- v0.4: CANN-version-specific rule sets (`cann-8.x.json` / `cann-9.x.json`)
- v1.0: full migration assistant (IDE integration, capability dashboard)
