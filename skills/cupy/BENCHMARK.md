# Skill Benchmark: cupy

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cupy`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 17 evaluation tasks (15 positive, 2 negative)
- Dataset digest: `sha256:7a15809daadc42b86b4e942e23f0cf3d5802689a18e50b9f7a74e4329fd83747` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cupy | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cupy | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cupy | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cupy | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cupy | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cupy`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 72.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 70.6% → 64.7% (-5.9 points) |
| Correctness | 80.0% → 90.6% (+10.6 points) | 87.1% → 90.6% (+3.5 points) |
| Discoverability | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.5% → 63.0% (+7.5 points) | 61.6% → 59.0% (-2.6 points) |
| Efficiency | 87.5% — baseline ran, but no comparable score was available; uplift unavailable | 64.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 94.5% → 93.3% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.6% → 82.5% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,619,796 | 3,520,866 | +1,098,930 | +31.21% | skill 17/17; base 17/17 |
| claude-code | cupy-async-timing | 77,881 | 36,303 | +41,578 | +114.53% | skill 1/1; base 1/1 |
| claude-code | cupy-duplicate-index-assign | 347,711 | 341,867 | +5,844 | +1.71% | skill 1/1; base 1/1 |
| claude-code | cupy-float64-consumer-gpu | 78,393 | 36,906 | +41,487 | +112.41% | skill 1/1; base 1/1 |
| claude-code | cupy-float64-promotion-slowdown | 296,659 | 372,737 | -76,078 | -20.41% | skill 1/1; base 1/1 |
| claude-code | cupy-griddata-port | 385,549 | 447,884 | -62,335 | -13.92% | skill 1/1; base 1/1 |
| claude-code | cupy-jit-block-reduction | 1,119,969 | 435,789 | +684,180 | +157.00% | skill 1/1; base 1/1 |
| claude-code | cupy-matrix-exponential | 334,307 | 228,481 | +105,826 | +46.32% | skill 1/1; base 1/1 |
| claude-code | cupy-multiprocessing-fork | 78,073 | 71,859 | +6,214 | +8.65% | skill 1/1; base 1/1 |
| claude-code | cupy-port-heat-diffusion | 344,442 | 225,952 | +118,490 | +52.44% | skill 1/1; base 1/1 |
| claude-code | cupy-port-montecarlo-stats | 502,294 | 322,378 | +179,916 | +55.81% | skill 1/1; base 1/1 |
| claude-code | cupy-port-signal-pipeline | 346,465 | 441,387 | -94,922 | -21.51% | skill 1/1; base 1/1 |
| claude-code | cupy-rawkernel-debug-garbage | 78,884 | 38,084 | +40,800 | +107.13% | skill 1/1; base 1/1 |
| claude-code | cupy-small-launch-bound | 399,728 | 374,159 | +25,569 | +6.83% | skill 1/1; base 1/1 |
| claude-code | cupy-tf32-matmul | 77,925 | 37,617 | +40,308 | +107.15% | skill 1/1; base 1/1 |
| claude-code | cupy-whole-array-reduction | 78,391 | 37,160 | +41,231 | +110.96% | skill 1/1; base 1/1 |
| claude-code | negative-numba-cuda-kernel | 35,860 | 35,494 | +366 | +1.03% | skill 1/1; base 1/1 |
| claude-code | negative-numpy-no-gpu | 37,265 | 36,809 | +456 | +1.24% | skill 1/1; base 1/1 |
| codex | All cases | 4,143,560 | 3,722,375 | +421,185 | +11.31% | skill 17/17; base 17/17 |
| codex | cupy-async-timing | 35,110 | 18,107 | +17,003 | +93.90% | skill 1/1; base 1/1 |
| codex | cupy-duplicate-index-assign | 454,542 | 313,588 | +140,954 | +44.95% | skill 1/1; base 1/1 |
| codex | cupy-float64-consumer-gpu | 56,585 | 38,585 | +18,000 | +46.65% | skill 1/1; base 1/1 |
| codex | cupy-float64-promotion-slowdown | 339,450 | 415,064 | -75,614 | -18.22% | skill 1/1; base 1/1 |
| codex | cupy-griddata-port | 382,796 | 365,176 | +17,620 | +4.83% | skill 1/1; base 1/1 |
| codex | cupy-jit-block-reduction | 153,810 | 122,371 | +31,439 | +25.69% | skill 1/1; base 1/1 |
| codex | cupy-matrix-exponential | 248,762 | 222,279 | +26,483 | +11.91% | skill 1/1; base 1/1 |
| codex | cupy-multiprocessing-fork | 39,339 | 26,903 | +12,436 | +46.23% | skill 1/1; base 1/1 |
| codex | cupy-port-heat-diffusion | 488,617 | 359,498 | +129,119 | +35.92% | skill 1/1; base 1/1 |
| codex | cupy-port-montecarlo-stats | 538,368 | 401,077 | +137,291 | +34.23% | skill 1/1; base 1/1 |
| codex | cupy-port-signal-pipeline | 533,884 | 627,828 | -93,944 | -14.96% | skill 1/1; base 1/1 |
| codex | cupy-rawkernel-debug-garbage | 108,372 | 32,403 | +75,969 | +234.45% | skill 1/1; base 1/1 |
| codex | cupy-small-launch-bound | 540,906 | 615,933 | -75,027 | -12.18% | skill 1/1; base 1/1 |
| codex | cupy-tf32-matmul | 85,720 | 43,702 | +42,018 | +96.15% | skill 1/1; base 1/1 |
| codex | cupy-whole-array-reduction | 82,363 | 62,106 | +20,257 | +32.62% | skill 1/1; base 1/1 |
| codex | negative-numba-cuda-kernel | 22,348 | 26,336 | -3,988 | -15.14% | skill 1/1; base 1/1 |
| codex | negative-numpy-no-gpu | 32,588 | 31,419 | +1,169 | +3.72% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,763,356 | 7,243,241 | +1,520,115 | +20.99% | skill 34/34; base 34/34 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Skill name very short: 'cupy' (`skills/cupy/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cupy/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cupy/SKILL.md`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: write a custom kernel.
   The same applies to many small problems: stack them into one batched array
   (`cp.linalg.inv`, `solve`, `matmul`, `fft` all accept a leading batch dimension).
   Loops of in (`SKILL.md:89`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: write a custom kernel.
   The same applies to many small problems: stack them into one batched array
   (`cp.linalg.inv`, `solve`, `matmul`, `fft` all accept a leading batch dimension).
   Loops of in (`SKILL.md:89`)
- 7 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
