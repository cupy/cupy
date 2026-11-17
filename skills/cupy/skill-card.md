## Description: <br>
Use for any task or question involving CuPy: writing, porting, speeding up, benchmarking, or debugging Python code that runs NumPy or SciPy array code on NVIDIA GPUs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to port NumPy/SciPy code to CuPy and cupyx.scipy, optimize and correctly benchmark GPU array code, debug custom CUDA kernels (RawKernel, ElementwiseKernel, cupyx.jit), and run CuPy with threads, multiprocessing, multiple GPUs, or containers. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html) <br>
- [Differences from NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html) <br>
- [Custom Kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html) <br>
- [cupyx.jit API](https://docs.cupy.dev/en/stable/reference/kernel.html) <br>
- [Performance Tips](https://docs.cupy.dev/en/stable/user_guide/performance.html) <br>
- [API Reference](https://docs.cupy.dev/en/stable/reference/) <br>
- [Upgrade Notes](https://docs.cupy.dev/en/stable/upgrade.html) <br>
- [CuPy source (GitHub)](https://github.com/cupy/cupy) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
17 evaluation tasks (15 positive, 2 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox` environment); dataset digest `sha256:7a15809daadc42b86b4e942e23f0cf3d5802689a18e50b9f7a74e4329fd83747` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.9`; evaluation date 2026-11-17. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 94.5% → 93.3% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.6% → 82.5% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
0eaef9c (source: git SHA, committed 2026-11-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
