# Evaluation runs: Agents 2 and 3

Real-model runs on the two seeded Django apps, the 30,000+ character policy test, and how to reproduce them.
The result JSON files are not committed (`before_*.json` and `after_*.json` are git-ignored); this file records what was run and what came out.

## Setup

- Claims: `tests/fixtures/agent1_sample_output.json` (12 claims: 8 out of scope for Agents 2 and 3, 4 in scope).
- Apps: `tests/fixtures/apps/acmenotes_violating` and `tests/fixtures/apps/acmenotes_baseline`.
- Model: the `LLM_MODEL` in the runner's `.env`, temperature 0. Record it with
  `python -c "from llm.config import get_llm_config; print(get_llm_config().model)"`.
  Model used for these runs: `openai/gpt-oss-120b` (printed by the command above on the runner's machine)
- Script: `scripts/try_agent2_agent3.py` (options: `--policy`, `--policy-text`, `--drop-category`, `-q`, `--save`).
  It prints the commit, Agent 2 and Agent 3 retries and failures, `failed_categories`, `failed_calls`,
  the number of unmatched verdict ids, and the number of findings with `reason_code` `invalid_model_output`.
  With `--save` it writes the full Agent 2 and Agent 3 output plus a `run_info` block (commit, arguments).
- Comparison helper: `scripts/compare_runs.py` prints status, category, reason code and the first 60 characters of each claim for saved runs.

## Commits

| Label | Commit | Notes |
|---|---|---|
| before | `9eedae9` | parent of `3f5aae4`; Agent 3 matches verdicts to claims by position |
| after | `6018f1a` | claim IDs; branch `feature/person2-measure-and-agent2-fixes` with main `61f71a5` merged (later merged to main as `9ab8892`) |

Between the two, only `agents/alignment_agent/agent.py` and `agents/alignment_agent/schemas.py` differ in logic
(`git diff --stat 3f5aae4^ 61f71a5 -- agents/alignment_agent agents/policy_agent llm tests/fixtures`);
Agent 2's only change in `6018f1a` is the new `failed_categories` field. The Agent 3 diff contains more than claim IDs
(reason codes, coverage gap, untrusted-data fences, `policy_text`), so the comparison cannot attribute differences to claim IDs alone.
The "before" runs used a copy of the newer script placed in a worktree of the old commit, so their `commit:` line shows uncommitted changes.

## Result files (local, not committed)

| File | Commit | App | Policy text |
|---|---|---|---|
| `before_violating_1.json`, `before_violating_2.json` | 9eedae9 | violating | none |
| `before_baseline_1.json`, `before_baseline_2.json` | 9eedae9 | baseline | none |
| `after_violating_1.json`, `after_violating_2.json` | 6018f1a | violating | none |
| `after_baseline_1.json`, `after_baseline_2.json` | 6018f1a | baseline | none |
| `after_short_text_violating.json` | 6018f1a | violating | `synthetic_privacy_policy.txt` (856 characters, not truncated) |
| `after_long_violating.json`, `after_long_violating_2.json` | 6018f1a | violating | `long_policy_tmp.txt` (47,830 characters, truncated) |

## Results

Counts are aligned / partial / violated / undeclared. `out_of_scope` was 8 in every run.
In every run: 0 LLM retries, 0 failures, `failed_calls` empty, `failed_categories` empty (after runs), and, for the after runs, 0 unmatched verdict ids and 0 findings with `reason_code` `invalid_model_output`.
The before code has no unmatched-id count (it matches by position).

| Run | Counts |
|---|---|
| violating before 1 | 1 / 1 / 2 / 3 |
| violating before 2 | 0 / 1 / 3 / 2 |
| violating after 1 and 2 | 0 / 1 / 3 / 2 |
| baseline before 1 and 2 | 1 / 3 / 0 / 0 |
| baseline after 1 and 2 | 2 / 2 / 0 / 0 |
| violating, real text, 856 chars | 0 / 1 / 3 / 2 |
| violating, long text, truncated, run 1 | 1 / 0 / 3 / 3 |
| violating, long text, truncated, run 2 | 1 / 0 / 3 / 2 |

Agent 2 evidence coverage (same before and after):

- violating (3 files): model_field 7, form_input 5, data_write 1, outbound_email 1, personal_data_logged 1, data_sharing 1.
- baseline (5 files): retention_logic 2, model_field 5, consent_logic 2, retention_field 1, setting 1, purpose_text 1, form_input 3, consent_ui 2, privacy_notice_link 1, data_write 1, outbound_email 1.

Agent 2 facts, counted per category and fact type in `after_violating_1.json` and `after_baseline_1.json` (present / absent / partial):

| Category | violating | baseline |
|---|---|---|
| transparency_notice | 3 / 3 / 0 | 6 / 1 / 0 |
| consent | 2 / 2 / 0 | 6 / 0 / 0 |
| purpose_limitation | 4 / 2 / 0 | 3 / 1 / 0 |
| data_minimization | 2 / 1 / 0 | 2 / 1 / 1 |
| storage_retention | 1 / 3 / 0 | 3 / 1 / 0 |
| total | 12 / 11 / 0 (23 facts) | 20 / 4 / 1 (25 facts) |

An `absent` fact means the static scan did not find the implementation; it is not a finding of non-compliance, and the baseline app also has some.
The fact statements themselves are in the saved JSON under `implementation.facts`.

Per-claim statuses seen in the after runs:

- violating (runs 1 and 2 identical): consent violated; purpose_limitation violated; storage_retention violated; transparency_notice partial. Undeclared: email logged (`views.py` line 13, purpose_limitation); Aadhaar and date of birth stored (`models.py` lines 10 and 11, transparency_notice, grouped as one).
- baseline (run 1 inspected per claim; run 2 has identical counts): consent aligned; purpose_limitation aligned; storage_retention partial; transparency_notice partial. In the before runs the baseline consent claim was partial.

Large-policy test: a 47,830-character policy gave `policy_text_chars` 47830 and `policy_text_truncated` True, with no failures. The transparency claim was aligned in both truncated runs and partial in the runs without truncated text; by the code, `policy_text` only reaches the undeclared call, so this is recorded as an open observation (the same verdict also appeared once in `before_violating_1`, which had no policy text).
Limits: the filler follows the real policy text, so the 30,000-character cap removed only filler; the case where the cut removes text that discloses a practice was not run, and the baseline app was not run with the long text.

## How to reproduce

From the repo root on the "after" commit (venv active, `.env` with a key):

```
python scripts\try_agent2_agent3.py tests\fixtures\apps\acmenotes_violating -q --save after_violating_1.json
python scripts\try_agent2_agent3.py tests\fixtures\apps\acmenotes_baseline -q --save after_baseline_1.json
```

Before runs (old commit in a separate worktree, using the same claims file and apps from the main folder):

```
git worktree add ..\dpdp-before 3f5aae4^
Copy-Item .env ..\dpdp-before\.env
Copy-Item scripts\try_agent2_agent3.py ..\dpdp-before\scripts\try_agent2_agent3.py
cd ..\dpdp-before
python scripts\try_agent2_agent3.py ..\DPDP-Compliance-System\tests\fixtures\apps\acmenotes_violating --policy ..\DPDP-Compliance-System\tests\fixtures\agent1_sample_output.json -q --save ..\DPDP-Compliance-System\before_violating_1.json
cd ..\DPDP-Compliance-System
git worktree remove ..\dpdp-before
```

Long policy (47,830 characters; the file is generated, not committed):

```
python -c "from pathlib import Path; t=Path('tests/fixtures/synthetic_privacy_policy.txt').read_text('utf-8'); f=''.join(f'\n\nAppendix {i}. Additional general information about our services, partners and operations, included for completeness.' for i in range(1,400)); Path('long_policy_tmp.txt').write_text(t+f,'utf-8'); print(len(t+f))"
python scripts\try_agent2_agent3.py tests\fixtures\apps\acmenotes_violating -q --policy-text long_policy_tmp.txt --save after_long_violating.json
```

Compare saved runs:

```
python scripts\compare_runs.py before_violating_1.json before_violating_2.json after_violating_1.json
```

Temperature 0 is not fully deterministic, so compare over at least two runs; two runs per condition is a small sample.

## Where failures appear

- Agent 2: a category whose LLM call gives up (after one retry on unusable output) adds 1 to `llm_failures`, is listed in `failed_categories`, and has no facts.
- Agent 3: a category with no facts becomes `unverifiable` with `reason_code` `evidence_gap` and no model call, so a failed Agent 2 category cannot become `violated`.
  A failed Agent 3 call adds 1 to `llm_failures` and a string to `failed_calls` (a `list[str]` on the result); its claims become `unverifiable` with `model_failure`.
  A verdict that cannot be matched to a claim id is recorded in `agent.unknown_verdict_ids` (an agent attribute, not part of the result dump) and the claim becomes `unverifiable` with `invalid_model_output`; it does not count in `llm_failures`.
- Orchestrator: `analysis_stage.py` marks Agent 2 incomplete when `llm_failures` is above 0; `alignment_stage.py` retries Agent 3 on model failures.

## Not done yet

- Answer keys per seeded app and metrics against them (precision, recall, alignment accuracy, false-positive rate).
- Long-policy variant where the cut removes disclosing text; long-policy run on the baseline app.
- Retention parser wired into Agent 3.