# ADR-0011: DeepSeek tool-calling check (step 9a) and the LLM driver (step 10)

Status: accepted. 9a done. Step 10 is done: the driver was contract-tested offline,
the first real episodes ran (user-approved, 2026-11-12) and the token budget was
revised. The scored episode is committed as `samples/deepseek-flash-frozenlake/`
(see docs/EVIDENCE.md).

## 9a: findings (DeepSeek official docs, read 2026-11-11; no API calls)
Sources: api-docs.deepseek.com (`/updates`, `/quick_start/pricing`,
`/guides/thinking_mode`, `/guides/tool_calls`, `/api/create-chat-completion`,
`/quick_start/error_codes`, `/quick_start/rate_limit`). The docs were read
through a summarising fetch tool: quoted rules are exact, the rest is close
paraphrase. The first real call is therefore a `GET /models` preflight.

1. **`deepseek-reasoner` and `deepseek-chat` are retired.** The 2026-04-24
   update: "The two legacy API model names, `deepseek-chat` and
   `deepseek-reasoner`, will be discontinued in three months (2026-07-24)."
   Current ids:
   - `deepseek-flash` (V4.1-Flash);
   - `deepseek-v4-pro`.

   Both have 1M context, thinking and non-thinking modes, and thinking on by default.
2. **Tool calling works in thinking mode** ("Starting with DeepSeek-V3.2, the API
   supports tool use in thinking mode"). Restrictions:
   - `tool_choice` may only be `auto` or `none`; `required` or a named function is a 400;
   - `temperature` and the penalty parameters have no effect in thinking mode.
3. **`reasoning_content` must be passed back.** When a request contains `tools`,
   "the `reasoning_content` must be fully passed back to the API in all subsequent
   requests". Otherwise: "the API will return a 400 error."
4. Thinking is toggled with the body field `thinking: {"type": "enabled"|"disabled"}`;
   effort with `reasoning_effort`.
5. `finish_reason` values: `stop`, `length`, `content_filter`, `tool_calls`,
   `insufficient_system_resource`, `aborted`.
6. Errors and retries:
   - 400/401/402 (insufficient balance)/422 are client errors: don't retry;
   - 429/500/503: retry after a wait;
   - non-started requests are closed after 10 minutes;
   - no `seed` parameter is documented, so outputs aren't deterministic.
7. `usage` reports `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` and
   `completion_tokens_details.reasoning_tokens`.
8. Price per 1M tokens, off-peak / peak:

   | Model | Cache-hit input | Cache-miss input | Output |
   |---|---|---|---|
   | `deepseek-flash` | 0.003 / 0.006 | 0.15 / 0.30 | 0.60 / 1.20 |
   | `deepseek-v4-pro` | 0.022 / 0.044 | 0.66 / 1.32 | 1.98 / 3.96 |

**Conclusion for the driver:**
- default model `deepseek-flash`, thinking enabled;
- pass every assistant message back unchanged (including `reasoning_content`);
- `tool_choice: auto`;
- `max_tokens` set explicitly (32K; the thinking-mode default is 64K);
- reject the retired ids with a clear error.

## Decision: the driver (`rlenv/agents/llm.py`)
| Aspect | Choice | Why |
|---|---|---|
| HTTP | stdlib `urllib`, not the `openai` SDK | no dependency; every request detail is visible and testable (the key only ever sits in the `Authorization` header). The `llm` extra was dropped from `pyproject.toml` |
| Tools | `openai_tools()` from the single `TOOL_SPECS` | one definition for every driver (ADR-0005) |
| Tool-call loop | calls executed one by one through the harness. Every tool-call id gets exactly one `role: tool` reply, in order, before the next request | API contract. Parallel-call semantics are undocumented, so execution is sequential |
| Reasoning | `reasoning_policy="pass_back"` (default; DeepSeek requires it with tools). `strip_all` exists for other OpenAI-compatible providers | 9a finding 3; a contract test checks every request carries all earlier reasoning |
| Model mistakes | invalid JSON → harness tool error "arguments are not valid JSON: …"; unknown tool → "unknown tool"; text-only reply → bounded nudges, then the driver stops | model mistakes are learning signal (verl practice), never crashes; every call is budgeted |
| `finish_reason` | `insufficient_system_resource`/`aborted` → retry the same request (≤ 2); `length` without tool calls and `content_filter` → error | server-side vs request problems |
| API failures | 429/5xx retried with backoff by the client; persistent errors raise → `agent_error` → reward **None** (`agent_driver_error`) | an outage is not evidence about the policy (ADR-0007) |
| Cost guard | **billable-token budget** `max_billable_tokens` (default 200k = cache-miss input + output) plus a loose **hard cap** `max_total_tokens` (default 2M, includes cached input). From 80 % of either, every tool result carries a `WARNING` to wrap up and submit (`llm_budget_warning` event, once); at 100 % the driver stops (`llm_stop`) | bounded spend even if the model loops, measured in what actually costs money; the model gets a chance to finish. Revised after the first real run, see "Token budget revision" |
| Transcript | every model call logged as an `llm_call` event (new messages, full response including reasoning, params, usage with cache/reasoning tokens, latency) | the exact conversation is in the trajectory; the `messages` export uses it (`"transcript": "exact"`) instead of reconstructing |
| Determinism | not available (no seed; temperature ignored in thinking mode). `--thinking disabled` + temperature 0 is the most reproducible setting | documented; replay works from the transcript |
| Secrets | key read once from `DEEPSEEK_API_KEY`; never in messages, events or error texts; error bodies truncated; the trajectory redactor is a second line; the sandbox never inherits the env | project rule: the key is read only from the `DEEPSEEK_API_KEY` environment variable and is never written to the repo, trajectories or logs; tests set a fake key and grep every written file |

### Opt-in (no accidental spending)
- `rlenv run TASK --agent llm` requires `--allow-external-api` **and**
  `DEEPSEEK_API_KEY`; without either it refuses before any network access.
- `rlenv llm-preflight --allow-external-api` makes one `GET /models` call to check
  the key and that the model id exists.
- As a standing project rule, a real run also requires explicit user confirmation in the conversation.

## Cost estimate for one real episode (`deepseek-flash`)
About 10–30 calls. Context grows to roughly 10–30k tokens, and most input is a
cache hit after the first call. Expected cost is a few US cents; the first real
run cost $0.03–0.06.

Upper bound with the default budgets, priced pessimistically (peak hours):
- 200k billable tokens, all priced as output ($1.20/M): $0.24;
- the 2M hard cap, all cached input ($0.006/M): $0.012;
- plus at most one call of overshoot (the budget is checked after each call).

That is about **$0.30** for `deepseek-flash`, and about $1 for `deepseek-v4-pro`.

## Alternatives considered
| Option | Why not |
|---|---|
| `deepseek-reasoner` | retired 2026-07-24 |
| `openai` SDK | an extra dependency and less control over headers, errors and retries; nothing we need is missing from a 100-line client |
| Strip earlier reasoning to save tokens | the API returns 400 when tools are present |
| Force `submit` with `tool_choice` | not allowed in thinking mode; budgets and nudges terminate the episode instead |
| Strict tool schemas (beta endpoint) | our schemas use `maxItems`/`minimum`, which strict mode doesn't support; the harness validates arguments anyway |

## First real runs (2026-11-12, user-approved)
| Run | Model | Outcome |
|---|---|---|
| 1 | `deepseek-flash` | 5 tool calls, then **reward None** (`agent_driver_error`). The cause was a harness bug, not the model: the sandbox child chdirs to the workspace before `preexec_fn`, so the CLI's relative `--out runs` broke every sandbox run. Tests always used absolute paths. Fixed in `ad048a2` (absolute paths in the sandbox, plus a regression test). This is the unscorable path working as designed: an infrastructure fault did not become a 0. 13.7k tokens |
| 2 | `deepseek-flash` | **reward 1.0** (success, progress and report all 1.0). Expected SARSA: 3 independent learners trained sequentially, best table selected by rollouts. Hidden maps: worst-map z = 1.034 (map 0: 0.468 vs optimal 0.503; map 1: 1.0 vs 1.0). Report claims matched logged dev means; audit flags none. 29 tool calls, 10/10 training runs, 17 model calls, 580 s wall clock (231 s in the API) |

`deepseek-v4-pro` was not run: the fallback was approved only if flash failed.

Usage for run 2:
- prompt tokens: 399,459 (386,816 cache hits, about 97 %, and 12,643 misses);
- completion tokens: 43,515 (25,628 of them reasoning).

Cost is about **$0.03 (off-peak) to $0.06 (peak)**, roughly 90 % of it output tokens.

**Finding: the token budget is not aligned with cost.** `max_total_tokens` adds up
`total_tokens` on every call. Each call re-sends the whole growing context, so
cached prompt tokens dominate the count, although they are the cheapest tokens.
Run 2 hit the 400k cap without calling `submit`. Its last action wrote REPORT.md,
and the workspace was scored as designed. The cap stopped the run at about $0.05,
well below the ≈$0.50 bound estimated above. The budget is also checked after each
call, so it can overshoot by one call (443k against 400k).

## Token budget revision (decided by the user after run 2)
1. **The cost cap counts billable tokens only**: `completion_tokens +
   prompt_cache_miss_tokens` per call (`billable_tokens()`).
   - If the provider reports no cache usage, every prompt token counts (fail-safe:
     never under-count).
   - Measured in run-2 terms: 56k billable tokens instead of 443k, so the model
     would have had room to submit.
2. **A loose total-token hard cap** (2M) stays as a backstop. It covers a model
   that loops on cached context, and providers that mis-report cache usage.
3. **Warning from 80 %** of either budget.
   - From then on, every tool result gets a `WARNING` field: the budget is X % used,
     the episode stops when it runs out, ensure train.py/policy.py/REPORT.md exist,
     call submit, and start no new experiments.
   - The field is a key inside the JSON result. Upper case sorts first under
     `sort_keys`, so it survives truncation of long results and the message stays
     valid JSON.
   - The crossing is logged once as `llm_budget_warning`.

Rejected alternative: **a USD budget from a price table.** It needs per-model,
peak/off-peak prices kept in the code, and those change (the 2026-04 update changed
them). It would also make the stop point depend on the time of day, which makes
episodes harder to compare. Billable tokens are stable, provider-reported, and close
to cost: output dominates at 50–200× the price of cache hits.

Not changed: the budget is still checked after each call, because a call's cost is
unknown in advance. Overshoot is bounded by one call (`max_tokens` = 32k output plus
the new context). The 80 % warning makes reaching 100 % rare for a cooperative model.

Tests (fake LLM):
- the formula;
- a cache-heavy run is not stopped by the cost budget (fails under the old
  behaviour);
- the billable cap stops the agent;
- the warning appears only after crossing 80 %, is logged once, and the model can
  still submit;
- the warning survives truncation;
- invalid budget settings are rejected.
