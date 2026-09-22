# Spec: ONNX port of the laya multilingual checkpoint

**Status:** accepted, not yet implemented
**Date:** 2026-09-21
**Plan:** [`../plans/2026-09-21-laya-onnx-port.md`](../plans/2026-09-21-laya-onnx-port.md)

## Goal

Run laya on commodity CPU hardware without PyTorch, by exporting `DecisionModel` to ONNX and
reimplementing the surrounding pipeline against `onnxruntime`. The target workload is English
prose, source code, JSON and logs.

## Why this is tractable

The model's export boundary already sits in the right place.

`DecisionModel.forward` (`laya/common.py:105`) is a pure tensor function:

```
(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)
```

Everything before it (`build_sequence`, `laya/common.py:49`) is pure Python over token ids.
Everything after it (`Agent.system_one`, `laya/agent.py:317-360`) is already pure NumPy —
temperature scaling, softmax, argmax, expected score, answer formatting. Neither side touches
torch tensors. So this is a single-graph export with a Python pipeline on both ends, not a
multi-stage model conversion.

## Checkpoint choice: `multilingual` (mmBERT-base), not `english`

The workload is English-and-code, so the `english` checkpoint (ModernBERT-large) looks like the
obvious pick. It is not, and the reason is the input budget.

| | `english` (ModernBERT-large) | `multilingual` (mmBERT-base) |
|---|---|---|
| params | 421M | 322M |
| context | 512 | **1024** |
| `head_max_len` | 192 | **256** |
| **state budget** | ~317 tokens | **~768 tokens** |
| English suites (macro) | **0.684** | 0.619 |
| prompt-injections | **0.698** | 0.578 |
| latency, 1 question (T4) | 39.5 ms | **32.8 ms** |
| fitted temperatures | ships with them | **none at all** |

**The deciding factor is truncation, not accuracy.** Code is token-dense; a medium function or a
stack trace exceeds 317 tokens, and `build_sequence` truncates silently (`laya/common.py:82`). A
model that cannot see the second half of the input still answers confidently about the half it
saw. That is a correctness failure with no signal attached. Six points of prose-benchmark macro
accuracy is a quality cost you can measure and live with; a truncated state is neither.

Note also that the published accuracy gap was measured on *short English prose* benchmarks
(AG News, BoolQ, SST-5), where nothing gets truncated. Those numbers do not describe this
workload.

**Secondary reason — mmBERT is smaller where it counts.** Its tokenizer is Gemma-based (the
reason `laya/agent.py:41` exists at all), so a large share of its 322M parameters sit in a
~256k-row embedding table. An embedding lookup is memory, not compute. ModernBERT-large's ~50k
English vocab puts far more of its 421M into the transformer body, which is what determines CPU
latency. The README's 2x GPU speedup should widen on CPU, which is more compute-bound.

**What this buys for free:** multilingual input works. The port does not need a script guard, and
an earlier draft of this spec that called for one has been dropped.

**What it costs:** the `multilingual` checkpoint ships with **no fitted temperatures at all**.
Calibration moves from a post-quantization contingency to a prerequisite — see below.

## Scope

**In scope**

- Export the `multilingual` checkpoint (mmBERT-base, 322M, `max_len=1024`, `head_max_len=256`).
- A `laya_onnx` package that answers the same typed questions with the same result shape.
- fp32 parity with the torch path, then int8 dynamic quantization.
- Temperature fitting, then re-measured calibration (ECE) on the quantized graph.
- Visibility into truncation, so the 768-token budget can be verified against real inputs.

**Out of scope**

- The `english` and `typed-decisions` checkpoints.
- `laya/router.py` — with one checkpoint there is nothing to route.
- `laya/lang.py` — mmBERT reads the scripts this would have rejected.
- Training, fine-tuning, and the RLCD reward path (`proper_reward`, `td_lambda_targets`).
- GPU execution providers. The point is commodity CPU.

## Locked decisions

### 1. Vendor the pipeline, do not refactor `laya/`

`build_sequence` and `render_options` live in `laya/common.py`, which imports torch at module
scope. Importing them drags torch back in, which defeats the purpose — dropping torch (~2 GB)
for onnxruntime (~50 MB) is most of the commodity-hardware win.

**Decision: vendor those functions into `laya_onnx/sequence.py`.** This repository is a fork that
tracks `NandhaKishorM/laya`; refactoring `laya/` would conflict on every upstream sync. The cost
of vendoring is drift, paid for by a parity test (Task 1) asserting the copy still produces
byte-identical sequences to upstream's.

**`laya/` is not modified by this work.** That is a hard constraint, not a preference.

### 2. fp32 parity first, int8 second

Landing fp32 first gives a known-good baseline to measure int8 drift against. Quantizing before
parity is established means any discrepancy has two possible causes.

### 3. Temperatures must be fitted, not just preserved

This checkpoint ships none. Until they are fitted, its probabilities are uncalibrated and the
`confidence` field is not meaningful — the README is explicit that `laya-multilingual` needs
temperatures fitted before anyone relies on its probabilities. Int8 then perturbs logits, so the
fit is done *against the quantized graph* that will actually ship.

Fitted values are clamped to `[0.5, 5.0]` (`clamp_temperature`, `laya/common.py:232`). The clamp
is not optional: the `english` checkpoint's shipped `choice:11+` bucket of 0.1006 multiplies
logits ~10x, publishing a 0.24 top probability as 0.99.

## Risks

| risk | mitigation |
|---|---|
| **The checkpoint is uncalibrated as shipped.** Headline risk, and larger than it was for `english`: there is no baseline to regress against, only one to establish. | Task 9 fits temperatures per `temp_bucket` on held-out data against the quantized graph, then reports ECE per bucket. Until that lands, treat `confidence` as unusable and say so in the README. |
| **Quantization silently degrades calibration.** Argmax accuracy looks fine while ECE drifts. | Measure ECE fp32 vs int8 before shipping int8. Ship fp32 if int8 cannot be calibrated back. |
| **Truncation is silent.** The reason this checkpoint was chosen; it must be verifiable, not assumed. | Task 2 reports tokens dropped per question, so the 768-token budget can be checked against real inputs instead of trusted. |
| **Shape-dependent branch bakes in at export.** `if p.size(-1) >= 2` / `topk(2)` (`laya/common.py:115`) traces to whichever branch the sample input hits. | Export with >= 2 markers; reject `k < 2` at the runtime boundary rather than answering from an untraced path. |
| **Vendored sequence builder drifts from upstream.** | Task 1's parity test fails the moment upstream changes `build_sequence`. |
| **Gemma tokenizer config quirks.** `laya/agent.py:41` exists because mmBERT/Gemma checkpoints store `extra_special_tokens` as a list where transformers expects a mapping, which breaks `AutoTokenizer`. | The adapter reads `special_tokens_map.json` and resolves ids via `token_to_id`, sidestepping `tokenizer_config.json` entirely. Task 3 tests this explicitly rather than assuming it. |
| **mmBERT export quirks.** If it shares ModernBERT's architecture — its RoPE 8192 context suggests it does — the flash-attn / unpadding path is not exportable. | Force `attn_implementation="sdpa"` (already the default in `build_model`) and disable `reference_compile` if present. Confirm at export, do not assume. |

## Interface contract

`laya_onnx` mirrors the torch API so a caller swaps one import:

```python
import laya_onnx
agent = laya_onnx.load("path/to/exported")
result = agent.predict(state, questions)
result["answers"]["department"]["choice"]
```

`predict` returns the same dict shape as `Agent.system_one`: `{"model", "answers", "usage"}`, with
per-question `choice` / `score` / `noul` payloads carrying `probabilities`, `confidence`, `action`.
`usage` gains a `truncated` block (Task 2) that the torch path does not have.

## Success criteria

1. fp32 ONNX logits match torch within `max|delta| < 1e-4` on the tiny test model and the real checkpoint.
2. `laya_onnx` imports and runs with **no torch installed**.
3. Answer dicts are structurally identical to the torch path on the shipped presets.
4. Truncation is reported, and measured against a sample of real inputs.
5. Temperatures are fitted and per-bucket ECE is reported for fp32 and int8.
6. CPU latency for a single question is measured against the README's 193-464 ms torch-CPU baseline.
