# laya-onnx

A torch-free ONNX runtime for laya's `multilingual` checkpoint (mmBERT-base, 322M, 1024/256
token budget). `laya_onnx.load(dir)` returns an `OnnxAgent` whose `predict` / `system_one`
return the same answer dicts as `laya.Agent`, from a process that has one runtime (OpenVINO by
default, or onnxruntime) and numpy, and neither torch nor transformers.

The fp32 export is verified against the torch path on real weights: max |delta| on the logits is
**3.719e-05** (`tests/test_onnx_local_e2e.py`). That test needs the real weights plus a ~1.3 GB
export on disk, so it is **not in CI and nothing in CI reproduces this number** — it is run by
hand. If you change the forward path and do not run it, the figure above has stopped being
checked by anything.

**Two runtimes, one export; OpenVINO is the default.** `laya_onnx.load(dir)` runs `model.onnx`
on OpenVINO's CPU plugin with fp32 pinned; `laya_onnx.load(dir, backend="onnxruntime")` runs the
same file on onnxruntime. The same 318 real-weights checks pass on both (max |delta| **4.005e-05**
on OpenVINO, **3.719e-05** on onnxruntime), and OpenVINO is the faster of the two on every
configuration measured here — see [OpenVINO backend](#openvino-backend) under Latency. Every
latency table *above* that section was taken on onnxruntime, before the default changed.

**Ship the fp32 export.** The int8 build is in the tree, is reproducible, and is measurably
worse where it matters: `noul:2` accuracy falls 0.8833 → 0.7800 (McNemar p = 1.6e-06; pooled
p = 3.1e-05), while ECE does not clearly separate the three graphs. The section below has the
full study. The latency table below is the other half of that decision — on the machine
measured here, int8 bought 4–23% and cost the labels.

## An export is two files, not one

`export_fp32` writes `model.onnx` **and** `model.onnx.data`, because the mmBERT-base weights
(~1.29 GB) exceed protobuf's 2 GB message limit well before you get to a single-file
serialization that onnx can load safely. `model.onnx` alone is ~2.8 MB of graph structure with
every initializer pointing at the sidecar. Copy only `model.onnx` to a serving host and you
deploy a model that cannot run. Both ends check for it: the exporter enforces the pairing at
write time, and `OnnxSession.__init__` walks the graph's declared external-data locations
(`declared_external_data`, a 40-line protobuf varint walker, because the runtime deliberately
does not ship `onnx`) and raises `FileNotFoundError` naming the missing sidecar before
onnxruntime gets to report it as an opaque external-data error from inside tensor loading. The
remaining rough edge is that the walker reads the whole graph file, which is free on fp32
(2.8 MB) and not on the single-file int8 builds (325–915 MB). Treat the export directory as the
unit of deployment:

```
<export_dir>/
  model.onnx                 # graph, ~2.8 MB
  model.onnx.data            # external initializers, ~1.29 GB -- required
  rl_agent_config.json       # max_len / head_max_len / temperatures -- required
  tokenizer/                 # required
```

`rl_agent_config.json` is equally required and fails better: `OnnxAgent.__init__` raises
`FileNotFoundError` naming the file if it is absent, and `ValueError` naming the missing key if
`max_len` or `head_max_len` is not in it. It never falls back to a default token budget, because
the wrong budget truncates the state silently instead of erroring (see `laya_onnx/runtime.py`'s
module docstring for why a 1024 default would be actively dangerous against a 512-token graph).

## Latency

One `system_one` call answers N typed questions about one state in one forward pass, so the
curve worth knowing is latency vs. questions per call. Every table in this section down to
"OpenVINO backend" is onnxruntime (`--backend onnxruntime`), taken before OpenVINO became the
default. `laya_onnx/bench/bench_latency.py`
measures it: 5 discarded warmup runs, then 50 timed runs, reported as nearest-rank p50 and p95
so every figure is a latency that was actually observed.

The warmup exists because onnxruntime defers part of its setup to the first `run()` at a given
input *shape*, and all three axes here are dynamic — but on this build the effect is smaller
than that reasoning suggests, and it is worth saying so rather than implying a big first-call
penalty. Measured:

- 1 question, runs 1–8: 169, 168, 155, 153, 152, 152, 151, 150 ms. Run 1 over run 6 is
  169/152 = **1.11x**, and the series has settled by run 3.
- 50 questions, runs 1–8: 6884, 6953, 6722, 6860, 6764, 6676, 6525, 6837 ms. Run 1 over run 6 is
  6884/6676 = **1.03x**, inside the run-to-run spread.

Both ratios are computed from the rounded values printed above, so they re-derive exactly; the
harness printed whole milliseconds for this series and no unrounded copy was kept. `warmup=5` is
cheap insurance here, not a large correction.

Reproduce with:

```bash
python -m laya_onnx.bench.bench_latency ~/laya_onnx_models/multilingual --backend onnxruntime --runs 50 --warmup 5
```

**Measured on one machine, and these numbers do not generalize.** Dual Intel Xeon Gold 6148
@ 2.40 GHz (40 physical cores / 80 logical, two sockets), Windows 11 26200, Python 3.13.9,
onnxruntime 1.27.0 (CPUExecutionProvider), numpy 2.3.4, torch 2.10.0+cpu. A dual-socket host is
close to the worst case for onnxruntime's default thread pool, which sizes itself to every
physical core and then pays NUMA traffic for the privilege; a 4-core container will produce a
different shape of curve, not just a shifted one.

**The declared floor is onnxruntime 1.30.0, and the tables were taken on 1.27.0.** Both are
true and neither is a contradiction: the floor is the release everything here was last *run*
on (the eight offline suites, and the shipped graph re-measured below), and the tables are left
as measured rather than re-taken, because re-taking them changed nothing. The shipped graph on
1.30.0, same host, same protocol as the 4-thread table (3 warmup / 20 timed):

| threads | questions per call | 1.27.0 p50 | 1.30.0 p50 |
|---|---|---|---|
| 4 | 1 | 231.3 ms | 235.3 ms |
| 4 | 5 | 1177.7 ms | 1164.5 ms |
| 4 | 10 | 2474.8 ms | 2316.4 ms |
| default | 1 | 149.5 ms | 158.1 ms |
| default | 5 | 641.0 ms | 666.4 ms |
| default | 10 | 1226.3 ms | 1249.0 ms |

Every row but one sits inside the run-to-run spread documented above, and that one (4 threads,
10 questions, 6%) is a single pair of runs. So the bump is a testing baseline, not a speed win;
an earlier one-off run that suggested 8–10% did not survive a clean repeat, and is not claimed.

fp32, onnxruntime's default thread count:

| questions per call | p50 ms | p95 ms |
|---|---|---|
| 1 | 164.5 | 169.5 |
| 5 | 627.4 | 681.6 |
| 10 | 1201.2 | 1288.3 |
| 50 | 6689.7 | 6871.5 |

Run-to-run variation across full repeats of the table was worst at 1 question — 164.5 and 155.0
p50 on two runs, a 6.1% spread — and under 2% at 5 and 50 questions (627.4 vs. 634.1; 6689.7 vs.
6752.4). Read these to two significant figures, not four.

**At 1 question, fp32 ONNX lands at 164.5 ms p50 — below the 193–464 ms torch-CPU band in the
root README, by 14.8% against its lower bound.** Faster than the band, not inside it. That
comparison is weaker than it looks: the README does not say what
machine produced 193–464 ms, so it is a published figure rather than a measurement anyone can
line up against this one. The comparison that *is* controlled is the same-machine one below.

### Same machine, torch vs. ONNX

`laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")`, identical state,
identical questions, same 5-warmup/50-run protocol, same host, each library at its own default
thread count:

| questions per call | torch p50 | ONNX fp32 p50 | ONNX vs. torch |
|---|---|---|---|
| 1 | 175.4 ms | 164.5 ms | **0.94x** (ONNX faster) |
| 5 | 447.8 ms | 627.4 ms | 1.40x (ONNX slower) |
| 10 | 1135.6 ms | 1201.2 ms | 1.06x (ONNX slower) |
| 50 | 3689.5 ms | 6689.7 ms | **1.81x** (ONNX slower) |

**Nothing in CI reproduces either column.** `bench_latency.py` measures the ONNX column only —
by construction, not by omission: it refuses to let torch into the process at all (item 4 of
`laya_onnx/bench/bench_latency.py`'s docstring, "Torch-free by construction", explains why a
latency number for the ONNX path measured beside a resident torch is not the number a
deployment gets), and its `main()` calls `laya_onnx.load` and nothing else. The torch column
comes from `laya_onnx/bench/bench_torch.py`, which imports the same `STATE`, `questions()` and
`measure()` and runs them against `laya.load(...)` in its own process. The table above predates
that script — its torch p50s came from an equivalent off-tree run with the same protocol — and
the 4-thread table below is the first one both scripts produced. Re-running the comparison is
always two processes, not one: a single process that timed both would let whichever library
initialised first shape the other's thread pool.

**At each library's default thread count the ONNX port is not a speed win on this host, and
above one question per call it is a loss.** That is worth saying plainly, because "export to
ONNX" is usually pitched as an optimization. The reason to take this port is the one in the
package docstring: a serving process with onnxruntime and numpy and *neither torch nor
transformers*, which is a smaller image, a faster cold start and a much smaller dependency
surface. Latency parity at N=1 is the bar it has to clear, and it clears it. The loss at N=5
and N=50 is a property of the 80-way default pools on this two-socket machine, not of the
graph: pin both libraries to 4 threads and it is gone (the "Four threads" table below). *Why*
torch's default pool batches better than onnxruntime's on two sockets is still unmeasured, and
stays a hypothesis.

### Thread count

fp32 again, with `threads=8` (`OnnxSession` sets `intra_op_num_threads`), which is closer to a
CPU-quota'd container than the 80-way default:

| questions per call | p50 ms (ORT default) | p50 ms (threads=8) |
|---|---|---|
| 1 | 164.5 | 153.0 |
| 5 | 627.4 | 737.2 |
| 10 | 1201.2 | 1460.7 |
| 50 | 6689.7 | 8309.1 |

Eight threads is the better setting for single-question calls on this host and the worse one
for batched calls. There is no default that is right for both; pass `threads=` deliberately,
especially under a process pool, where leaving it unset gives every worker a full-width pool.

#### Four threads, both libraries

The commodity case, as close as this host can get to it: `bench_latency --backend onnxruntime --threads 4` and
`bench_torch --threads 4`, same state, same questions, two processes, **3 warmup and 20 timed
runs** rather than the 5/50 of the tables above — read these to two significant figures too.
Four threads of a Xeon Gold 6148 (AVX-512, 2.4 GHz) is a proxy for a small CPU-quota'd
container, not for a laptop; nobody has measured a laptop.

| questions per call | torch p50 (4 threads) | ONNX fp32 p50 (4 threads) | ONNX vs. torch |
|---|---|---|---|
| 1 | 288.7 ms | 254.3 ms | **0.88x** (ONNX faster) |
| 5 | 1187.6 ms | 1190.7 ms | 1.00x |
| 10 | 2312.7 ms | 2434.5 ms | 1.05x (ONNX slower) |

The 1.40x and 1.81x losses of the default-threads table are gone. The honest summary is
therefore: at the thread counts this port is for, fp32 ONNX is at parity with torch or ahead of
it, and the two-socket numbers above are what happens when each library is handed 40 physical
cores and sizes its own pool.

**Where the remaining time goes, and what does not recover it.** An onnxruntime profile of a
5-question call at 4 threads puts `MatMul` at 56% of kernel time, with `Transpose`,
`LayerNormalization`, `Softmax` and `Split` — the rotary and attention glue of a ModernBERT
layer — at another ~27%. onnxruntime's own graph optimizer (`ORT_ENABLE_ALL`, the level
`OnnxSession` sets) fuses almost none of it on this dynamo capture: one
`SkipLayerNormalization`, 24 `FusedMatMul`, no GELU and no attention fusion. Its offline
transformer optimizer (`onnxruntime.transformers.optimizer`, `model_type="bert"`, 12 heads,
hidden 768) fuses the 24 GELUs and nothing else — zero `Attention`, `MultiHeadAttention` or
`RotaryEmbedding` matches — and the fused graph answered the six benchmark questions
identically to four decimals at 0.3–2.0% lower p50 at 4 threads, inside run-to-run spread. So
the fp32 headroom is real but not a switch: fusing this model's attention means custom pattern
work against this specific capture (rotary on q and k, a 128-token sliding-window mask on every
layer but each third), and nothing in this tree has attempted it.

### OpenVINO backend

The same shipped fp32 export, loaded unmodified by each runtime, measured with
`bench_latency --backend onnxruntime|openvino`, 3 warmup / 20 timed runs, nearest-rank p50, one
process per row. Same host as every table above; Python 3.13.9, onnxruntime 1.30.0, OpenVINO
2026.4.0, numpy 2.5.3, tokenizers 0.22.2.

| threads | questions per call | onnxruntime p50 | OpenVINO p50 | ORT / OpenVINO |
|---|---|---|---|---|
| default | 1 | 152.4 ms | 73.9 ms | **2.06x** |
| default | 5 | 671.5 ms | 255.6 ms | **2.63x** |
| default | 10 | 1262.1 ms | 486.3 ms | **2.60x** |
| default | 50 | 6749.2 ms | 2378.9 ms | **2.84x** |
| 4 | 1 | 217.7 ms | 207.9 ms | 1.05x |
| 4 | 5 | 1125.8 ms | 951.7 ms | 1.18x |
| 4 | 10 | 2332.4 ms | 1791.8 ms | 1.30x |
| 4 | 50 | 13167.6 ms | 9028.5 ms | 1.46x |

**Read the two halves separately, because they answer different questions.**

- *At each runtime's default* OpenVINO is 2.1–2.8x faster. Part of that is not kernels but
  thread placement: OpenVINO's latency hint sizes its pool to one socket's physical cores
  (`OpenVinoSession.threads` reports 20 here) and keeps it there, where onnxruntime's default pool
  is the one this README already shows paying for the second socket. On this host that is a real
  and repeatable win, but it is a property of a two-socket box, not of the graph.
- *At an equal 4 threads* — the commodity-container proxy used above — the gap is 1.05x at one
  question, inside run-to-run spread, growing to 1.46x at 50. That growth is the kernel and fusion
  difference showing through as batches get larger; a single question on four cores gains nothing
  worth claiming.

So on a many-core host, `backend="openvino"` roughly halves latency or better; on a small quota,
expect parity at one question and a moderate gain on batched calls. Neither half measures a
laptop or a Linux box.

Arithmetic is not what changed. The real-weights e2e suite passes 318/318 under OpenVINO, with
max |delta| on raw logits against torch of **4.005e-05** (onnxruntime: 3.719e-05) and identical act
probabilities; `tests/test_onnx_openvino.py` holds the two backends within 1e-4 of each other on
the offline fixture at shapes the trace never saw, in CI. OpenVINO's inference precision is
pinned to f32 and read back after compile: on a CPU with AMX or AVX512_BF16 its plugin would
otherwise choose bf16 by itself, which would be a different model from the one every number in
this file describes.

**The second socket is still idle.** Windows shows a process one processor group — 40 of this
host's 80 logical CPUs — and OpenVINO's threads stayed on socket 0 even when the process was
started on node 1; its multi-stream and `TENSOR_PARALLEL` modes gained 0–26% in exploratory runs
(`docs/superpowers/specs/2026-09-23-openvino-backend.md`). Using both sockets means one worker
process per socket, or Linux; neither is measured here.

Reproduce:

```bash
python -m laya_onnx.bench.bench_latency ~/laya_onnx_models/multilingual --backend openvino --runs 20 --warmup 3
python -m laya_onnx.bench.bench_latency ~/laya_onnx_models/multilingual --backend openvino --threads 4 --runs 20 --warmup 3
```

### int8, for completeness — still not recommended

| questions per call | fp32 p50 | int8 p50 | speedup |
|---|---|---|---|
| 1 | 164.5 | 134.2 | 1.23x |
| 5 | 627.4 | 598.0 | 1.05x |
| 10 | 1201.2 | 1153.3 | 1.04x |
| 50 | 6689.7 | 6262.4 | 1.07x |

int8 is 4–23% faster and ~4x smaller on disk (325 MB in one file, versus 1.29 GB in two), and it
loses `noul:2` accuracy 0.8833 → 0.7800 at p = 1.6e-06. That is the trade, measured; the next
section is why it is not worth taking.

## int8 quantization: measured, and not recommended

`laya_onnx/export/quantize_int8.py` produces a dynamically quantized int8 copy of an fp32
export. It works, and it is up to 3.97x smaller. **Ship fp32 anyway.** The reason is in the
tables below, and it is not the one you would expect: **the accuracy is what breaks.** Paired
McNemar puts the `noul:2` accuracy loss at p = 1.6e-06 and the pooled loss over 750 held-out
rows at p = 3.1e-05. ECE, meanwhile, does not distinguish the three graphs at this sample
size — with one nominal exception, `choice:3-5` on int8-body at permutation p = 0.030, which is
recorded and weighed below rather than smoothed away. So the failure is in the labels, not in
the calibration that a temperature refit could repair.

Two int8 configurations were measured, not one, on the same data and the same held-out split:

| tag | `quantize_embeddings` | model size | quantized initializers | `MatMulInteger` nodes |
|---|---|---|---|---|
| `fp32` | — | 1290.5 MB (2.9 MB graph + 1287.7 MB `.onnx.data`) | 0 | 0 |
| `int8-all` | `True` (stock `quantize_dynamic`) | 324.7 MB | 102 | 100 |
| `int8-body` | `False` (**default**) | 914.6 MB | 100 | 100 |

Each size above is rounded on its own, which is why the fp32 row's two parts read as 1290.6
against a 1290.5 total. Neither figure was adjusted to make the addition come out.

The only difference between the two int8 graphs is the [256000, 768] token-embedding table,
which holds ~196M of the checkpoint's ~322M parameters. Stock `quantize_dynamic` quantizes it
through `Gather`, *per-tensor* — the written graph carries a scalar `..._scale` and
`..._zero_point`, one pair for all 256,000 rows. `int8-body` leaves it in `FLOAT`. The body is
quantized identically in both (100 `MatMulInteger`), so this is a controlled one-variable
comparison.

### What was measured

Five labelled public suites, 300 examples each (1500 records, one question per record), chosen
to populate four different `temp_bucket`s rather than one:

| suite | bucket | k | n |
|---|---|---|---|
| `fancyzhx/ag_news` | `choice:3-5` | 4 | 300 |
| `dair-ai/emotion` | `choice:6-10` | 6 | 300 |
| `mteb/banking77` | `choice:11+` | 20 | 300 |
| `SetFit/enron_spam` | `noul:2` | 2 | 300 |
| `lmsys/toxic-chat` (jailbreaking) | `noul:2` | 2 | 300 |

banking77 is asked against a 20-label candidate subset that always contains the gold label, not
against all 77: the full label list does not fit in this checkpoint's 256-token option budget,
and a truncated option list would silently drop the gold label for some examples. That makes it
an easier task than the published 77-way banking77 numbers and **its accuracy is not comparable
to them** — but it is a genuine high-cardinality choice, which is what the `choice:11+` bucket
needs.

Every bucket was split in half by `eval_ece.split_rows`: temperatures fitted on one half by
minimising NLL, all numbers below reported on the other half, which the fit never saw. fp32 and
int8 are scored on the same held-out examples.

Reproduce with:

```bash
pip install datasets                       # the five suites are fetched from the Hub
python -m laya_onnx.export.quantize_int8 <fp32_dir> <int8_dir>                      # int8-body
python -m laya_onnx.export.quantize_int8 <fp32_dir> <dir> --quantize-embeddings     # int8-all
python -m laya_onnx.bench.eval_ece <fp32_dir> <int8_dir> --n 300 --out study.json
```

`datasets` is needed only by `eval_ece`'s `__main__`, which is where the Hub download lives;
`measure_ece` and `refit` themselves take data as an argument and import nothing of the kind
(see `laya_onnx/bench/__init__.py`). It is not a dependency of the package.

### Per-bucket results (held-out half)

`ECE` is the standard top-1-probability estimator (`laya.common.ece_score`, 15 bins), the same
one the repository's other numbers use. `ECE_pub` measures the `confidence` field laya actually
publishes, which for `choice` is a normalized-entropy confidence rather than the top
probability; for `noul` the two coincide.

**Before fitting** — `laya-multilingual` ships no temperatures at all, so this is the raw
softmax, and it is the honest starting point rather than a regression:

| bucket | n | acc fp32 / int8-all / int8-body | ECE fp32 / int8-all / int8-body | ECE_pub fp32 / int8-all / int8-body |
|---|---|---|---|---|
| `choice:3-5` | 150 | 0.9400 / 0.9067 / 0.9000 | 0.0444 / 0.0751 / 0.0694 | 0.0348 / 0.0352 / 0.0529 |
| `choice:6-10` | 150 | 0.5400 / 0.4800 / 0.5000 | 0.3170 / 0.3515 / 0.3327 | 0.2712 / 0.2873 / 0.3032 |
| `choice:11+` | 150 | 0.6800 / 0.7267 / 0.6933 | 0.2064 / 0.1257 / 0.1438 | 0.2047 / 0.1335 / 0.1492 |
| `noul:2` | 300 | 0.8833 / 0.7800 / 0.7800 | 0.0980 / 0.1700 / 0.1636 | 0.0980 / 0.1700 / 0.1636 |

**After fitting** one temperature per bucket, each fitted against its own graph on the fit half:

| bucket | n | temp fp32 / int8-all / int8-body | acc fp32 / int8-all / int8-body | ECE fp32 / int8-all / int8-body | ECE_pub fp32 / int8-all / int8-body |
|---|---|---|---|---|---|
| `choice:3-5` | 150 | 1.4634 / 1.6894 / 1.6199 | 0.9400 / 0.9067 / 0.9000 | 0.0324 / 0.0351 / 0.0609 | 0.0653 / 0.0849 / 0.0891 |
| `choice:6-10` | 150 | 3.3767 / 3.2923 / 3.3370 | 0.5400 / 0.4800 / 0.5000 | 0.1158 / 0.1176 / 0.1505 | 0.2562 / 0.2269 / 0.2570 |
| `choice:11+` | 150 | 1.8443 / 1.5264 / 1.5198 | 0.6800 / 0.7267 / 0.6933 | 0.1159 / 0.1054 / 0.1048 | 0.1444 / 0.1438 / 0.1135 |
| `noul:2` | 300 | 3.5028 / 4.5373 / 4.9490 | 0.8833 / 0.7800 / 0.7800 | 0.0494 / 0.0658 / 0.0587 | 0.0494 / 0.0658 / 0.0587 |

No fitted temperature hit the `[0.5, 5.0]` clamp; the closest was int8-body `noul:2` at
**4.9490**, 99% of the way to the upper rail — close enough that a slightly different sample
would have clamped it, and a clamped bucket is one whose calibration was not achieved. Every
temperature is **above** 1, i.e. every bucket needed softening — this checkpoint is
over-confident everywhere, exactly as the main README's "Honest limits" section says.

Accuracy is identical before and after fitting in every row, as it must be: a temperature
divides all logits by the same scalar and cannot reorder them.

`choice:2` and every `score:*` bucket are **unmeasured** — no suite here produces them. They
remain unfitted and fall back to the per-qtype default of 1.0, i.e. a raw softmax. That is
**four of nine reachable buckets covered**. Nine, not twelve: `choice` and `score` each reach
all four size classes, but `sequence.render_options` builds a `noul` question's options as a
fixed `[false, true]` pair, so `noul` is always `k = 2` and `noul:3-5` / `noul:6-10` /
`noul:11+` cannot occur (`refit_temps.REACHABLE_BUCKETS`).

### How much of this is sampling noise?

Every number above is an estimate from 150 or 300 held-out rows, and an ECE quoted to four
decimal places invites comparisons that `n` does not support. So the noise is measured too.

**95% percentile-bootstrap intervals** (2000 resamples, seeded; the bootstrap rather than a
closed form because ECE is a sum over occupied bins of |mean confidence − mean accuracy| and
both the occupancy and the within-bin means are random). These are **not symmetric about the
point estimate and for ECE they lean high** — a sum of absolute values is bounded below by 0
and unbounded above, so resampling inflates a bin gap further than it cancels one, which is why
every ECE estimate below sits at or just above its lower bound. Read the upper end as "how bad
could this be", not as a symmetric error bar:

| bucket | n | accuracy fp32 / int8-all / int8-body | ECE fp32 / int8-all / int8-body |
|---|---|---|---|
| `choice:3-5` | 150 | 0.940 [0.900, 0.973] / 0.907 [0.860, 0.953] / 0.900 [0.853, 0.947] | 0.0324 [0.0221, 0.0731] / 0.0351 [0.0263, 0.0864] / 0.0609 [0.0418, 0.1055] |
| `choice:6-10` | 150 | 0.540 [0.460, 0.620] / 0.480 [0.400, 0.553] / 0.500 [0.420, 0.580] | 0.1158 [0.0942, 0.2096] / 0.1176 [0.0962, 0.2119] / 0.1505 [0.1170, 0.2434] |
| `choice:11+` | 150 | 0.680 [0.607, 0.747] / 0.727 [0.653, 0.793] / 0.693 [0.620, 0.767] | 0.1159 [0.0960, 0.2052] / 0.1054 [0.0854, 0.1833] / 0.1048 [0.0895, 0.1904] |
| `noul:2` | 300 | 0.883 [0.847, 0.920] / 0.780 [0.737, 0.827] / 0.780 [0.737, 0.827] | 0.0494 [0.0402, 0.0929] / 0.0658 [0.0448, 0.1112] / 0.0587 [0.0407, 0.1071] |

**Is the ECE difference distinguishable?** The comparison is *paired*: both graphs answered the
same questions, so their ECEs come from the same rows. `paired_ece_gap` therefore resamples row
indices **once** and scores both graphs on the same resampled rows, and its permutation null
exchanges each row's two `(confidence, correctness)` pairs with probability 1/2. Two answers per
cell — a bootstrap CI on the signed gap `ECE_fp32 − ECE_int8`, and a permutation p-value against
a null in which the graphs are interchangeable row by row:

| bucket | n | fp32 vs int8-all: gap [95% CI], floor, p | fp32 vs int8-body: gap [95% CI], floor, p |
|---|---|---|---|
| `choice:3-5` | 150 | −0.0027 [−0.0412, +0.0215], 0.0288, p = 0.86 | **−0.0285** [−0.0533, **+0.0006**], 0.0256, **p = 0.030** |
| `choice:6-10` | 150 | −0.0018 [−0.0763, +0.0756], 0.0697, p = 0.96 | −0.0347 [−0.1107, +0.0571], 0.0751, p = 0.38 |
| `choice:11+` | 150 | +0.0106 [−0.0470, +0.0792], 0.0519, p = 0.70 | +0.0112 [−0.0498, +0.0775], 0.0584, p = 0.72 |
| `noul:2` | 300 | −0.0164 [−0.0470, +0.0251], 0.0327, p = 0.33 | −0.0093 [−0.0419, +0.0299], 0.0303, p = 0.55 |

"floor" is the 95th percentile of |gap| under the permutation null: the smallest gap that would
be surprising at this `n`. A negative gap means int8 has the *higher* (worse) ECE.

**Seven of the eight gaps sit below their floor. One does not.** `choice:3-5` on int8-body has
|gap| = 0.0285 against a floor of 0.0256, permutation p = 0.030 — int8-body is *worse* calibrated
than fp32 there. Three things keep that from overturning the conclusion, and none of them is
that it is inconvenient:

- its paired bootstrap CI still contains zero, by 0.0006;
- it is one of **eight** comparisons. Under a global null the chance of at least one p < 0.05 is
  1 − 0.95⁸ ≈ 0.34, and a Bonferroni threshold would be 0.00625. A single nominal p = 0.030 is
  what this many tests produce by chance;
- it is in `int8-body`, the configuration that is not recommended on any grounds.

So the defensible claim is **ECE is not where int8 fails** — not the stronger "calibration
survives", and not the blanket "every gap is below its floor" an earlier draft of this file
asserted. Nothing here would have stopped int8 shipping; the accuracy result below is what does.

> **Correction.** The floors in the previous revision (0.0330 / 0.0398 / 0.0731 / 0.0821) were
> computed from two *independent* bootstrap resamples, one per graph. That discards the
> covariance between two ECEs measured on identical rows and inflates the floor by roughly
> 1.1–1.4x — and it meant this study used a paired test for accuracy and an unpaired null for
> ECE in the same section. The prose also described the procedure as splitting rows "into two
> independent halves", which was not what the code did (it drew two full-size samples). Both
> are fixed; the floors above are paired and full-sample, and the statistic now lives in
> `eval_ece.paired_ece_gap` with tests, rather than in an off-tree script. Tightening the floor
> promoted exactly one cell from "below" to "marginally above", recorded above rather than
> smoothed over.

**Paired McNemar on accuracy** — paired because both graphs answered the same questions, so the
two accuracies are strongly dependent and an unpaired test would overstate the variance. `b` is
fp32-right/int8-wrong, `c` the reverse; only discordant pairs carry information. Exact two-sided
binomial p-values (`b + c` is in the tens, so the chi-square approximation is unnecessary):

| bucket | n | fp32 vs int8-all (b, c, p) | fp32 vs int8-body (b, c, p) |
|---|---|---|---|
| `choice:3-5` | 150 | 6, 1, p = 0.125 | 6, 0, p = 0.031 |
| `choice:6-10` | 150 | 16, 7, p = 0.093 | 14, 8, p = 0.286 |
| `choice:11+` | 150 | 8, 15, p = 0.21 | 11, 13, p = 0.84 |
| `noul:2` | 300 | **35, 4, p = 3.4e-07** | **37, 6, p = 1.6e-06** |
| **pooled** | 750 | **65, 27, p = 9.3e-05** | **68, 27, p = 3.1e-05** |

This both confirms and narrows the conclusion. `noul:2` is decisive on its own, and the pooled
test over all 750 held-out rows is decisive. But the individual `choice` buckets are **not**
significant at n = 150 — including `choice:11+`, where int8 scored *higher* (p = 0.21 / 0.84),
which is the number behind this document's refusal to treat that gain as evidence. Earlier
drafts listed the per-bucket `choice` drops as though each were a finding; they are
directionally consistent but individually within noise, and the recommendation rests on
`noul:2` and the pooled result.

Every interval, p-value and floor above is emitted by the driver itself:

```bash
python -m laya_onnx.bench.eval_ece <fp32_dir> <int8_dir> --n 300 --rows-out rows.npz --out study.json
python -m laya_onnx.bench.eval_ece --rows-in rows.npz --rows-keys fp32,int8   # no weights needed
```

`--rows-in` reads the cached logits that `--rows-out` writes — under the array names `fp32` and
`int8`, which is what the second command passes and which is also the default, so `--rows-keys`
can be dropped. (An earlier revision of this file documented `fp32,int8_body`; no run of
`--rows-out` has ever written that name, and the command as printed would have failed with a
`KeyError`. The int8 *configuration* being compared is chosen by which export directory the
first command is pointed at, not by the cache key.)

**And the second command needs a cache this repository does not contain.** No `rows.npz` is
committed — it is tens of megabytes of logits — so "re-derives without the 1.3 GB checkpoint"
means: once someone has run the first command, every interval, p-value and floor above can be
recomputed, by them or by anyone they hand the file to, on a machine with no weights. It is not
a path from a fresh clone to these numbers. Nothing in CI runs either command. What CI does
check is the statistics themselves: `bootstrap_ci`, `paired_ece_gap`, `paired_mcnemar`,
`accuracy_of` and `ece_of` in `laya_onnx.bench.eval_ece` are unit-tested against hand-computable
cases in `tests/test_onnx_calibration.py`, which also drives `main()` end to end over a
synthetic cache.

### How faithful is the int8 graph, really?

The bucket table pools two suites into `noul:2`. Split per suite, over all 300 records of each
(not the held-out half, so these accuracies are not the table's), the picture is much starker:

| suite | fp32 acc | int8-all acc | int8-body acc | agree (all) | agree (body) | max abs logit delta (all / body) |
|---|---|---|---|---|---|---|
| `enron_spam` | 0.9967 | 0.9933 | 0.9933 | 0.9967 | 0.9900 | 16.37 / 17.38 |
| `toxic_chat` | 0.7900 | 0.5733 | **0.5400** | 0.7300 | **0.6967** | 16.13 / 19.17 |
| `ag_news` | 0.9400 | 0.9233 | 0.9233 | 0.9667 | 0.9667 | 14.61 / 13.77 |
| `emotion` | 0.5633 | 0.5000 | 0.5100 | 0.7933 | 0.8233 | 6.81 / 6.45 |
| `banking77` (20-way) | 0.6600 | 0.7300 | 0.7067 | 0.7700 | 0.7467 | 13.60 / 14.10 |

The whole of the `noul:2` loss is `toxic_chat`, which drops **21.7 points**. And the agreement
columns are the real verdict: **int8 returns a different label from fp32 on 17–30% of examples**
in three of the five suites. The fp32 ONNX export agrees with the torch path to 3.719e-05 on the
logits; both int8 graphs disagree with fp32 by up to **16–19 nats**. This is not a small
perturbation that calibration can absorb — the graphs are answering different questions.

The `banking77` gain sits inside that noise, and now there is a number for it: McNemar on
`choice:11+` gives p = 0.21 (int8-all) and p = 0.84 (int8-body). It is not evidence that int8
helps, and it was never treated as such.

**The embedding table is not the culprit.** The obvious suspect was the per-tensor quantized
256k-row embedding table in `int8-all`, so it was excluded and re-measured (`int8-body`). It
made no material difference: `noul:2` held-out accuracy is **0.7800 either way**, `toxic_chat`
got *worse* (0.5733 → 0.5400) with fp32-agreement falling 0.7300 → 0.6967, and the logit
deviations stayed in the same range. Sparing the table costs 590 MB — 324.7 MB becomes
914.6 MB — and buys nothing. **The damage is in the body's 100 quantized linear layers**, which
is where dynamic int8's per-tensor activation scaling meets an encoder whose activation ranges
are wide. That result is what moves "ship fp32" from one configuration's verdict to a
property of dynamic int8 on this checkpoint.

### The conclusion

- **ECE does not distinguish the three graphs at this sample size — a weaker claim than
  "calibration survives", and the one the data supports.** Seven of the eight per-bucket gaps
  sit below their own paired noise floor. The eighth (`choice:3-5`, int8-body: gap −0.0285,
  floor 0.0256, permutation p = 0.030) is nominally above it, but its bootstrap CI still
  contains zero and it is one of eight comparisons, where a single p = 0.030 is expected by
  chance. An earlier draft claiming int8 ECE is "within 0.02 of fp32 in every bucket" was
  quoting differences smaller than the measurement error. What can be said: **ECE is not where
  int8 fails**, and nothing in the calibration numbers would have stopped int8 shipping.
- **Argmax accuracy does not survive it, in either configuration — and unlike ECE, this one
  is significant.** `noul:2` drops 10.3 points, 0.8833 to 0.7800, over 300 held-out examples,
  landing on exactly 0.7800 whether the embedding table is quantized or not. Paired McNemar on
  that bucket: **p = 3.4e-07** (int8-all) and **1.6e-06** (int8-body). Pooled over all 750
  held-out rows: **p = 9.3e-05** and **3.1e-05**. Per suite the loss is `toxic_chat`, down 21.7
  (all) or 25.0 (body) points, and fp32 and int8 pick a *different label* on up to 30% of
  examples. The plan's rule for this case is explicit: *int8 argmax accuracy materially worse →
  do not ship int8.*
- **The individual `choice` buckets, however, do not carry the result.** `choice:6-10`
  (p = 0.093 / 0.286), `choice:11+` (p = 0.21 / 0.84) and `choice:3-5` (p = 0.125 for int8-all)
  are **not** significant at n = 150. An earlier draft listed their point drops as if they were
  evidence; they are directionally consistent but individually indistinguishable from noise.
  The decision rests on `noul:2` and on the pooled test.
- So the recommendation is **fp32**, and the interesting part is that the failure mode ran
  backwards from the one this task was written to catch. The warning was that accuracy would
  look fine while calibration rotted silently. What actually happened is that the calibration
  was repairable and the labels were not.

The quantizer, the measurement and the refit all stay in the tree, and
`quantize_embeddings` keeps both configurations reproducible. What has been ruled out is the
embedding table. What has not been tried is per-layer exclusion inside the body, static
(calibrated) quantization, or int8 with per-channel weights — any of which might recover the
labels, none of which this study measured. Those are hypotheses, written here as hypotheses.

### Fitted temperatures for the fp32 export

These are the fp32 column above, fitted on the fit half only. They are *the first temperatures
this checkpoint has ever had* — `laya-multilingual` ships `temperature_by_options: {}`:

```json
{"choice:3-5": 1.4634, "choice:6-10": 3.3767, "choice:11+": 1.8443, "noul:2": 3.5028}
```

**This mapping covers four of the nine reachable buckets.** The other five — `choice:2`,
`score:2`, `score:3-5`, `score:6-10`, `score:11+` — are not in it, and `build_answers` falls
back to the per-qtype default of 1.0 for any bucket it cannot find. A config carrying four
fitted temperatures looks calibrated and is not, so `write_temperatures` records the coverage
alongside the mapping, under `temperature_by_options_provenance`:

```bash
python -m laya_onnx.export.refit_temps <export_dir> '{"choice:3-5": 1.4634, ...}'   --provenance '{"graph": "fp32", "datasets": ["ag_news", "emotion", "banking77-20way",
                 "enron_spam", "toxic_chat"], "n_per_dataset": 300, "split": "50/50 per bucket"}'
```

It prints the fitted and the unfitted buckets, and writes both into the config. `OnnxAgent`
reads only `temperature_by_options`, so the provenance key is additive and an older runtime
ignores it.

Two further limits on these numbers. They are fitted on five **English** suites; the checkpoint
serves 100+ languages and nothing here measures whether a temperature fitted on English
transfers to Khmer or Telugu. And `noul:2` fitted at 3.5028 on fp32 but at **4.9490** on
int8-body — 99% of the way to the 5.0 clamp, which `rail_status` reports as `near_rail`: a fit
that lands on the rail is a fit whose optimum lies at or past the edge of the publishable
range, and should not be read as converged. Treat all of this as a first calibration, not a
finished one.
