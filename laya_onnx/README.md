# laya-onnx

A torch-free ONNX runtime for laya's `multilingual` checkpoint (mmBERT-base, 322M, 1024/256
token budget). `laya_onnx.load(dir)` returns an `OnnxAgent` whose `predict` / `system_one`
return the same answer dicts as `laya.Agent`, from a process that has onnxruntime and numpy and
neither torch nor transformers.

The fp32 export is verified against the torch path on real weights: max |delta| on the logits is
**3.719e-05** (`tests/test_onnx_local_e2e.py`, run manually — it needs weights on disk).

**Ship the fp32 export.** The int8 build is in the tree, is reproducible, and is measurably
worse where it matters: `noul:2` accuracy falls 0.8833 → 0.7800 (McNemar p = 1.6e-06; pooled
p = 3.1e-05), while ECE does not clearly separate the two graphs. The section below has the
full study. The latency table below is the other half of that decision — on the machine
measured here, int8 bought 6–18% and cost the labels.

## An export is two files, not one

`export_fp32` writes `model.onnx` **and** `model.onnx.data`, because the mmBERT-base weights
(~1.29 GB) exceed protobuf's 2 GB message limit well before you get to a single-file
serialization that onnx can load safely. `model.onnx` alone is ~2.8 MB of graph structure with
every initializer pointing at the sidecar. Copy only `model.onnx` to a serving host and you
deploy a model that cannot run — and the failure arrives at `OnnxSession.__init__`, as a raw
onnxruntime error about a missing external-data file, not as anything that mentions the
export. **That is a known rough edge**: the exporter enforces the pairing at write time, the
session does not re-check it at load time. Treat the export directory as the unit of
deployment:

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
curve worth knowing is latency vs. questions per call. `laya_onnx/bench/bench_latency.py`
measures it: 5 discarded warmup runs (the first call with a given input *shape* pays
onnxruntime's fusion and arena setup, and all three axes here are dynamic), then 50 timed runs,
reported as nearest-rank p50 and p95 so every figure is a latency that was actually observed.

Reproduce with:

```bash
python -m laya_onnx.bench.bench_latency ~/laya_onnx_models/multilingual --runs 50 --warmup 5
```

**Measured on one machine, and these numbers do not generalize.** Dual Intel Xeon Gold 6148
@ 2.40 GHz (40 physical cores / 80 logical, two sockets), Windows 11 26200, Python 3.13.9,
onnxruntime 1.27.0 (CPUExecutionProvider), numpy 2.3.4, torch 2.10.0+cpu. A dual-socket host is
close to the worst case for onnxruntime's default thread pool, which sizes itself to every
physical core and then pays NUMA traffic for the privilege; a 4-core container will produce a
different shape of curve, not just a shifted one.

fp32, onnxruntime's default thread count:

| questions per call | p50 ms | p95 ms |
|---|---|---|
| 1 | 164.5 | 169.5 |
| 5 | 627.4 | 681.6 |
| 10 | 1201.2 | 1288.3 |
| 50 | 6689.7 | 6871.5 |

Run-to-run variation across full repeats of the table was within about 6% at every question
count (e.g. 1 question: 164.5 and 155.0 p50 on two runs), so read these to two significant
figures, not four.

**At 1 question, fp32 ONNX lands at 164.5 ms p50 — inside the README's 193–464 ms torch-CPU
band, at the fast end.** That comparison is weaker than it looks: the README does not say what
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

**The ONNX port is not a speed win on this host, and above one question per call it is a
loss.** That is worth saying plainly, because "export to ONNX" is usually pitched as an
optimization. The reason to take this port is the one in the package docstring: a serving
process with onnxruntime and numpy and *neither torch nor transformers*, which is a smaller
image, a faster cold start and a much smaller dependency surface. Latency parity at N=1 is the
bar it has to clear, and it clears it; throughput at N=50 is a cost it currently pays. The
likely cause is that torch's CPU GEMM path batches better than onnxruntime's on this two-socket
machine, but nothing here measures that, so it stays a hypothesis.

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
rows at p = 3.1e-05. ECE, meanwhile, does not distinguish the graphs at this sample size at
all — so the failure is in the labels, not in the calibration that a temperature refit could
repair.

Two int8 configurations were measured, not one, on the same data and the same held-out split:

| tag | `quantize_embeddings` | model size | quantized initializers | `MatMulInteger` nodes |
|---|---|---|---|---|
| `fp32` | — | 1290.5 MB (2.9 MB graph + 1287.7 MB `.onnx.data`) | 0 | 0 |
| `int8-all` | `True` (stock `quantize_dynamic`) | 324.7 MB | 102 | 100 |
| `int8-body` | `False` (**default**) | 914.6 MB | 100 | 100 |

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
python -m laya_onnx.bench.eval_ece --rows-in rows.npz --rows-keys fp32,int8_body
```

`--rows-in` reads the cached logits that `--rows-out` writes, so the whole statistical section
re-derives **without the 1.3 GB checkpoint** — the underlying functions are `bootstrap_ci`,
`paired_ece_gap`, `paired_mcnemar`, `accuracy_of` and `ece_of` in `laya_onnx.bench.eval_ece`,
all unit-tested in `tests/test_onnx_calibration.py`.

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
