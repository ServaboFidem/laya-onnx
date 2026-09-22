# laya-onnx

A torch-free ONNX runtime for laya's `multilingual` checkpoint (mmBERT-base, 322M, 1024/256
token budget). `laya_onnx.load(dir)` returns an `OnnxAgent` whose `predict` / `system_one`
return the same answer dicts as `laya.Agent`, from a process that has onnxruntime and numpy and
neither torch nor transformers.

The fp32 export is verified against the torch path on real weights: max |delta| on the logits is
**3.719e-05** (`tests/test_onnx_local_e2e.py`, run manually — it needs weights on disk).

## int8 quantization: measured, and not recommended

`laya_onnx/export/quantize_int8.py` produces a dynamically quantized int8 copy of an fp32
export. It works, and it is up to 3.97x smaller. **Ship fp32 anyway.** The reason is in the
tables below, and it is not the one you would expect: the calibration survives quantization,
and the *accuracy* does not.

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
both the occupancy and the within-bin means are random):

| bucket | n | accuracy fp32 / int8-all / int8-body | ECE fp32 / int8-all / int8-body |
|---|---|---|---|
| `choice:3-5` | 150 | 0.940 [0.900, 0.973] / 0.907 [0.860, 0.953] / 0.900 [0.853, 0.947] | 0.0324 [0.0221, 0.0731] / 0.0351 [0.0263, 0.0864] / 0.0609 [0.0418, 0.1055] |
| `choice:6-10` | 150 | 0.540 [0.460, 0.620] / 0.480 [0.400, 0.553] / 0.500 [0.420, 0.580] | 0.1158 [0.0942, 0.2096] / 0.1176 [0.0962, 0.2119] / 0.1505 [0.1170, 0.2434] |
| `choice:11+` | 150 | 0.680 [0.607, 0.747] / 0.727 [0.653, 0.793] / 0.693 [0.620, 0.767] | 0.1159 [0.0960, 0.2052] / 0.1054 [0.0854, 0.1833] / 0.1048 [0.0895, 0.1904] |
| `noul:2` | 300 | 0.883 [0.847, 0.920] / 0.780 [0.737, 0.827] / 0.780 [0.737, 0.827] | 0.0494 [0.0402, 0.0929] / 0.0658 [0.0448, 0.1112] / 0.0587 [0.0407, 0.1071] |

**What ECE gap is even distinguishable here?** Resampling one graph's rows into two independent
halves and taking |ECE_a − ECE_b| gives the distribution of a gap produced by sampling alone:

| bucket | n | 95th percentile of the null ECE gap |
|---|---|---|
| `choice:3-5` | 150 | 0.0330 |
| `noul:2` | 300 | 0.0398 |
| `choice:11+` | 150 | 0.0731 |
| `choice:6-10` | 150 | 0.0821 |

**Every fp32-vs-int8 ECE gap in this study is smaller than its bucket's noise floor.** The
largest is 0.0347 (`choice:6-10`, int8-body) against a floor of 0.0821. So ECE does not
separate the three graphs at this sample size, and any statement of the form "int8 ECE is
within X of fp32" for X in the 0.002–0.035 range is reporting measurement error. The
defensible claim is the weaker one: **ECE is not where int8 fails.**

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

Reproduce the intervals and tests from `laya_onnx.bench.eval_ece`'s `bootstrap_ci`,
`paired_mcnemar`, `accuracy_of` and `ece_of`.

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

- **No detectable ECE difference — which is a weaker claim than "calibration survives", and
  is the one the sample size supports.** Every ECE gap between fp32 and either int8 build is
  **below the noise floor at this n**. See the uncertainty section: the largest gap is 0.0347
  (`choice:6-10`, int8-body) against a null-resampling 95th percentile of 0.0821 for that
  bucket. ECE therefore does not distinguish the three graphs here, and an earlier draft of
  this file claiming int8 ECE is "within 0.02 of fp32" was quoting a difference smaller than
  the measurement error. What can be said: **ECE is not where int8 fails**, and nothing in the
  calibration numbers would have stopped it shipping.
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
