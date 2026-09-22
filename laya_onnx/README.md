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
python -m laya_onnx.export.quantize_int8 <fp32_dir> <int8_dir>                      # int8-body
python -m laya_onnx.export.quantize_int8 <fp32_dir> <dir> --quantize-embeddings     # int8-all
python -m laya_onnx.bench.eval_ece <fp32_dir> <int8_dir> --n 300 --out study.json
```

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
remain unfitted and fall back to the per-qtype default of 1.0.

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

The `banking77` gain sits inside that noise: a ~25% disagreement rate can move a weak bucket's
accuracy either way, and it moved up here. It is not evidence that int8 helps.

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

- **Calibration survives int8.** After fitting, int8 ECE is within 0.02 of fp32 in every
  bucket for `int8-all` (worst: `noul:2`, 0.0658 vs 0.0494) and within 0.035 for `int8-body`
  (worst: `choice:6-10`, 0.1505 vs 0.1158); both are slightly *better* than fp32 in
  `choice:11+`. Had ECE been the only concern, int8 would ship.
- **Argmax accuracy does not survive it, in either configuration.** `noul:2` drops 10.3
  points, 0.8833 to 0.7800, over 300 held-out examples — and lands on exactly 0.7800 whether
  the embedding table is quantized or not. Per suite that is `toxic_chat` losing 21.7 (all) or
  25.0 (body) points. `choice:6-10` drops 6.0 / 4.0 and `choice:3-5` 3.3 / 4.0. Underneath
  those, fp32 and int8 pick a *different label* on up to 30% of examples. The plan's own rule
  for this case is explicit: *int8 argmax accuracy materially worse → do not ship int8.*
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

Write them into an export with:

```bash
python -m laya_onnx.export.refit_temps <export_dir> '{"choice:3-5": 1.4634, ...}'
```

They are fitted on five English-language suites. The checkpoint serves 100+ languages, and
nothing here measures whether a temperature fitted on English transfers to Khmer or Telugu.
Treat them as a first calibration, not a finished one.
