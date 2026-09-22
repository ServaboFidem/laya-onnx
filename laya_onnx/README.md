# laya-onnx

A torch-free ONNX runtime for laya's `multilingual` checkpoint (mmBERT-base, 322M, 1024/256
token budget). `laya_onnx.load(dir)` returns an `OnnxAgent` whose `predict` / `system_one`
return the same answer dicts as `laya.Agent`, from a process that has onnxruntime and numpy and
neither torch nor transformers.

The fp32 export is verified against the torch path on real weights: max |delta| on the logits is
**3.719e-05** (`tests/test_onnx_local_e2e.py`, run manually — it needs weights on disk).

## int8 quantization: measured, and not recommended

`laya_onnx/export/quantize_int8.py` produces a dynamically quantized int8 copy of an fp32
export. It works, and it is 3.97x smaller. **Ship fp32 anyway.** The reason is in the table
below, and it is not the one you would expect: the calibration survives quantization, and the
*accuracy* does not.

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
python -m laya_onnx.bench.eval_ece <fp32_dir> <int8_dir> --n 300 --out study.json
```

### Per-bucket results (held-out half)

`ECE` is the standard top-1-probability estimator (`laya.common.ece_score`, 15 bins), the same
one the repository's other numbers use. `ECE_pub` measures the `confidence` field laya actually
publishes, which for `choice` is a normalized-entropy confidence rather than the top
probability; for `noul` the two coincide.

**Before fitting** — `laya-multilingual` ships no temperatures at all, so this is the raw
softmax, and it is the honest starting point rather than a regression:

| bucket | n | fp32 acc | int8 acc | fp32 ECE | int8 ECE | fp32 ECE_pub | int8 ECE_pub |
|---|---|---|---|---|---|---|---|
| `choice:3-5` | 150 | 0.9400 | 0.9067 | 0.0444 | 0.0751 | 0.0348 | 0.0352 |
| `choice:6-10` | 150 | 0.5400 | 0.4800 | 0.3170 | 0.3515 | 0.2712 | 0.2873 |
| `choice:11+` | 150 | 0.6800 | 0.7267 | 0.2064 | 0.1257 | 0.2047 | 0.1335 |
| `noul:2` | 300 | 0.8833 | 0.7800 | 0.0980 | 0.1700 | 0.0980 | 0.1700 |

**After fitting** one temperature per bucket, each fitted against its own graph on the fit half:

| bucket | n | temp fp32 | temp int8 | fp32 acc | int8 acc | fp32 ECE | int8 ECE | fp32 ECE_pub | int8 ECE_pub |
|---|---|---|---|---|---|---|---|---|---|
| `choice:3-5` | 150 | 1.4634 | 1.6894 | 0.9400 | 0.9067 | 0.0324 | 0.0351 | 0.0653 | 0.0849 |
| `choice:6-10` | 150 | 3.3767 | 3.2923 | 0.5400 | 0.4800 | 0.1158 | 0.1176 | 0.2562 | 0.2269 |
| `choice:11+` | 150 | 1.8443 | 1.5264 | 0.6800 | 0.7267 | 0.1159 | 0.1054 | 0.1444 | 0.1438 |
| `noul:2` | 300 | 3.5028 | 4.5373 | 0.8833 | 0.7800 | 0.0494 | 0.0658 | 0.0494 | 0.0658 |

No fitted temperature hit the `[0.5, 5.0]` clamp; the closest was int8 `noul:2` at 4.5373. Every
temperature is **above** 1, i.e. every bucket needed softening — this checkpoint is
over-confident everywhere, exactly as the main README's "Honest limits" section says.

Accuracy is identical before and after fitting in every row, as it must be: a temperature
divides all logits by the same scalar and cannot reorder them.

`choice:2` and every `score:*` bucket are **unmeasured** — no suite here produces them. They
remain unfitted and fall back to the per-qtype default of 1.0.

### How faithful is the int8 graph, really?

The bucket table pools two suites into `noul:2`. Split per suite, over all 300 records of each
(not the held-out half, so these accuracies are not the table's), the picture is much starker:

| suite | fp32 acc | int8 acc | delta | fp32/int8 argmax agreement | max abs logit delta |
|---|---|---|---|---|---|
| `enron_spam` | 0.9967 | 0.9933 | −0.0033 | 0.9967 | 16.37 |
| `toxic_chat` | 0.7900 | 0.5733 | **−0.2167** | **0.7300** | 16.13 |
| `ag_news` | 0.9400 | 0.9233 | −0.0167 | 0.9667 | 14.61 |
| `emotion` | 0.5633 | 0.5000 | −0.0633 | 0.7933 | 6.81 |
| `banking77` (20-way) | 0.6600 | 0.7300 | +0.0700 | 0.7700 | 13.60 |

The whole of the `noul:2` loss is `toxic_chat`, which drops **21.7 points**. And the agreement
column is the real verdict: **int8 returns a different label from fp32 on 23–27% of examples**
in three of the five suites. The fp32 ONNX export agrees with the torch path to 3.719e-05 on the
logits; the int8 graph disagrees with fp32 by up to **16 nats**. This is not a small perturbation
that calibration can absorb — the two graphs are answering different questions.

The `banking77` gain sits inside that noise: a 23% disagreement rate can move a weak bucket's
accuracy either way, and it moved up here. It is not evidence that int8 helps.

### The conclusion

- **Calibration survives int8.** After fitting, int8 ECE is within 0.02 of fp32 in every bucket
  (worst: `noul:2`, 0.0658 vs 0.0494), and is slightly *better* in `choice:11+`. Had ECE been
  the only concern, int8 would ship.
- **Argmax accuracy does not survive it.** `noul:2` drops 10.3 points, 0.8833 to 0.7800, over
  300 held-out examples; per suite that is `toxic_chat` losing 21.7 points. `choice:6-10` drops
  6.0 points and `choice:3-5` 3.3. Underneath those, fp32 and int8 pick a *different label* on
  up to 27% of examples. The plan's own rule for this case is explicit: *int8 argmax accuracy
  materially worse → do not ship int8.*
- So the recommendation is **fp32**, and the interesting part is that the failure mode ran
  backwards from the one this task was written to catch. The warning was that accuracy would
  look fine while calibration rotted silently. What actually happened is that the calibration
  was repairable and the labels were not.

The quantizer, the measurement and the refit all stay in the tree: the study is reproducible,
and a future per-op exclusion experiment starts from here rather than from scratch. The 16-nat
logit deltas suggest looking first at the layers with the widest activation ranges rather than
at the head — but that is a hypothesis this study did not test, and it is written here as one.

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
