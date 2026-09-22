"""Measurement code for the ONNX port. Not part of the runtime, and not shipped-path.

`laya_onnx/bench/` exists so that a claim about the port -- "int8 costs 0.0X ECE", "the refit
recovered it" -- has a script behind it that anyone can re-run, rather than a number in a README
with no provenance. It is the ONNX-side counterpart of the repo's `research/` directory, and it
follows that directory's one hard rule in reverse: `research/` is never imported by the package,
and *this* package is never imported by `research/`.

Two boundaries hold here:

1. **`bench/` may import torch.** It is not the serving path. `eval_ece` reuses
   `laya.common.ece_score` rather than reimplementing the binning, because an ECE number that
   was computed by a second, subtly different estimator is not comparable to the ones already
   published in this repository's README and `research/results/*.json`. The cost of that reuse
   is a torch import, which is fine here and would not be fine one directory up.

2. **`bench/` never fetches data.** `measure_ece(agent, dataset)` and `refit(agent, dataset)`
   take the labelled examples as an argument. Downloading `fancyzhx/ag_news` belongs in a
   script or a `__main__` block, not in a function that a caller might reasonably run offline,
   in CI, or against their own proprietary evaluation set.

Nothing is re-exported at package level: `eval_ece` pulls in torch transitively (see above), so
importing this namespace must stay cheap. Import the module you want explicitly, e.g.
`from laya_onnx.bench.eval_ece import measure_ece`.
"""
