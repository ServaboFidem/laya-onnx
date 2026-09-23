# Spec: an OpenVINO backend for `laya_onnx`

**Status:** implemented (see `laya_onnx/README.md` § OpenVINO backend for the measured figures)
**Date:** 2026-09-23
**Builds on:** [`2026-09-21-laya-onnx-port.md`](2026-09-21-laya-onnx-port.md)
**Branch:** `feat/onnx-openvino` (from `feat/laya-onnx-port`)

## Goal

Run the *existing* fp32 ONNX export under OpenVINO's CPU runtime as an alternative to
onnxruntime, selected per agent, with answers equivalent to the onnxruntime backend and to the
torch path. The export, tokenizer, sequence builder, collate and postprocessing are unchanged;
only the object that turns a collated batch into `(logits, act_logits)` is swapped.

## Why: what was measured

Exploratory runs on the dual Xeon Gold 6148 host of `laya_onnx/README.md` (Skylake-SP: AVX-512,
no VNNI, no AVX512_BF16), Windows 11, onnxruntime 1.30.0, OpenVINO 2026.4.0, the shipped
multilingual export loaded unmodified by both runtimes, default threads, `PERFORMANCE_HINT =
LATENCY`, 2 warmup / 6 timed, median:

| state | questions | onnxruntime | OpenVINO | speedup | max \|Δlogit\| |
|---|---|---|---|---|---|
| short (~260 tok) | 1 | 136.3 ms | 73.5 ms | 1.86x | 5.7e-06 |
| short | 6 | 720.8 ms | 301.3 ms | 2.39x | 1.0e-05 |
| short | 24 | 3055.8 ms | 1080.1 ms | 2.83x | 1.0e-05 |
| long (1024 tok) | 1 | 700.1 ms | 280.0 ms | 2.50x | 2.0e-05 |
| long | 6 | 4334.7 ms | 1405.7 ms | 3.08x | 3.2e-05 |
| long | 24 | 18906.0 ms | 5812.0 ms | 3.25x | 3.2e-05 |

These are scratch measurements with a short protocol and are **not** the figures the README
will publish; the implementation re-measures with `laya_onnx/bench/bench_latency.py`'s own
protocol and only those numbers go into `laya_onnx/README.md`. What the table establishes is
that the gap is large enough to be worth a backend, and that the arithmetic is the same graph's
arithmetic (Δlogit at fp32 reordering scale, top-1 agreement 100%).

## Design

1. **`laya_onnx/session_openvino.py` — `OpenVinoSession(model_path, threads=None)`**, with the
   exact `run(batch) -> (logits, act_logits)` contract of `OnnxSession`: same five inputs selected
   by name, float32 outputs fetched by name. numpy and `openvino` only.
2. **fp32 is pinned, not defaulted.** OpenVINO's CPU plugin picks bf16 inference precision on its
   own on hardware with AMX or AVX512_BF16 (Sapphire Rapids onwards). The measured host has
   neither, so the default and fp32 coincide *here* — and would silently diverge on a newer
   Xeon. `INFERENCE_PRECISION_HINT = f32` is set explicitly and asserted after compile, because
   bf16 is a different model as far as the calibration study is concerned.
3. **`PERFORMANCE_HINT = LATENCY`.** One request at a time is how `OnnxAgent` is called. `threads`
   maps to `INFERENCE_NUM_THREADS`, the counterpart of onnxruntime's `intra_op_num_threads`.
4. **One infer request per thread.** `CompiledModel.__call__` reuses a single internal
   `InferRequest`, which is not safe to share between threads; onnxruntime's `run` is. Requests
   are held in a `threading.local`, so concurrent `predict` calls on one agent stay correct
   without a lock that would serialise them.
5. **The sidecar check is shared, not duplicated.** The protobuf walker behind
   `declared_external_data` moves to `laya_onnx/external_data.py`, together with the
   missing-sidecar `FileNotFoundError`, so the OpenVINO backend runs the same check without
   importing onnxruntime. `laya_onnx.session` re-exports `declared_external_data`, so existing
   imports keep working.
6. **Selection: `OnnxAgent(model_dir, threads=None, backend="onnxruntime")`**, and the same
   keyword on `laya_onnx.load`. The backend module is imported inside `__init__`, so a process
   holding only one of the two runtimes can use that one. An unknown name raises `ValueError`
   listing the valid ones. `agent.backend` records the choice.
7. **The default stays `onnxruntime`.** Flipping it is a one-line change, but a separate
   decision: it changes what every existing caller runs, and it should follow a recorded
   real-weights parity run under OpenVINO rather than be bundled with the code that makes one
   possible.
8. **A new `onnx-openvino` extra** (`openvino`, `tokenizers`, `numpy`), carrying the same
   "checkout-only, not in the wheel" caveat as the `onnx` extra above it.

## Tests

- `tests/test_onnx_openvino.py` (offline, in both `ci.yml` and `release.yml`): on the tiny
  exported fixture, OpenVINO vs onnxruntime logits at shapes the trace never saw (batch, sequence
  and marker axes each moved independently), identical answer dicts end to end, pinned f32,
  honoured `threads`, the missing-sidecar error, the unknown-backend error, and concurrent calls
  from several threads returning what a serial call returns.
- `tests/test_onnx_no_torch.py` imports `laya_onnx.session_openvino` under the torch blocker.
- `tests/test_onnx_local_e2e.py` honours `LAYA_ONNX_BACKEND`, so the real-weights parity run
  (not in CI) can be repeated against OpenVINO by hand.
- `laya_onnx/bench/bench_latency.py` gains `--backend`.

## Out of scope

- Using the second socket. On Windows a process sees one processor group (40 of 80 logical
  CPUs here); OpenVINO's threads stayed on socket 0 even when the process was started in group 1,
  and multi-stream / `TENSOR_PARALLEL` modes gained 0–26% at best. Linux has no processor groups
  and is where that belongs to be measured.
- Saving OpenVINO IR (`.xml`/`.bin`), multi-stream throughput mode, the OpenVINO GPU plugin,
  int8/bf16 under OpenVINO, an English-checkpoint export, and changing the default backend.
