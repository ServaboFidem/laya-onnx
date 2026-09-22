# Spec: the fused fp32 export, and sharing it on the Hugging Face Hub

**Status:** accepted, not yet implemented
**Date:** 2026-09-22
**Plan:** [`../plans/2026-09-22-fused-export.md`](../plans/2026-09-22-fused-export.md)
**Builds on:** [`2026-09-21-laya-onnx-port.md`](2026-09-21-laya-onnx-port.md)

## Goal

Make the fp32 ONNX export of the `multilingual` checkpoint run on the fused attention kernels
onnxruntime already ships, keep it answer-equivalent to the torch path under the existing
parity test, and package it as a directory anyone can load by Hub repo id from a process
holding onnxruntime and numpy and nothing else.

## Why: what was measured

The shipped export (opset 18, torch.export capture of ModernBERT) reaches onnxruntime as 1393
nodes of which almost nothing fuses: one `SkipLayerNormalization`, no GELU, no attention. A
4-thread profile puts `MatMul` at 56% of kernel time and the rotary/attention glue
(`Transpose`, `Softmax`, `Split`, `Slice`, `Neg`, `Concat`, `Mul`) at roughly another 27%.

Exporting at opset 23 emits the standard `Attention`, `RotaryEmbedding` and `Gelu` ops (775
nodes). Two onnxruntime CPU-kernel strictnesses then have to be worked around, on 1.27.0 and
still on 1.30.0: `Attention` rejects a mask whose query dimension is 1 (legal by broadcast; the
decision head uses it) and `RotaryEmbedding` rejects cos/sin caches whose batch dimension is 1
(legal by broadcast; every encoder layer uses it). Both are fixed with `Expand` nodes. A further
per-layer rewrite onto the 3D `[B, S, H*D]` form of both ops removes 44 `Transpose`, 66 `Slice`
and 66 `Squeeze` nodes.

The result, measured on the dual Xeon Gold 6148 host of `laya_onnx/README.md` (onnxruntime
1.27.0, 3 warmup / 20 timed runs, nearest-rank p50), with answers identical to the shipped graph
and raw logits within 4.864e-05 of torch across the full e2e suite (318/318):

| threads | questions | shipped | fused 3D | change | torch, same threads |
|---|---|---|---|---|---|
| 4 | 1 | 231.3 ms | 198.5 ms | -14% | 288.7 ms |
| 4 | 5 | 1177.7 ms | 1033.5 ms | -12% | 1187.6 ms |
| 4 | 10 | 2474.8 ms | 2119.4 ms | -14% | 2312.7 ms |
| default | 1 | 149.5 ms | 103.4 ms | -31% | 175.4 ms |
| default | 5 | 641.0 ms | 373.2 ms | -42% | 447.8 ms |
| default | 10 | 1226.3 ms | 648.9 ms | -47% | 1135.6 ms |

These numbers came from scratch scripts. This spec turns them into a checked-in, tested export
path and re-measures with the checked-in benches before any figure enters the docs.

## Scope

In:

1. `laya_onnx/export/fuse.py`: the fusion pass, a pure function over `onnx.ModelProto`.
2. `export_fp32` defaults to opset 23 and runs the pass; `--no-fuse` and `--opset 18` keep the
   old graph reachable. It writes `export_manifest.json` and copies the model card.
3. `laya_onnx.load()` accepts a Hub repo id, with `revision=` and `token=`.
4. `laya_onnx/hub/README.md`: the model card.
5. `tests/test_onnx_fuse.py`: a tiny ModernBERT from config, exported, fused and checked
   against torch in CI; `onnxscript` joins the ONNX CI step.
6. Docs: package README section, AGENTS.md, and the card's tables.

Out:

- int8 on the fused graph. The int8 study was on the opset-18 graph; its answer may change and
  is a separate study.
- Shipping `laya_onnx` in the wheel. Unchanged from the port spec.
- Any change under `laya/` or to the `laya/` test suites. Hard constraint, unchanged.
- A `push_hub` CLI. Upload is one documented `hf upload` command.

## Locked decisions

**D1. Fusion is a post-pass on the exporter's own output, not a hand-built graph.** The
exporter at opset 23 already emits the target ops; the pass only removes layout plumbing and
works around two kernel strictnesses. A hand-built encoder would arrive at the same ops and
would own 22 layers of weight plumbing for no further gain.

**D2. The pass matches on structure, never on node names.** The scratch version keyed on
exporter-generated names (`val_43`, `view_51`). Every shape the pass needs is derived with
`Shape` nodes from the tensors the matched pattern itself owns: a layer's cos/sin expansion
uses the shape of that layer's own q; the head mask expansion uses the shape of that
`Attention`'s own Q. Head count is read from the matched `RotaryEmbedding`'s `num_heads`
attribute.

**D3. Each layer keeps its own rotary cache.** ModernBERT uses a different rotary base for
global (every third) and local layers, so the graph carries two distinct cos/sin caches. The
first scratch rewrite shared one across all layers and produced wrong answers (max delta 0.28)
that the e2e test caught. The pass expands each layer's own cache and never dedupes them.

**D4. All-or-nothing.** The pass matches every candidate on a copy and commits only if every
`Attention` node in the graph was classified as either an encoder layer (rewritten) or a head
layer (mask expanded). An unrecognised shape raises `FusionError` naming the node. Zero
encoder matches is also `FusionError`: running the pass on an opset-18 graph or on an
already-fused one fails loudly rather than returning its input.

**D5. The exporter verifies the fused file, not the pre-fusion one.** `_verify_opset` and
`onnx.checker.check_model` run after the pass, on the file that will be shipped.

**D6. Opset 23 is the default; 18 stays reachable.** `--opset 18` produces the previous graph
(no fusion possible, none attempted). `--no-fuse` at opset 23 writes the unfused opset-23 graph,
which does not run on onnxruntime 1.27 through 1.30 because of the two strictnesses; the
exporter says so on stdout when that flag is used. That path exists for parity debugging, not
deployment.

**D7. The declared onnxruntime floor stays 1.30.** The opset-23 kernels exist earlier, but 1.30
is the release the fused graph is verified on. `Attention`-23 and `RotaryEmbedding`-23 kernels
are a hard requirement of the fused graph; the model card says so.

**D8. Hub loading uses `huggingface_hub` lazily.** It is already a core dependency of `laya`,
imports no torch, and is imported inside the Hub branch of `load()` only, so a purely local
deployment never touches it. The download uses `allow_patterns` for exactly the artifact's
files, as the torch path does.

**D9. A repo id is anything that is not an existing directory and matches `owner/name`.**
`load("./exports/x")` and `load("C:/models/x")` are paths; `load("user/laya-onnx")` is a repo.
A string that is neither raises `FileNotFoundError` naming both interpretations.

**D10. The model card is hand-written and its numbers are copied from the package README.**
The exporter copies it verbatim into the export directory. Nothing generates figures. The
manifest records provenance (versions, opset, fusion report, source path and revision when
known), which is what a reader needs to reproduce the artifact and what a card should not
paraphrase.

## Components

### `laya_onnx/export/fuse.py`

```
class FusionError(RuntimeError): ...

@dataclass
class FusionReport:
    encoder_layers: int          # Attention nodes rewritten to 3D
    head_attentions: int         # Attention nodes whose mask was expanded
    nodes_before: int
    nodes_after: int
    removed: Dict[str, int]      # op_type -> count removed, for the manifest and stdout

def fuse(model: onnx.ModelProto) -> FusionReport   # mutates `model`; raises FusionError
```

Encoder-layer pattern (all six upstream/downstream nodes must be present, single-consumer where
the rewrite deletes them):

```
qkv = MatMul(x, W_qkv)                       [B,S,3*H*D]
Reshape(qkv, [B,S,3,H,D]) -> Transpose(0,3,2,1,4) -> Slice(axis 2, i) -> Squeeze(axis 2)   x3
q -> RotaryEmbedding(q, cos, sin; num_heads=H)     cos/sin: [1,S,D/2]
k -> RotaryEmbedding(k, cos, sin; num_heads=H)
Attention(q_rot, k_rot, v, mask) -> Transpose(0,2,1,3) -> Reshape([B,S,H*D]) -> MatMul(W_o)
```

Replacement:

```
q3,k3,v3 = Split(qkv, axis=-1, num_outputs=3)
shape    = Concat(Shape(q3)[0:1], Shape(q3)[1:2], Shape(cos)[2:3])
cosB,sinB = Expand(cos, shape), Expand(sin, shape)
q3r = RotaryEmbedding(q3, cosB, sinB; num_heads=H)
k3r = RotaryEmbedding(k3, cosB, sinB; num_heads=H)
out = Attention(q3r, k3r, v3, mask; q_num_heads=H, kv_num_heads=H)   [B,S,H*D]
MatMul(out, W_o)
```

The `(3, H, D)` ordering of the projection's last axis is asserted from the matched `Reshape`'s
shape constant before `Split` is trusted to yield q, k, v in that order.

Head pattern: an `Attention` whose q and k are not produced by `RotaryEmbedding`, with a
4-D mask input. Replacement: `mask' = Expand(mask, Concat(Shape(mask)[0:2], Shape(Q)[2:3],
Shape(mask)[3:4]))`. Nothing else in the head changes.

### `laya_onnx/export/export_fp32.py`

- `export_fp32(model, out_path, opset=23, fuse=True) -> ExportResult` where the result carries
  the path and the `FusionReport` or `None`.
- Fusion runs only when `opset >= 23 and fuse`; a request for `fuse=True` with `opset < 23` is a
  `ValueError` at the top of the function, not a silent skip.
- After fusion the model is re-saved with external data to the same paths, then verified.
- `main()` writes `export_manifest.json`:

```json
{
  "source": {"model_dir": "...", "hub_repo": "convaiinnovations/laya", "hub_revision": "1c5edc1...", "subfolder": "multilingual"},
  "opset": 23,
  "fusion": {"encoder_layers": 22, "head_attentions": 2, "nodes_before": 775, "nodes_after": 667, "removed": {"Transpose": 44, "Slice": 66, "Squeeze": 66, "Reshape": 44}},
  "versions": {"torch": "2.10.0+cpu", "transformers": "4.57.3", "onnx": "1.23.0", "onnxscript": "0.7.2", "onnxruntime": "1.30.0"}
}
```

  Hub repo and revision are filled when `--model-dir` lies inside a `huggingface_hub` snapshot
  cache path (`models--owner--name/snapshots/<rev>/`); otherwise the keys are absent, never
  guessed. `fusion` is `null` when the pass did not run.
- `main()` copies `laya_onnx/hub/README.md` to `<out>/README.md`.
- stdout gains one line per fusion: `fused: 22 encoder layers, 2 head masks; 775 -> 667 nodes`.

### `laya_onnx/runtime.py`

```
def load(model_dir_or_repo: str, threads: Optional[int] = None,
         revision: Optional[str] = None, token: Optional[str] = None) -> OnnxAgent
```

Hub branch: `snapshot_download(repo_id, revision=revision, token=token,
allow_patterns=["model.onnx", "model.onnx.data", "rl_agent_config.json", "tokenizer/*",
"export_manifest.json"])`, then the existing local constructor on the returned path. A
download that yields no `model.onnx` raises `FileNotFoundError` naming the repo and the
patterns, so a repo that holds only the torch checkpoint fails in words.

### `laya_onnx/hub/README.md`

Front matter: `license: apache-2.0`, `base_model: convaiinnovations/laya`, no `library_name`
(there is no Hub library integration), tags `onnx`, `onnxruntime`, `text-classification`,
`modernbert`, `multilingual`. Body: what it is (one paragraph, linking the source repo and this
fork), load snippet by repo id, the file table with the two-file warning, the requirements
(onnxruntime >= 1.30, tokenizers, numpy), the measured tables above with the host block and
the scoping sentence, limits (confidence not calibrated; at least 2 options per question;
English suites only for any accuracy number cited; token budget 1024/256), and the reproduce
block (export command, e2e test, both benches).

## Testing

### `tests/test_onnx_fuse.py` (CI)

Builds a `DecisionModel` around a two-layer `ModernBertConfig` (hidden 64, 4 heads,
intermediate 128, `global_attn_every_n_layers=2`, `local_attention=8`, vocab 512), random
weights, `attn_implementation="sdpa"`, `reference_compile=False`. Exports at opset 23 with the
real `export_fp32`, then:

1. `fuse()` reports `encoder_layers == 2`, `head_attentions == head_layers`, and the fused
   graph has zero `Transpose` nodes between any `Attention` and its output projection.
2. ORT (fused) vs torch logits and act_logits, `max|d| < 1e-4`, at shapes: `(1, 6, 2)`,
   `(3, 20, 4)` with mixed padding, `(2, 40, 3)` where 40 > local window so the sliding mask is
   exercised, and a markers count above the traced sample's.
3. ORT (fused) vs ORT (unfused opset-23 with only the two Expand workarounds applied) at the
   same shapes, `max|d| < 1e-5`: the rewrite is a layout change and must be tighter than the
   torch comparison.
4. `fuse()` on the opset-18 export of the same model raises `FusionError` (zero matches).
5. `fuse()` on an already-fused graph raises `FusionError`.
6. A graph where one layer's `Transpose` perm is altered raises `FusionError` naming that node
   and leaves the input model unmodified (all-or-nothing).
7. `export_fp32(..., opset=18, fuse=True)` raises `ValueError`.

This test imports torch, transformers, onnx, onnxscript and onnxruntime; it runs in the ONNX
CI step, whose pip line gains `onnxscript`.

### `tests/test_onnx_runtime.py` (CI, extended)

- `load("owner/name")` with `snapshot_download` monkeypatched: called once with the exact
  `allow_patterns`, `revision` and `token` forwarded, and the returned path is what the agent
  loads.
- `load("./not/a/dir")` raises `FileNotFoundError` naming both interpretations.
- `load("owner/name")` whose mocked download returns a directory without `model.onnx` raises
  `FileNotFoundError` naming the repo.
- `tests/test_onnx_no_torch.py` unchanged and must stay green: `huggingface_hub` is imported
  lazily and imports no torch.

### By hand, before numbers enter the docs

- `tests/test_onnx_local_e2e.py` against the exporter's real output (not a scratch graph).
- `bench_latency` at `--threads 4` and default; `bench_torch` at `--threads 4` and default; both
  with the README's 5-warmup / 50-run protocol so the new tables match the old ones' method.

## Success criteria

1. `python -m laya_onnx.export.export_fp32 --model-dir <snapshot>/multilingual --out <dir>`
   with no other flags writes a fused opset-23 artifact, a manifest and a card, and prints the
   fusion line.
2. The e2e test passes on that artifact at the existing thresholds.
3. All CI suites pass, including the new one, on onnxruntime 1.30.0.
4. `laya_onnx.load("<owner>/<repo>")` works from a process without torch after `hf upload`.
5. Every latency figure in the README, AGENTS.md and the card was produced by a checked-in
   script, and says which host and protocol produced it.

## Risks

- **The pattern is torch-version-specific.** A future torch may capture ModernBERT differently
  and the pass will refuse (D4). That is the intended failure; the CI test detects it the day
  the pinned torch moves.
- **`Attention`-23 CPU kernel maturity.** It matched the pieces it replaced at 4 threads and
  beat them at default threads; the gain is layout removal plus the kernel, and the split
  between the two is not measured. If a later onnxruntime speeds the kernel up, the fused graph
  benefits without change.
- **Two-file artifact on the Hub.** `model.onnx.data` is 1.29 GB and must be uploaded with the
  graph; the card and the loader's `allow_patterns` both name it.
