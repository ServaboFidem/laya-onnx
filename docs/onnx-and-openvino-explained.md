# laya on ONNX and OpenVINO, in plain English

**Short version.** We took laya's multilingual model out of PyTorch and saved it as an ONNX file:
a fixed, self-contained recipe that other software can run. That made the model lighter to
deploy, but not much faster on its own. The speed came from the next step: running that same
file through **OpenVINO**, Intel's compiler for this kind of model, which reorganizes the work
to suit the CPU in front of it. On the test machine, answers now come back **about 2–2.8× faster
than onnxruntime** (the ONNX engine we started with), and the answers themselves have not
changed.

This page explains what each step does and why the answers stay the same. For the one-picture
version, open [`stock-onnx-openvino.html`](stock-onnx-openvino.html) in a browser. The technical record,
with every table and caveat, is [`laya_onnx/README.md`](../laya_onnx/README.md). Every number
here comes from there or was measured the same way on the same machine (see
[Where the numbers come from](#where-the-numbers-come-from)).

> Scope: all of this applies to laya's **multilingual** checkpoint. The English and
> typed-decisions checkpoints have not been exported to ONNX yet and still run on PyTorch only.

---

## Where we started: laya on PyTorch

laya is a *decision model*: you give it some text (an email, a support ticket, a JSON record)
and a few typed questions ("which team owns this?", "is the customer angry, 0–4?"), and it
answers all of them in one pass, with a probability for each option.

Stock laya runs inside **PyTorch**, the framework the model was trained in. PyTorch is excellent
for building and training models, and it is a heavy thing to ship just to *use* one:

- It is big. On the test machine the CPU-only PyTorch install is **462 MB**, and the GPU build is
  **4.2 GB**.
- It builds the model from Python code each time it starts, together with the `transformers`
  library that defines the model's layers.
- It runs the model one operation at a time, as the Python code describes it. It does not look
  at the whole model ahead of time to find a faster way to do the same work.

## Step 1: converting to ONNX, or writing the recipe down

**ONNX** (Open Neural Network Exchange) is a standard file format for trained models. Converting
to ONNX means running the model once and recording exactly what it computes, step by step, into
a file. It is like turning a chef's working kitchen into a printed recipe: the chef and the
kitchen are no longer needed, and any cook who can read the format can make the dish.

For laya the recipe is two files: `model.onnx` (2.8 MB, the list of steps) and `model.onnx.data`
(1.29 GB, the trained numbers the steps use). They always travel together.

What that bought us:

| Benefit | What it means in practice |
|---|---|
| **No PyTorch at run time** | The serving program needs only a runtime engine, numpy and a tokenizer. A test fails the build if PyTorch or `transformers` ever sneaks back in. |
| **Much lighter to install** | The onnxruntime engine is **42 MB** on disk and OpenVINO is **245 MB**, compared with 462 MB for CPU-only PyTorch or 4.2 GB for GPU PyTorch. |
| **A frozen, inspectable model** | The file is exactly what gets run. Nobody's Python version or library update can quietly change the computation. |
| **A choice of engines** | Any ONNX-compatible runtime can execute the same file. This is what made Step 2 possible without retraining or re-exporting anything. |
| **Same answers** | Compared with PyTorch on the real weights, the largest difference in the model's raw scores is **0.00003719**, far too small to change any answer. |
| **One new feature** | The ONNX version reports when a long input had to be cut to fit the model's size limit. PyTorch laya cuts silently. |

**What converting did *not* do: make it much faster.** Run by onnxruntime, the ONNX model took
164.5 ms for one question against PyTorch's 175.4 ms. It was *slower* than PyTorch on large
batches at default settings, and about even when both were limited to 4 CPU threads. The recipe
was the same work written down; nothing had been reorganized yet.

We also tried shrinking the model's numbers from 32-bit to 8-bit ("int8" quantization), a
common speed trick. It was only 4–23% faster, and it measurably hurt the answers: on a yes/no
benchmark, accuracy fell from 88% to 78%. We kept full 32-bit precision.

## Step 2: compiling with OpenVINO, or tailoring the recipe to the kitchen

onnxruntime follows the recipe fairly literally. **OpenVINO** reads the *whole* recipe first,
the way a compiler reads a whole program, then rewrites it into a plan tuned for the specific CPU
it is running on. It does this once, when the model loads, and every request after that uses
the tuned plan.

We inspected what OpenVINO actually builds from laya's file. The main changes:

**1. It merges steps that always happen together.** The big matrix multiplications are the core of
the work and cannot be skipped; in a profiled onnxruntime call they took 56% of the time. Another
27% or so went to hundreds of small steps in between, mostly the rotary and attention steps
described below. When small steps run one after another, a literal runner writes each
intermediate result out to memory, reads it back for the next step, and pays a start-up cost for
every step. A merged ("fused") step does the whole chain in one go, keeping the intermediate
results in the CPU's fast cache.

- **Attention**, the part of the model that lets each word look at every other word, is four
  steps per layer: multiply, mask, softmax, multiply. OpenVINO fused all four into **one piece of
  generated code in each of the model's 24 attention layers**.
- **Word-position math** (the "rotary" encoding that tells the model where each word sits) is a
  chain of small slicing, flipping and multiplying steps in the file. OpenVINO recognized that
  pattern and replaced it with **a single dedicated operation, in 44 places** (queries and keys in
  each of the 22 encoder layers).
- Some of the big multiplications are merged with the activation function that follows them.

For contrast, onnxruntime fused almost nothing in this same file: one layer-normalization step
and no attention.

**2. It pre-arranges the trained numbers.** The model's 100 largest multiplications use fixed
trained weights. OpenVINO prepares those weights **once**, in the memory layout the CPU's vector
units read most efficiently, and reuses them for every request instead of reshuffling each time.

**3. It writes code for *this* processor.** The test machine's Xeon CPUs support AVX-512, which
processes 16 numbers per instruction. OpenVINO generates AVX-512 code for **289 of the compiled
model's operations**, rather than using one-size-fits-all code.

**4. It organizes the CPU threads for one request at a time.** laya answers one request and waits
for it, so OpenVINO is told to optimize for *latency*. On this two-socket machine it keeps its
work on one processor's 20 physical cores, rather than spreading across both processors and
paying for the traffic between them.

## Why the answers did not change

Faster is only useful if the answers are the same. Four things ensure they are:

- **The model file is untouched.** OpenVINO loads the same `model.onnx` that onnxruntime does.
  Nothing is retrained, re-exported or converted on disk. Everything before and after the model
  (reading the text, splitting it into tokens, turning scores into answers) is the same code for
  both engines.
- **Full precision is locked in.** On newer Intel processors OpenVINO would, by default, quietly
  switch to a lower-precision number format (bf16) to go faster. laya forces 32-bit precision and
  checks after loading that OpenVINO obeyed. If it didn't, laya refuses to run rather than serve a
  subtly different model.
- **It is checked against the original.** On the real weights, the same 318-point comparison
  used for onnxruntime passes in full under OpenVINO. The largest difference in raw scores against
  PyTorch is **0.00004005**, and the reported probabilities differ by at most 0.0001, one unit in
  the last decimal place laya returns. A smaller version of that comparison runs automatically in
  CI on every push.
- **It is safe under load.** Each thread that calls the model gets its own OpenVINO workspace.
  We confirmed that sharing one between threads fails ("Infer Request is busy"), so this is a real
  guard, not a precaution on paper.

## The numbers

Time to answer, multilingual model, same machine, same test inputs. "Default threads" lets each
engine choose how many CPU threads to use:

| Questions per call | onnxruntime | OpenVINO | Speed-up |
|---|---|---|---|
| 1 | 152.4 ms | 73.9 ms | **2.06×** |
| 5 | 671.5 ms | 255.6 ms | **2.63×** |
| 10 | 1262.1 ms | 486.3 ms | **2.60×** |
| 50 | 6749.2 ms | 2378.9 ms | **2.84×** |

**Read the fine print.** Some of that gain comes from point 4 above: OpenVINO uses this big
two-processor machine more sensibly than onnxruntime does by default. When both engines are
limited to the same 4 threads, closer to a small cloud container, the gain is **1.05× for one
question**, which is effectively a tie, rising to **1.46× for 50 questions**. So:

- **Big multi-core server:** expect roughly half the wait time, or better.
- **Small container or modest machine:** expect about the same speed for single questions, and a
  worthwhile gain when you ask many questions at once.
- **Laptop, or Linux on this machine:** not measured yet.

Against stock PyTorch on the same machine, measured with the repo's benchmark scripts but not in
the same sitting, OpenVINO answers one question in 73.9 ms against PyTorch's 175.4 ms (about
2.4×). For 50 questions it is 2378.9 ms against 3689.5 ms (about 1.55×).

## What this does not do (yet)

- **The second processor sits idle.** Windows gives each program one "processor group", which on
  this machine is one of the two CPUs. Using both means running two copies of laya, one per
  processor, or running on Linux. Neither has been set up or measured.
- **Only the multilingual model is converted.** The English and typed-decisions checkpoints still
  run on PyTorch.
- **No GPU numbers.** This work is CPU-only. GPU acceleration is a separate path and has not been
  measured here.
- **Confidence is still uncalibrated.** Faster, identical answers inherit the original model's
  over-confidence. The multilingual checkpoint shipped with no fitted temperatures, so treat its
  confidence scores with care.

## Where the numbers come from

| Claim | Source |
|---|---|
| Latency tables (both engines, default and 4 threads) | `laya_onnx/README.md`, "OpenVINO backend": `laya_onnx/bench/bench_latency.py`, 3 warm-up plus 20 timed runs, median |
| PyTorch latency | `laya_onnx/README.md`, "Same machine, torch vs. ONNX": `laya_onnx/bench/bench_torch.py` |
| Answer parity (0.00003719, 0.00004005, 318 checks) | `tests/test_onnx_local_e2e.py` (real weights; `LAYA_ONNX_BACKEND=openvino` for OpenVINO) |
| int8 accuracy loss | `laya_onnx/README.md`, "int8 quantization" |
| What onnxruntime fuses, and its 56% / 27% time split | `laya_onnx/README.md` latency profile; `docs/superpowers/specs/2026-09-22-fused-export-design.md`, "Why" |
| What OpenVINO fuses (24 attention blocks, 44 rotary, 100 pre-arranged multiplications, 289 AVX-512 operations, 20 threads) | Inspected from OpenVINO's compiled model; reproduce with the snippet below |
| Install sizes (42 MB, 245 MB, 462 MB, 4.2 GB) | On-disk size of each package in a Windows Python 3.13 environment: onnxruntime 1.30.0, OpenVINO 2026.4.0, torch 2.10.0+cpu, torch 2.11.0+cu128 |

Test machine: two Intel Xeon Gold 6148 CPUs (20 cores each, AVX-512), Windows 11,
OpenVINO 2026.4.0, onnxruntime 1.30.0.

To see what OpenVINO compiled from the export yourself (run it from the export's directory):

```python
import collections, openvino as ov

core = ov.Core()
compiled = core.compile_model(
    core.read_model("model.onnx"), "CPU",
    {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"})
ops = compiled.get_runtime_model().get_ops()
info = lambda op, key: op.get_rt_info()[key].astype(str)
print(collections.Counter(info(op, "layerType") for op in ops))      # Subgraph, RoPE, FullyConnected, ...
print(sum(info(op, "primitiveType").startswith("jit_avx512") for op in ops), "ops use AVX-512 code")
print(compiled.get_property("INFERENCE_NUM_THREADS"), "threads")
del ops  # release the inspected graph first; without this the script can crash on exit
```
