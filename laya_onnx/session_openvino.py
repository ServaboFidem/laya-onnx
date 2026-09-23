"""OpenVINO session wrapper: the same contract as `session.OnnxSession`, on OpenVINO's CPU plugin.

Runtime code: numpy and openvino only -- never torch, and not onnxruntime either, so a process
that chose this backend does not also need the other runtime installed. It reads the very same
fp32 export (`model.onnx` + `model.onnx.data`); nothing is re-exported or converted on disk.

Why a second backend at all: on the dual-socket Skylake-SP host `laya_onnx/README.md` is
measured on, OpenVINO ran the unmodified export 2.1-2.8x faster than onnxruntime at each
runtime's default thread count, and 1.05-1.46x faster with both pinned to 4 threads (1 to 50
questions per call), at the same fp32 arithmetic. The default-thread gap is partly thread
placement on a two-socket box rather than kernels; the README's "OpenVINO backend" section has
both tables and says which is which.

Three settings are pinned rather than left to OpenVINO, each for a stated reason:

- `INFERENCE_PRECISION_HINT = f32`. The CPU plugin chooses bf16 by itself on hardware with AMX or
  AVX512_BF16 (Sapphire Rapids onwards). On the measured host the default and f32 coincide, which
  is exactly why it must be explicit: the same code on a newer Xeon would otherwise run a
  different-precision model than the one the parity and calibration numbers describe, with no
  error anywhere. `__init__` also reads the property back after compiling and refuses to run if
  the plugin did not honour it.
- `PERFORMANCE_HINT = LATENCY`. `OnnxAgent` submits one batch at a time and waits for it, which
  is the latency case; the throughput hint splits the cores into streams that a single request
  cannot use.
- `threads` maps to `INFERENCE_NUM_THREADS`, the counterpart of onnxruntime's
  `intra_op_num_threads`, with the same advice as `session.py`: leave it `None` for one process
  on a host, pass a small number under a process pool or a CPU quota. On a multi-socket Windows
  host OpenVINO's default is one socket's worth of logical CPUs (one processor group); it was
  not observed to use the second socket from a single process.

Thread safety. `CompiledModel.__call__` runs every call through one internal `InferRequest`,
which must not be used by two threads at once; onnxruntime's `InferenceSession.run` may be.
`OnnxAgent` is called concurrently wherever `laya.Router` shares an agent (the #95 design this
fork keeps), so each thread gets its own request from a `threading.local`. That keeps concurrent
calls correct without a lock that would serialise them.
"""
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import openvino as ov

from .external_data import require_external_data

# The same five inputs and two outputs as session.py, selected by name for the same reason: an
# extra or missing key means collate() and the exported graph have drifted apart.
_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
_OUTPUTS = ("logits", "act_logits")


class OpenVinoSession:
    """A loaded fp32 ONNX decision model, run on OpenVINO's CPU plugin."""

    def __init__(self, model_path: str, threads: Optional[int] = None):
        # Same check, same message as OnnxSession: OpenVINO's own report of a missing sidecar is
        # a read failure on a path the caller never named.
        require_external_data(model_path)
        config: Dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
        if threads is not None:
            config["INFERENCE_NUM_THREADS"] = int(threads)
        core = ov.Core()
        self._compiled = core.compile_model(core.read_model(model_path), "CPU", config)

        precision = self._compiled.get_property("INFERENCE_PRECISION_HINT")
        if precision != ov.Type.f32:
            raise RuntimeError(
                "OpenVINO compiled %r at %s inference precision although f32 was requested. The "
                "export's parity and calibration figures are fp32 figures; refusing to serve a "
                "different-precision model under them." % (model_path, precision))
        self.threads = int(self._compiled.get_property("INFERENCE_NUM_THREADS"))
        self._outputs = [self._compiled.output(name) for name in _OUTPUTS]
        self._local = threading.local()

    def _request(self) -> "ov.InferRequest":
        req = getattr(self._local, "request", None)
        if req is None:
            req = self._local.request = self._compiled.create_infer_request()
        return req

    def run(self, batch: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run one collated batch. Returns `(logits, act_logits)` as float32.

        The outputs are copied out of the request's tensors: those buffers belong to the request
        and are overwritten by this thread's next call, so handing out views would let a caller
        holding one answer see it change under them.
        """
        req = self._request()
        req.infer({k: batch[k] for k in _INPUTS})
        logits, act = (np.array(req.get_tensor(o).data, dtype=np.float32, copy=True) for o in self._outputs)
        return logits, act
