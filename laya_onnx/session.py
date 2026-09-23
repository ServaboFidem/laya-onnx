"""onnxruntime session wrapper. Runtime code: numpy and onnxruntime only, never torch.

Thread count is the caller's choice. onnxruntime defaults `intra_op_num_threads` to every
physical core, which oversubscribes badly when the host already runs several workers - each
worker then spawns a full-width thread pool and they spend their time descheduling each other.
Leaving `threads=None` keeps onnxruntime's default for the single-process case; passing a small
number is the right move under a process pool or a container with a CPU quota.

The fixed `_INPUTS` tuple is deliberate. `collate()` emits exactly these five arrays with
exactly the dtypes the exported graph declares (int64, except the bool `marker_mask`), so the
feed is built by selection rather than by passing the dict through: an extra key would be a
sign the two sides have drifted, and onnxruntime would reject it at a less obvious moment.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort

# `declared_external_data` is re-exported: it lived here before the OpenVINO backend needed the
# same check without onnxruntime, and tests/test_onnx_export.py imports it from this module.
from .external_data import declared_external_data, require_external_data  # noqa: F401

_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
_OUTPUTS = ["logits", "act_logits"]


class OnnxSession:
    """A loaded fp32 ONNX decision model, run on CPU."""

    def __init__(self, model_path: str, threads: Optional[int] = None):
        require_external_data(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None:
            opts.intra_op_num_threads = int(threads)
        self._sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def run(self, batch: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run one collated batch. Returns `(logits, act_logits)` as float32.

        `np.asarray(..., dtype=np.float32)` is a no-op copy-free cast for the graph's own fp32
        outputs; it is here so the return type stays float32 if a later quantized export ever
        hands back something else, rather than letting the dtype leak into postprocessing.
        """
        logits, act = self._sess.run(_OUTPUTS, {k: batch[k] for k in _INPUTS})
        return np.asarray(logits, dtype=np.float32), np.asarray(act, dtype=np.float32)
