"""OnnxAgent: the torch-free equivalent of laya.Agent, for the multilingual (mmBERT) checkpoint.

This module is the seam where Tasks 1-6 become one callable pipeline: TokenizerAdapter ->
build_sequence -> collate -> OnnxSession (or OpenVinoSession) -> build_answers, wrapped the same
way laya.agent.Agent wraps the torch path (laya/agent.py:266-370), so `OnnxAgent.system_one`
returns the same shaped answer dict for the same inputs.

Two things bind this module and are each covered by tests/test_onnx_runtime.py rather than
merely asserted in a comment:

1. `max_len`/`head_max_len` MUST come from the checkpoint's own `rl_agent_config.json`, never
   from `build_sequence`'s defaults (512/192, upstream's English-checkpoint numbers kept for
   byte-parity) and never from any other hardcoded fallback either. The multilingual checkpoint
   this port targets runs at 1024/256 -- a ~768-token state budget instead of ~317 -- and that
   budget is the whole reason this port exists on top of mmBERT rather than the English
   checkpoint. A fallback that leans toward those multilingual numbers is *still* wrong: point
   this loader at an english-checkpoint export (a 512-token graph) whose config happens to omit
   `max_len`, and a 1024 fallback would silently build a sequence the traced graph was never
   shaped for -- no error, just a wrong answer. So `__init__` requires both keys to be present
   and raises `ValueError` naming whichever is missing, the same way it already raises
   `FileNotFoundError` for a missing config file altogether; `system_one` reads
   `self.max_len`/`self.head_max_len`, set once at load time, with no `.get(..., default)`
   anywhere in the call path.

2. A question with fewer than 2 options must be rejected before it reaches the graph. The graph
   was traced with 3 markers (laya_onnx/export/export_fp32.py's `markers = 3` sample) specifically
   because laya/common.py:115 branches on `p.size(-1) >= 2` and only the true side calls
   `p.topk(2, -1)` -- a tracer resolves that Python `if` once, against the sample, and bakes
   whichever branch it took into the graph permanently. A single-option question would feed the
   traced (>=2) branch an input it structurally cannot handle and fail deep inside ONNX Runtime's
   TopK op with a shape error that says nothing about *why*. Reject it here instead, in terms
   the caller can act on.
"""
import json
import os
from typing import Any, Dict, Optional, Union

from .collate import collate
from .postprocess import QTYPES, build_answers
from .sequence import build_sequence, render_options
from .tokenizer import TokenizerAdapter
from .truncation import truncation_report


def _to_internal(qdef: Dict) -> Dict:
    """Mirror of laya/agent.py:255 (`Agent._to_internal`). Kept as a free function here since
    OnnxAgent has no torch-side counterpart to inherit it from."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


BACKENDS = ("onnxruntime", "openvino")


def _make_session(backend: str, model_path: str, threads: Optional[int]):
    """Build the session for `backend`, importing only that backend's runtime.

    The imports are deliberately here and not at module scope: a process that serves through
    OpenVINO need not have onnxruntime installed, and vice versa. Both session classes share one
    contract -- `run(batch) -> (logits, act_logits)` -- so nothing downstream knows which ran.
    """
    if backend == "onnxruntime":
        from .session import OnnxSession
        return OnnxSession(model_path, threads=threads)
    if backend == "openvino":
        from .session_openvino import OpenVinoSession
        return OpenVinoSession(model_path, threads=threads)
    raise ValueError("unknown backend %r; expected one of %s" % (backend, ", ".join(BACKENDS)))


class OnnxAgent:
    """Loads a laya-onnx checkpoint directory (model.onnx + rl_agent_config.json + tokenizer/)
    and answers typed questions against it without torch or transformers in the process.

    `backend` picks the runtime that executes the exported graph: "onnxruntime" (the default)
    or "openvino". Same export, same inputs, same postprocessing; see
    docs/superpowers/specs/2026-09-23-openvino-backend.md for why the default is unchanged.
    """

    def __init__(self, model_dir: str, threads: Optional[int] = None, backend: str = "onnxruntime"):
        # Checked before any file is read, so a typo fails on the argument, not on a side effect.
        if backend not in BACKENDS:
            raise ValueError("unknown backend %r; expected one of %s" % (backend, ", ".join(BACKENDS)))
        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                "Incompatible checkpoint: %r does not contain 'rl_agent_config.json'. That file "
                "ships with every laya-onnx export and carries the token budget "
                "(max_len/head_max_len) this runtime must use -- load a directory produced by "
                "laya_onnx.export.export_fp32, not an arbitrary ONNX file." % model_dir
            )
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        # The token budget must come from THIS checkpoint's own config, never a hardcoded
        # fallback. A fallback here is not conservative -- it is actively dangerous: point this
        # at an english-checkpoint export (512-token graph) whose config happens to omit
        # max_len, and a 1024-default would silently build a sequence the traced graph was
        # never shaped for, producing a wrong answer with no error anywhere. So a config that is
        # present but missing either key is exactly as incompatible as one that is absent
        # entirely (see the FileNotFoundError above), and raises the same way.
        missing = [k for k in ("max_len", "head_max_len") if k not in self.cfg]
        if missing:
            raise ValueError(
                "Incompatible checkpoint: %r's rl_agent_config.json is missing %s. Both "
                "max_len and head_max_len are required -- this runtime never assumes a token "
                "budget, since the wrong one (e.g. falling back toward the English "
                "checkpoint's 512/192 or vice versa) silently truncates the state without "
                "any error." % (model_dir, ", ".join(sorted(missing)))
            )
        self.max_len = self.cfg["max_len"]
        self.head_max_len = self.cfg["head_max_len"]

        self.tok = TokenizerAdapter(model_dir)
        self.backend = backend
        self.session = _make_session(backend, os.path.join(model_dir, "model.onnx"), threads)

        # `laya-multilingual` ships no fitted temperatures at all -- these defaults (identity
        # scaling) are the normal path for that checkpoint, not a fallback for a broken one.
        # See postprocess.py's module docstring: build_answers clamps whatever comes out of
        # these lookups on every call, so an absent/empty mapping here is safe by construction.
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})

    def system_one(self, state: Union[str, dict, list],
                    questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single forward pass. See
        laya.agent.Agent.system_one for the docstring this mirrors; the return shape matches
        `{"model", "answers", "usage"}`, with `usage["truncated"]` as this port's one deliberate
        addition -- the torch path never reports how much of a state it had to drop.
        """
        ids = list(questions.keys())
        # Requirement 1 (see module docstring): self.max_len/self.head_max_len were read from
        # this checkpoint's own config in __init__, with no fallback -- a config missing either
        # key already raised there, so there is nothing to default here.
        max_len = self.max_len
        head_max_len = self.head_max_len

        items, internal = [], {}
        for qid in ids:
            q = _to_internal(questions[qid])
            internal[qid] = q
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            # Requirement 2 (see module docstring): the exported graph was traced with >= 2
            # markers and cannot answer a single-option question at all.
            if len(markers) < 2:
                raise ValueError(
                    "question %r has %d option(s), but the exported ONNX graph was traced with "
                    "laya/common.py's topk(2) branch and requires at least 2 markers per "
                    "question. Give it at least two options, or answer it with a 'noul' "
                    "question instead." % (qid, len(markers))
                )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        batch = collate(items, self.tok.pad_token_id)
        logits, act = self.session.run(batch)
        answers = build_answers(ids, internal, items, logits, act,
                                 self.temperature, self.temperature_by_options)

        truncated = {
            qid: truncation_report(self.tok, state, item["ids"], max_len, internal[qid], head_max_len)
            for qid, item in zip(ids, items)
        }

        return {
            "model": "laya-onnx",
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
                "truncated": truncated,
            },
        }

    predict = system_one


def load(model_dir: str, threads: Optional[int] = None, backend: str = "onnxruntime") -> OnnxAgent:
    """Load a laya-onnx checkpoint directory. Mirrors laya.agent.load's shape but takes a local
    path only -- there is no Hub download path here; that is out of scope for this module
    (checkpoint acquisition is Task 8's job). `backend` is "onnxruntime" or "openvino"."""
    return OnnxAgent(model_dir, threads=threads, backend=backend)
