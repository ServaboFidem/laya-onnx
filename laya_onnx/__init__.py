"""Torch-free ONNX runtime for laya's multilingual (mmBERT-base) checkpoint.

Only `runtime` and `truncation` are imported here. `laya_onnx.export` is deliberately left
unimported at package level -- it is the one subpackage that may import torch (see its own
docstring), and importing it from here would drag torch into every process that merely does
`import laya_onnx`. tests/test_onnx_no_torch.py checks that boundary directly.
"""
from .runtime import OnnxAgent, load
from .truncation import state_token_count, truncation_report

__all__ = ["OnnxAgent", "load", "truncation_report", "state_token_count"]
