"""Export-time code. This subpackage is the only place in `laya_onnx` that may import torch.

Everything else under `laya_onnx/` is runtime: it runs against onnxruntime and numpy on a host
that need never have PyTorch installed. Keeping the torch dependency fenced into one package is
what makes that claim checkable rather than aspirational - `grep -rn "import torch" laya_onnx/`
should only ever hit files under this directory.

Nothing is re-exported at package level on purpose: importing `laya_onnx.export` would then drag
torch into any process that merely touched the namespace. Import the module you want explicitly,
e.g. `from laya_onnx.export.export_fp32 import export_fp32`.
"""
