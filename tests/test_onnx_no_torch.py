"""The runtime path must not import torch or transformers.

Dropping torch (~2 GB) for onnxruntime (~50 MB) is most of the commodity-hardware win, and one
stray module-scope import silently gives it back. Two checks enforce this, at different points
in the failure:

1. A `sys.modules` scan after `import laya_onnx` -- catches a leak *after the fact*, by noticing
   torch or transformers ended up loaded somewhere in the process.
2. A `sys.meta_path` finder that raises ImportError the instant anything tries to import torch or
   transformers -- catches the leak *at the point of attempt*, with a traceback that names the
   importing module directly, rather than requiring you to go hunt through sys.modules for the
   culprit after the import already silently succeeded.

The second is strictly stronger: it also catches a `try: import torch / except ImportError:
pass` guard that (1) alone would miss, since torch would get removed from sys.modules on that
exception path in some import machinery, but the attempt -- and the ImportError it should never
need to except -- would still be real.

The finder implements `find_spec`, not `find_module`. `find_module` is the pre-3.4 finder
protocol; CPython no longer falls back to it when `find_spec` is absent (verified empirically on
3.13.9 -- a `find_module`-only finder on `sys.meta_path` does not block `import torch` at all, it
just imports normally). A blocker built on the wrong protocol is not a weaker guard, it is no
guard: `import/survives-torch-blocked` would pass regardless of whether laya_onnx ever touched
torch, because nothing was actually blocking anything. `blocker/actually-blocks` below exists so
that can never happen silently again -- it asserts the blocker itself raises on a bare
`import torch` before any laya_onnx import is allowed to "survive" it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("torch", "transformers", "laya", "laya_onnx")]:
    del sys.modules[mod]

import laya_onnx  # noqa: E402

leaked = sorted({m.split(".")[0] for m in sys.modules} & {"torch", "transformers", "laya"})
check("import/no-torch-or-transformers", not leaked, "leaked: %s" % leaked)
check("api/load-exists", hasattr(laya_onnx, "load"))
check("api/agent-exists", hasattr(laya_onnx, "OnnxAgent"))
check("api/truncation-exported", hasattr(laya_onnx, "truncation_report"))
check("api/predict-aliases-system_one",
      laya_onnx.OnnxAgent.predict is laya_onnx.OnnxAgent.system_one)


class _Blocker:
    """A meta_path finder that turns any attempted `import torch` / `import transformers` into
    an immediate, loudly-attributed ImportError, instead of letting it succeed and only noticing
    afterward via sys.modules.

    Implements the current (>=3.4) finder protocol, `find_spec`. A finder that only implements
    the legacy `find_module` is silently never consulted on modern CPython -- see the module
    docstring -- so this method name is load-bearing, not a style choice.
    """

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers"):
            raise ImportError("blocked: " + name)
        return None


for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("torch", "transformers", "laya", "laya_onnx")]:
    del sys.modules[mod]

sys.meta_path.insert(0, _Blocker())
try:
    # The blocker must be proven live BEFORE it is trusted to prove anything about laya_onnx.
    # A guard that cannot fail is worse than no guard, because it reads as proof: this is the
    # exact case that shipped inert (find_module, never called) while its own check still read
    # PASS, because nothing was attempting to import torch in the first place.
    try:
        import torch as _blocked_torch  # noqa: F401
        check("blocker/actually-blocks", False, "import torch succeeded; the blocker is inert")
    except ImportError as e:
        check("blocker/actually-blocks", "blocked: torch" in str(e), "raised %r instead" % (e,))

    import laya_onnx as laya_onnx_blocked          # noqa: E402
    import laya_onnx.collate                       # noqa: E402, F401
    import laya_onnx.postprocess                   # noqa: E402, F401
    import laya_onnx.sequence                      # noqa: E402, F401
    import laya_onnx.session                       # noqa: E402, F401
    import laya_onnx.tokenizer                     # noqa: E402, F401
    import laya_onnx.truncation                    # noqa: E402, F401
    check("import/survives-torch-blocked", True)
except ImportError as e:
    check("import/survives-torch-blocked", False, "raised %r" % (e,))
finally:
    sys.meta_path.remove(sys.meta_path[0])
check("api/load-exists-under-block", hasattr(laya_onnx_blocked, "load"))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all no-torch tests passed")
sys.exit(1 if FAIL else 0)
