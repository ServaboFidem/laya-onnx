"""The runtime path must not import torch or transformers.

Dropping torch (~2 GB) for onnxruntime (~50 MB) is most of the commodity-hardware win, and one
stray module-scope import silently gives it back. Two checks enforce this, at different points
in the failure:

1. A `sys.modules` scan after `import laya_onnx` -- catches a leak *after the fact*, by noticing
   torch or transformers ended up loaded somewhere in the process.
2. A `sys.meta_path` finder that intercepts any attempt to import torch or transformers --
   catches the leak *at the point of attempt*, with a traceback that names the importing module
   directly, rather than requiring you to go hunt through sys.modules for the culprit after the
   import already silently succeeded.

**What (2) adds over (1), precisely.** A module that does `try: import torch / except
ImportError: pass` leaves nothing behind in sys.modules, so (1) cannot see it at all. An earlier
version of this file claimed the finder caught that case; it did not. The finder raised
`ImportError`, which is indistinguishable from torch being absent, so the guard swallowed it,
the import block completed, and the check recorded PASS -- a stronger guarantee asserted in
prose than the mechanism delivered. Two changes make the claim true:

- the finder *records* every intercepted module name, and the test asserts that list is empty
  after importing laya_onnx. A swallowed exception cannot erase the record;
- what it raises is `_ImportAttempted`, which deliberately does **not** subclass `ImportError`,
  so an `except ImportError` guard cannot catch it and an attempted import fails loudly instead.

`blocker/catches-guarded-import` below proves both against a real guarded import, so the
docstring above is a description of a tested mechanism rather than a claim about one.

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


class _ImportAttempted(Exception):
    """Raised by `_Blocker` when torch or transformers is asked for.

    Deliberately **not** an `ImportError`. An ImportError here is indistinguishable from torch
    simply being absent, so `try: import torch / except ImportError: pass` -- the exact pattern
    an optional-dependency shim uses, and the one most likely to smuggle torch back into the
    runtime path -- would swallow it and the test would pass. A plain Exception cannot be
    caught by that guard, so the attempt surfaces.
    """


class _Blocker:
    """A meta_path finder that intercepts any attempted `import torch` / `import transformers`,
    records it, and raises `_ImportAttempted`, instead of letting it succeed and only noticing
    afterward via sys.modules.

    `attempts` is the part that does not depend on how the importing module handles exceptions.
    Whatever a caller catches, the name is already in the list, so a swallowed failure is still
    visible to the assertion at the end.

    Implements the current (>=3.4) finder protocol, `find_spec`. A finder that only implements
    the legacy `find_module` is silently never consulted on modern CPython -- see the module
    docstring -- so this method name is load-bearing, not a style choice.
    """

    def __init__(self):
        self.attempts = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers"):
            self.attempts.append(name)
            raise _ImportAttempted("blocked: " + name)
        return None


for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("torch", "transformers", "laya", "laya_onnx")]:
    del sys.modules[mod]

blocker = _Blocker()
sys.meta_path.insert(0, blocker)
try:
    # The blocker must be proven live BEFORE it is trusted to prove anything about laya_onnx.
    # A guard that cannot fail is worse than no guard, because it reads as proof: this is the
    # exact case that shipped inert (find_module, never called) while its own check still read
    # PASS, because nothing was attempting to import torch in the first place.
    try:
        import torch as _blocked_torch  # noqa: F401
        check("blocker/actually-blocks", False, "import torch succeeded; the blocker is inert")
    except _ImportAttempted as e:
        check("blocker/actually-blocks", "blocked: torch" in str(e), "raised %r instead" % (e,))
    check("blocker/records-the-attempt", blocker.attempts == ["torch"],
          "recorded %r" % (blocker.attempts,))

    # The case the old docstring claimed and the old mechanism missed: a module that guards its
    # torch import. Under the previous ImportError-raising blocker this snippet completed
    # silently and the suite reported PASS. Both halves of the fix are asserted here -- the
    # guard cannot swallow _ImportAttempted, and even a guard that did would leave the name in
    # `attempts`. Without this case the docstring above would be an untested claim again.
    _guarded = compile("\n".join(["try:", "    import torch",
                                  "except ImportError:", "    pass"]),
                       "<guarded>", "exec")
    blocker.attempts.clear()
    try:
        exec(_guarded, {})
        check("blocker/catches-guarded-import", False,
              "a try/except ImportError guard swallowed the block; it is invisible again")
    except _ImportAttempted:
        check("blocker/catches-guarded-import", True)
    check("blocker/records-the-guarded-attempt", blocker.attempts == ["torch"],
          "recorded %r" % (blocker.attempts,))

    blocker.attempts.clear()
    import laya_onnx as laya_onnx_blocked          # noqa: E402
    import laya_onnx.collate                       # noqa: E402, F401
    import laya_onnx.postprocess                   # noqa: E402, F401
    import laya_onnx.sequence                      # noqa: E402, F401
    import laya_onnx.session                       # noqa: E402, F401
    import laya_onnx.external_data                 # noqa: E402, F401
    import laya_onnx.session_openvino              # noqa: E402, F401
    import laya_onnx.tokenizer                     # noqa: E402, F401
    import laya_onnx.truncation                    # noqa: E402, F401
    check("import/survives-torch-blocked", True)
except (ImportError, _ImportAttempted) as e:
    check("import/survives-torch-blocked", False, "raised %r" % (e,))

# The positive assertion, and the one a guarded import cannot dodge: over every runtime module
# imported above, the finder was never asked for torch or transformers at all.
check("import/never-even-asked-for-torch", blocker.attempts == [],
      "runtime modules attempted: %r" % (blocker.attempts,))

# Not the port's constraint -- `export/` is explicitly allowed to import torch -- but a
# property worth pinning: `python -m laya_onnx.export.refit_temps` exists to write a JSON key
# into an export's config, and it is the one CLI under export/ a *serving* operator plausibly
# runs. Its torch-reaching import of `..bench.eval_ece` is deferred into `refit_report`, so the
# CLI stays usable on a host that has never installed torch. If a future change genuinely needs
# torch at that module's scope, delete this check deliberately rather than re-hiding the cost.
try:
    import laya_onnx.export.refit_temps             # noqa: E402, F401
    check("export/refit_temps-imports-without-torch", True)
except (ImportError, _ImportAttempted) as e:
    check("export/refit_temps-imports-without-torch", False, "raised %r" % (e,))
finally:
    sys.meta_path.remove(blocker)

check("api/load-exists-under-block", hasattr(laya_onnx_blocked, "load"))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all no-torch tests passed")
sys.exit(1 if FAIL else 0)
