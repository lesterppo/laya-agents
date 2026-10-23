"""Laya decisions for the Muse runtime — no Hermes agent, no tool registry.

Self-contained port of this repo's ``plugin/laya_tool.py`` (same semantics,
same validation, including the score-criteria-list rule), adapted to how Muse
invokes tools:

- No ``tools.registry`` import / ``register()`` — Muse has no Hermes
  registry; the agent calls this module directly.
- Clean Python API: :func:`predict` returns dicts (not JSON strings) and
  raises :class:`LayaError` on validation/usage errors instead of burying
  them in an envelope.
- Batch-friendly: one long-lived Router per process. A single CLI call per
  decision would pay the ~15-40 s checkpoint load every time, so batch mode
  (``--batch``) runs many requests through one warm process.

Runtime notes (applied at import, before anything imports laya):
- Some sandboxed/proxied environments ship a ``no_proxy``/``NO_PROXY`` with
  IPv6 literals that crash httpx inside huggingface_hub
  (``Invalid port: ':1]'``). If detected, the value is replaced with
  ``localhost,127.0.0.1``. Harmless elsewhere.
- ``HF_HUB_DISABLE_XET=1`` is set (only if unset) so checkpoint downloads
  fall back to plain HTTP resume when hf-xet stalls behind a proxy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

# --- runtime network fixes (must run before anything imports laya) ---
def _sanitize_proxy_env() -> None:
    # Some sandboxed/proxied environments ship no_proxy/NO_PROXY with IPv6
    # literals (e.g. "[::1]") that crash httpx inside huggingface_hub with
    # "Invalid port: ':1]'". Only touch the value when it looks broken.
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var, "")
        if ("[" in val or "]" in val) and val.strip():
            os.environ[var] = "localhost,127.0.0.1"


_sanitize_proxy_env()
# Plain-HTTP resume for checkpoint downloads when hf-xet stalls behind a proxy.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# SHA-256 pins for the upstream checkpoint weights, verified by laya >= 0.3.22
# before any weight file is parsed or executed (fail-closed: a tampered or
# silently-updated checkpoint raises instead of loading). Digests computed from
# convaiinnovations/laya @ 55cf4c4e (2026-09-30; files identical to the
# aa8c91ca snapshot). typed-decisions ships from a separate repo
# (convaiinnovations/laya-typed-decisions) and is left unpinned here — it is
# never auto-loaded. To re-pin after an upstream weight update: sha256sum the
# new model.safetensors files and replace the values below.
_CHECKPOINT_DIGESTS = {
    "english": {"model.safetensors": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"},
    "multilingual": {"model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"},
}
os.environ.setdefault("LAYA_SHA256_DIGESTS", json.dumps(_CHECKPOINT_DIGESTS))

_QTYPES = ("choice", "score", "noul")
_MODELS = ("auto", "english", "multilingual", "typed-decisions")
_PRESETS = ("triage", "guard", "moderation", "routemodel", "email")
_MAX_Q = 20
_MAX_OPTS = 50
_INLINE_CAP = 6000

_ROUTER = None


class LayaError(Exception):
    """Validation or usage error — raised, not returned."""


# laya + torch live in a dedicated venv (~/.hermes/laya-venv): torch's PyPI
# wheel drags in the CUDA runtime, and the managed-env resolver does not use
# the CPU index. Appending its site-packages keeps `import laya` working
# without a global install. Appended, not inserted, so packages already
# present keep priority.
_LAYA_VENV = os.path.expanduser("~/.hermes/laya-venv")


def _bootstrap_laya_path() -> bool:
    """Make the dedicated laya venv importable from this interpreter."""
    if "laya" in sys.modules:
        return True
    try:
        import laya  # noqa: F401

        return True
    except Exception:
        pass
    lib = os.path.join(_LAYA_VENV, "lib")
    if not os.path.isdir(lib):
        return False
    cands = [os.path.join(lib, e, "site-packages")
             for e in sorted(os.listdir(lib)) if e.startswith("python")]
    mine = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cands.sort(key=lambda p: (not p.startswith(os.path.join(lib, mine)), p))
    for sp in cands:
        if not os.path.isdir(sp):
            continue
        if sp not in sys.path:
            sys.path.append(sp)
        try:
            import laya  # noqa: F401

            return True
        except Exception:
            continue
    return False


def _get_router():
    global _ROUTER
    if _ROUTER is None:
        if not _bootstrap_laya_path():
            raise LayaError(
                "laya not installed: run scripts/install_laya_env.sh, or "
                "pip install laya (needs python>=3.10, torch, transformers)"
            )
        from laya import Router

        _ROUTER = Router()
    return _ROUTER


def _parse_json(s: Any, name: str) -> dict:
    if isinstance(s, dict):
        v = s
    elif not isinstance(s, str) or not s.strip():
        raise LayaError(f"{name} must be a non-empty JSON object string")
    else:
        try:
            v = json.loads(s)
        except Exception as exc:
            raise LayaError(f"{name} is not valid JSON: {exc}")
    if not isinstance(v, dict) or not v:
        raise LayaError(f"{name} must be a non-empty JSON object")
    return v


def _validate_questions(qs: dict) -> None:
    if len(qs) > _MAX_Q:
        raise LayaError(f"too many questions ({len(qs)} > {_MAX_Q})")
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise LayaError(f"question '{qid}' must be an object")
        if q.get("type") not in _QTYPES:
            raise LayaError(f"question '{qid}' needs type one of {list(_QTYPES)}")
        if not str(q.get("instructions", "")).strip():
            raise LayaError(f"question '{qid}' needs non-empty instructions")
        crit = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(crit, dict) or not crit:
                raise LayaError(f"choice question '{qid}' needs criteria object")
            if len(crit) > _MAX_OPTS:
                raise LayaError(
                    f"choice question '{qid}' has {len(crit)} options "
                    f"(> {_MAX_OPTS}); split or shortlist first"
                )
        elif q["type"] == "score":
            if not isinstance(crit, list) or not crit:
                raise LayaError(
                    f"score question '{qid}' needs criteria as a non-empty list of "
                    "level descriptions (index 0 = lowest)"
                )


def _compact_answers(answers: dict) -> dict:
    """Token-efficient projection of laya 0.4.x answer dicts.

    0.4.x answer keys: type, choice|noul|score, probabilities, legend,
    confidence, answer_confidence, action, abstention, abstention_threshold.
    """
    out: dict = {}
    for qid, a in answers.items():
        if not isinstance(a, dict):
            out[qid] = a
            continue
        c: dict = {}
        for k in ("choice", "noul", "score", "confidence", "answer_confidence",
                  "abstention", "abstention_threshold", "legend", "type"):
            if k in a:
                v = a[k]
                if isinstance(v, dict):
                    # legend values are level DESCRIPTIONS (strings); only the
                    # numeric ones get rounded.
                    v = {kk: (round(float(vv), 4) if isinstance(vv, (int, float))
                              and not isinstance(vv, bool) else vv)
                         for kk, vv in list(v.items())[:_MAX_OPTS]}
                elif isinstance(v, float):
                    v = round(v, 4)
                elif isinstance(v, list):
                    v = [round(float(x), 4) if isinstance(x, (int, float))
                         and not isinstance(x, bool) else x for x in v[:_MAX_OPTS]]
                c[k] = v
        probs = a.get("probabilities")
        if isinstance(probs, dict):
            items = sorted(probs.items(), key=lambda kv: -float(kv[1]))[:12]
            c["p"] = {str(kk): round(float(vv), 4) for kk, vv in items}
        out[qid] = c or {k: str(v)[:200] for k, v in list(a.items())[:6]}
    return out


def _coerce_state(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            maybe = json.loads(raw)
            return maybe if isinstance(maybe, dict) else raw
        except Exception:
            return raw
    raise LayaError("state/text must be non-empty (text or JSON object)")


def _preset_questions(name: str, categories: Optional[dict] = None) -> Tuple[dict, str]:
    """Return (questions, state_field) for a preset.

    laya presets name the state field they read (`message` for triage, `body`
    for email, `prompt` for guard, `post` for moderation, `request` for
    routemodel). Handing raw text over under the wrong key silently degrades
    the answer, so the field is resolved from the preset itself via
    laya.presets.state_field().
    """
    import laya
    from laya import presets as _presets

    if name == "triage":
        q = laya.triage_questions()
    elif name == "guard":
        q = laya.guard_questions()
    elif name == "moderation":
        q = laya.moderation_questions()
    elif name == "routemodel":
        q = laya.router_questions()
    elif name == "email":
        q = laya.email_questions(categories or None)
    else:
        raise LayaError(f"unknown preset '{name}'")
    return q, _presets.state_field(q) or "text"


def _wrap_state(state: Any, field: str) -> Any:
    """Put plain text under the field the preset reads; pass dicts through."""
    if isinstance(state, dict):
        return state
    return {field: state}


def route(state: Any) -> dict:
    """Sub-ms checkpoint routing; loads no model."""
    st = _coerce_state(state)
    if isinstance(state, str) and len(state) > 8000:
        raise LayaError("state too long for route probe (>8000 chars)")
    try:
        d = _get_router().route(st, {})
    except Exception as exc:
        raise LayaError(f"route failed: {exc}")
    return {"ok": True, "m": d.get("model"), "repo": d.get("repo"),
            "why": d.get("reason")}


def status() -> dict:
    from pathlib import Path

    try:
        _bootstrap_laya_path()
        import laya

        loaded = _get_router().loaded
    except Exception as exc:
        raise LayaError(f"status failed: {exc}")
    hf = Path.home() / ".cache" / "huggingface"
    return {"ok": True, "v": getattr(laya, "__version__", "?"),
            "loaded": loaded, "hf_cache": hf.exists()}


def predict(action: str = "decide", state: Any = None,
            questions: Any = None, categories: Any = None,
            model: str = "auto", text: Any = None) -> dict:
    """Run one decision. Returns the result dict (``ok``/``a``/``r``/``ms``).

    Raises :class:`LayaError` on validation/usage errors; model/predict
    failures are returned as ``{"ok": False, "e": ...}`` so batch runs
    survive them.
    """
    if not _bootstrap_laya_path():
        raise LayaError("laya not installed: run scripts/install_laya_env.sh, or "
                        "pip install laya (needs python>=3.10, torch, transformers)")
    action = str(action or "decide").lower()
    model = str(model or "auto").lower()
    if model not in _MODELS:
        raise LayaError(f"unknown model '{model}'; use one of {list(_MODELS)}")

    if action == "status":
        return status()
    if action == "route":
        return route(state if state is not None else text)

    if action not in _PRESETS and action != "decide":
        raise LayaError("unknown action '%s'; use decide|route|triage|guard|"
                        "moderation|routemodel|email|status" % action)

    st = _coerce_state(state if state is not None else text)
    if isinstance(st, str) and len(st) > 8000:
        raise LayaError("state too long (>8000 chars); pass the relevant excerpt")

    if action == "decide":
        qs = _parse_json(questions, "questions")
        field = ""
    else:
        cats = _parse_json(categories, "categories") if categories else None
        qs, field = _preset_questions(action, cats)
    _validate_questions(qs)
    if field:
        st = _wrap_state(st, field)

    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    try:
        kw: Dict[str, Any] = {}
        if model != "auto":
            kw["model"] = model
        res = router.predict(st, qs, **kw)
    except Exception as exc:
        return {"ok": False, "e": f"predict failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {})
    out: dict = {
        "ok": True,
        "a": _compact_answers(res.get("answers", {})),
        "r": {"m": routing.get("model"), "why": routing.get("reason")},
        "ms": ms,
    }
    s = json.dumps(out, separators=(",", ":"))
    if len(s) > _INLINE_CAP:
        out["a"] = {k: {"choice": v.get("choice"), "confidence": v.get("confidence"),
                        "noul": v.get("noul"), "score": v.get("score")}
                    for k, v in out["a"].items() if isinstance(v, dict)}
        s = json.dumps(out, separators=(",", ":"))
        if len(s) > _INLINE_CAP:
            return {"ok": False,
                    "e": f"answer payload too large ({len(s)} chars); ask fewer questions"}
    return out


def run_batch(requests: list) -> list:
    """Run many requests through one warm Router. Never raises per-item:
    failures come back as ``{"ok": False, "e": ...}`` dicts in place."""
    out = []
    for req in requests:
        if not isinstance(req, dict):
            out.append({"ok": False, "e": "batch line must be a JSON object"})
            continue
        try:
            out.append(predict(
                action=req.get("action", "decide"),
                state=req.get("state", req.get("text")),
                questions=req.get("questions"),
                categories=req.get("categories"),
                model=req.get("model", "auto")))
        except LayaError as exc:
            out.append({"ok": False, "e": str(exc)})
        except Exception as exc:  # noqa: BLE001 — batch must survive
            out.append({"ok": False, "e": f"laya error: {exc}"})
    return out


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Laya fast typed decisions (Muse runtime)")
    ap.add_argument("--action", default="decide")
    ap.add_argument("--state")
    ap.add_argument("--text")
    ap.add_argument("--questions")
    ap.add_argument("--categories")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--batch",
                    help="JSONL file (or - for stdin) of request objects; "
                         "one warm process handles all lines, one JSON object "
                         "per line on stdout")
    ns = ap.parse_args(argv)

    if ns.batch:
        fh = sys.stdin if ns.batch == "-" else open(ns.batch, encoding="utf-8")
        results: list = []
        pending: list = []
        pending_idx: list = []
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    pending.append(json.loads(line))
                    pending_idx.append(len(results))
                    results.append(None)
                except Exception as exc:
                    results.append({"ok": False,
                                    "e": f"batch line is not valid JSON: {exc}"})
        for i, r in zip(pending_idx, run_batch(pending)):
            results[i] = r
        for r in results:
            print(json.dumps(r, separators=(",", ":")))
        return 0

    try:
        d = predict(action=ns.action, state=ns.state, text=ns.text,
                    questions=ns.questions, categories=ns.categories,
                    model=ns.model)
    except LayaError as exc:
        print(json.dumps({"ok": False, "e": str(exc)}, separators=(",", ":")))
        return 0
    print(json.dumps(d, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
