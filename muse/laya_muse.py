"""Laya decisions for the Muse runtime — no Hermes agent, no tool registry.

Self-contained port of this repo's ``plugin/laya_tool.py`` (same semantics,
same validation, including the score-criteria-list rule and the memory
bounds below), adapted to how Muse invokes tools:

- No ``tools.registry`` import / ``register()`` — Muse has no Hermes
  registry; the agent calls this module directly.
- Clean Python API: :func:`predict` returns dicts (not JSON strings) and
  raises :class:`LayaError` on validation/usage errors instead of burying
  them in an envelope.
- Batch-friendly: one long-lived Router per process. A single CLI call per
  decision would pay the ~15-40 s checkpoint load every time, so batch mode
  (``--batch``) runs many requests through one warm process.

Actions (parity with the Hermes tool, upstream laya 0.4.14): decide, schema,
schemabatch, batch, long, tournament, detect, route, triage, guard,
moderation, routemodel, email, status.

Memory bounds (the OOM fix, shared with the plugin): upstream
``predict_batch`` / ``predict_long`` default to ONE forward pass over every
state/window, so peak RAM grows with the item count — that OOM-killed a
7 GB test box. The batch and long paths here always pass
``batch_size=_FORWARD_BATCH`` so peak activation memory stays flat, batch
states are capped per state, and long states are capped at
``_LONG_STATE_CAP`` chars.

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
from typing import Any, Dict, List, Optional, Tuple

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
_MAX_TOURNAMENT_OPTS = 400
_MAX_BATCH = 64
_INLINE_CAP = 6000
_STATE_CAP = 8000
# Long scans materialise many windows per call; see the module docstring.
_LONG_STATE_CAP = 40_000
# States/windows per forward pass on the batch and long paths (OOM bound).
_FORWARD_BATCH = 8

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


def _parse_json_any(s: Any, name: str) -> Any:
    """Like _parse_json but accepts arrays too (batch states)."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str) or not s.strip():
        raise LayaError(f"{name} must be a non-empty JSON string")
    try:
        v = json.loads(s)
    except Exception as exc:
        raise LayaError(f"{name} is not valid JSON: {exc}")
    if not isinstance(v, (dict, list)) or not v:
        raise LayaError(f"{name} must be a non-empty JSON object or array")
    return v


def _validate_questions(qs: dict, max_opts: int = _MAX_OPTS) -> None:
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
            if len(crit) > max_opts:
                raise LayaError(
                    f"choice question '{qid}' has {len(crit)} options "
                    f"(> {max_opts}); use action=tournament to narrow it by elimination"
                )
        elif q["type"] == "score":
            if not isinstance(crit, list) or not crit:
                raise LayaError(
                    f"score question '{qid}' needs criteria as a non-empty list of "
                    "level descriptions (index 0 = lowest)"
                )


def _round(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 4)
    return v


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


def _predict_kwargs(model: str, task: str = "", lang: str = "",
                    min_confidence: Optional[float] = None,
                    max_len: Optional[int] = None,
                    head_max_len: Optional[int] = None) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if model and model != "auto":
        kw["model"] = model
    if task:
        kw["task"] = task
    if lang:
        kw["lang"] = lang
    if min_confidence is not None:
        kw["min_confidence"] = min_confidence
    if max_len is not None:
        kw["max_len"] = max_len
    if head_max_len is not None:
        kw["head_max_len"] = head_max_len
    return kw


def _slim(a: dict) -> dict:
    """Fallback projection when a payload exceeds the inline cap."""
    return {k: {kk: v.get(kk) for kk in ("choice", "noul", "score", "confidence",
                                         "answer_confidence")}
            for k, v in a.items() if isinstance(v, dict)}


def _slim_values(a: dict) -> dict:
    """Last-resort projection: the answer value only, no confidences.

    A full batch (up to 64 states) still exceeds the inline cap after
    _slim, because every item carries two confidence numbers per
    question. Dropping those keeps the choices (and their order) while
    fitting the cap; the payload is flagged with "compact": true.
    """
    return {k: {kk: v.get(kk) for kk in ("choice", "noul", "score") if kk in v}
            for k, v in a.items() if isinstance(v, dict)}


def _coerce_state(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        stripped = raw.strip()
        if stripped[:1] in ("{", "["):
            try:
                v = json.loads(stripped)
            except Exception:
                v = None
            if isinstance(v, (dict, list)):
                return v
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
    if isinstance(state, str) and len(state) > _STATE_CAP:
        raise LayaError(f"state too long for route probe (>{_STATE_CAP} chars)")
    try:
        d = _get_router().route(st, {})
    except LayaError:
        raise
    except Exception as exc:
        raise LayaError(f"route failed: {exc}")
    det = d.get("detection") or {}
    return {"ok": True, "m": d.get("model"), "repo": d.get("repo"),
            "why": d.get("reason"),
            "lang": det.get("language"), "script": det.get("script")}


def detect(state: Any) -> dict:
    """Language + script detection; loads no model."""
    st = _coerce_state(state)
    try:
        import laya

        d = laya.detect_language(st, explain=True)
    except Exception as exc:
        raise LayaError(f"detect failed: {exc}")
    return {"ok": True, "script": d.get("script"), "lang": d.get("language"),
            "en": d.get("is_english"), "non_latin": d.get("non_latin_fraction"),
            "undecided": d.get("language_undecided"),
            "why": (d.get("diagnostic") or {}).get("reason")}


def status() -> dict:
    from pathlib import Path

    try:
        router = _get_router()
        import laya
    except Exception as exc:
        raise LayaError(f"status failed: {exc}")
    hf = Path.home() / ".cache" / "huggingface"
    return {"ok": True, "v": getattr(laya, "__version__", "?"),
            "loaded": list(router.loaded or []),
            "revs": dict(router.loaded_revisions or {}),
            "hf_cache": hf.exists()}


def _run_predict(state: Any, questions: dict, kw: Dict[str, Any]) -> dict:
    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    try:
        res = router.predict(state, questions, **kw)
    except TypeError:
        # older laya without min_confidence / max_len / head_max_len
        trimmed = {k: v for k, v in kw.items()
                   if k not in ("min_confidence", "max_len", "head_max_len")}
        try:
            res = router.predict(state, questions, **trimmed)
        except Exception as exc2:
            return {"ok": False, "e": f"predict failed: {str(exc2)[:300]}"}
    except Exception as exc:
        return {"ok": False, "e": f"predict failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {}) or {}
    out: dict = {
        "ok": True,
        "a": _compact_answers(res.get("answers", {}) or {}),
        "r": {"m": routing.get("model"), "why": routing.get("reason")},
        "ms": ms,
    }
    usage = res.get("usage") or {}
    if usage.get("truncated"):
        out["t"] = True
    s = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    if len(s) > _INLINE_CAP:
        out["a"] = _slim(out["a"])
        s = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
        if len(s) > _INLINE_CAP:
            return {"ok": False,
                    "e": f"answer payload too large ({len(s)} chars); ask fewer questions"}
    return out


def _action_schema(state: Any, schema: Any, kw: Dict[str, Any]) -> dict:
    """JSON-schema-driven typed values (enum choice / integer level / bool)."""
    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    try:
        res = router.decide(state, schema=schema, return_details=True, **kw)
    except Exception as exc:
        return {"ok": False, "e": f"schema decide failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    conf = getattr(res, "confidence", {}) or {}
    aconf = getattr(res, "answer_confidence", {}) or {}
    return {"ok": True, "v": getattr(res, "values", None),
            "c": {k: _round(v) for k, v in conf.items()},
            "ac": {k: _round(v) for k, v in aconf.items()},
            "ms": ms}


def _action_schemabatch(states: List[Any], schema: Any, kw: Dict[str, Any]) -> dict:
    """One JSON Schema over many states (upstream Router.decide_batch).

    The schema is planned once and its questions run over every state
    through the batched path, in input order. Chunked like `batch`
    (batch_size=_FORWARD_BATCH).
    """
    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    call_kw = {k: v for k, v in kw.items()
               if k in ("model", "task", "lang", "max_len", "head_max_len")}
    call_kw["batch_size"] = _FORWARD_BATCH
    mc = kw.get("min_confidence")
    try:
        results = router.decide_batch(states, schema=schema, return_details=True,
                                      min_confidence=mc, **call_kw)
    except Exception as exc:
        return {"ok": False, "e": f"schemabatch failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    out = []
    for res in results:
        conf = getattr(res, "confidence", {}) or {}
        aconf = getattr(res, "answer_confidence", {}) or {}
        routing = getattr(res, "routing", {}) or {}
        out.append({"v": getattr(res, "values", None),
                    "c": {k: _round(v) for k, v in conf.items()},
                    "ac": {k: _round(v) for k, v in aconf.items()},
                    "m": routing.get("model")})
    payload: dict = {"ok": True, "n": len(out), "batch": out, "ms": ms}
    s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(s) > _INLINE_CAP:
        for item in payload["batch"]:
            item["c"] = {}
            item["ac"] = {}
        s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(s) > _INLINE_CAP:
            for item in payload["batch"]:
                item.pop("m", None)
            payload["compact"] = True
            s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(s) > _INLINE_CAP:
            return {"ok": False,
                    "e": f"schemabatch payload too large ({len(s)} chars); fewer states"}
    return payload


def _action_batch(states: List[Any], questions: dict, kw: Dict[str, Any]) -> dict:
    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    # predict_batch takes routing hints per request, not per call; batch_size
    # bounds states per forward pass (see _FORWARD_BATCH).
    per_req = {k: v for k, v in kw.items()
               if k in ("model", "task", "lang", "max_len", "head_max_len")}
    call_kw = {k: v for k, v in kw.items() if k == "min_confidence"}
    call_kw["batch_size"] = _FORWARD_BATCH
    reqs = []
    for s in states:
        r = {"state": s, "questions": questions}
        r.update(per_req)
        reqs.append(r)
    try:
        results = router.predict_batch(reqs, **call_kw)
    except Exception as exc:
        return {"ok": False, "e": f"batch failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    out = []
    for res in results:
        routing = (res or {}).get("routing", {}) or {}
        out.append({"a": _compact_answers((res or {}).get("answers", {}) or {}),
                    "m": routing.get("model")})
    payload: dict = {"ok": True, "n": len(out), "batch": out, "ms": ms}
    s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(s) > _INLINE_CAP:
        for item in payload["batch"]:
            item["a"] = _slim(item["a"])
        s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(s) > _INLINE_CAP:
            for item in payload["batch"]:
                item["a"] = _slim_values(item["a"])
            payload["compact"] = True
            s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(s) > _INLINE_CAP:
            return {"ok": False,
                    "e": f"batch payload too large ({len(s)} chars); fewer states or questions"}
    return payload


def _action_long(state: Any, questions: dict, kw: Dict[str, Any],
                 window: Optional[int], stride: Optional[int]) -> dict:
    try:
        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    # predict_long takes routing hints + window/stride; it has no min_confidence.
    # batch_size caps windows per forward pass (see _FORWARD_BATCH).
    ckw = {k: v for k, v in kw.items() if k in ("model", "task", "lang")}
    try:
        wkw: Dict[str, Any] = {"batch_size": _FORWARD_BATCH}
        if window:
            wkw["window"] = window
        if stride:
            wkw["stride"] = stride
        res = router.predict_long(state, questions, **wkw, **ckw)
    except Exception as exc:
        return {"ok": False, "e": f"long predict failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {}) or {}
    usage = res.get("usage") or {}
    raw = res.get("answers", {}) or {}
    answers = _compact_answers(raw)
    for qid, a in answers.items():
        entry = raw.get(qid)
        win = entry.get("window") if isinstance(entry, dict) else None
        if isinstance(win, dict):
            a["w"] = win.get("index")
    return {"ok": True, "a": answers, "r": {"m": routing.get("model")},
            "win": usage.get("windows"), "ms": ms}


def _action_tournament(state: Any, questions: dict, kw: Dict[str, Any],
                       group_size: int) -> dict:
    try:
        import laya

        router = _get_router()
    except Exception as exc:
        return {"ok": False, "e": f"router init failed: {exc}"}
    t0 = time.time()
    try:
        res = laya.predict_tournament(router, state, questions,
                                      group_size=group_size, **kw)
    except Exception as exc:
        return {"ok": False, "e": f"tournament failed: {str(exc)[:300]}"}
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {}) or {}
    tour = res.get("tournament") or {}
    narrow = {qid: {"n": m.get("n"), "rounds": m.get("rounds"), "labels": m.get("labels")}
              for qid, m in tour.items()}
    return {"ok": True, "a": _compact_answers(res.get("answers", {}) or {}),
            "r": {"m": routing.get("model")}, "t": narrow, "ms": ms}


def _as_float(v: Any, name: str) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        raise LayaError(f"{name} must be a number")


def _as_int(v: Any, name: str) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        raise LayaError(f"{name} must be an integer")


def predict(action: str = "decide", state: Any = None,
            questions: Any = None, categories: Any = None,
            model: str = "auto", text: Any = None,
            schema: Any = None, states: Any = None,
            subject: str = "", sender: str = "",
            task: str = "", lang: str = "",
            min_confidence: Any = None, max_len: Any = None,
            head_max_len: Any = None,
            window: Any = None, stride: Any = None,
            group_size: Any = 16) -> dict:
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
    task = str(task or "")
    lang = str(lang or "")
    mc = _as_float(min_confidence, "min_confidence")
    ml = _as_int(max_len, "max_len")
    hml = _as_int(head_max_len, "head_max_len")
    kw = _predict_kwargs(model, task, lang, mc, ml, hml)

    if action == "status":
        return status()
    if action == "route":
        return route(state if state is not None else text)
    if action == "detect":
        return detect(state if state is not None else text)

    if action == "batch":
        states_v = _parse_json_any(states, "states")
        if not isinstance(states_v, list) or not states_v:
            raise LayaError("states must be a non-empty JSON array")
        if len(states_v) > _MAX_BATCH:
            raise LayaError(f"too many states ({len(states_v)} > {_MAX_BATCH})")
        for s in states_v:
            slen = len(s) if isinstance(s, str) else len(json.dumps(s, ensure_ascii=False))
            if slen > _STATE_CAP:
                raise LayaError(f"batch state too long (>{_STATE_CAP} chars); "
                                "pass the relevant excerpt per state")
        qs = _parse_json(questions, "questions")
        _validate_questions(qs)
        return _action_batch(states_v, qs, kw)

    if action == "schemabatch":
        states_v = _parse_json_any(states, "states")
        if not isinstance(states_v, list) or not states_v:
            raise LayaError("states must be a non-empty JSON array")
        if len(states_v) > _MAX_BATCH:
            raise LayaError(f"too many states ({len(states_v)} > {_MAX_BATCH})")
        for s in states_v:
            slen = len(s) if isinstance(s, str) else len(json.dumps(s, ensure_ascii=False))
            if slen > _STATE_CAP:
                raise LayaError(f"batch state too long (>{_STATE_CAP} chars); "
                                "pass the relevant excerpt per state")
        schema_v = _parse_json(schema, "schema")
        return _action_schemabatch(states_v, schema_v, kw)

    st = _coerce_state(state if state is not None else text)

    if action == "schema":
        schema_v = _parse_json(schema, "schema")
        return _action_schema(st, schema_v, kw)

    if action == "long":
        qs = _parse_json(questions, "questions")
        _validate_questions(qs)
        if isinstance(st, str) and len(st) > _LONG_STATE_CAP:
            raise LayaError(f"state too long for long scan (>{_LONG_STATE_CAP} chars)")
        win = _as_int(window, "window")
        strd = _as_int(stride, "stride")
        return _action_long(st, qs, kw, win, strd)

    if action == "tournament":
        qs = _parse_json(questions, "questions")
        _validate_questions(qs, max_opts=_MAX_TOURNAMENT_OPTS)
        gs = _as_int(group_size, "group_size")
        if gs is None or gs < 2:
            raise LayaError("group_size must be >= 2")
        return _action_tournament(st, qs, kw, gs)

    if action not in _PRESETS and action != "decide":
        raise LayaError("unknown action '%s'; use decide|schema|schemabatch|batch|long|"
                        "tournament|detect|route|triage|guard|moderation|routemodel|email|status"
                        % action)

    if isinstance(st, str) and len(st) > _STATE_CAP:
        raise LayaError(f"state too long (>{_STATE_CAP} chars); pass the relevant excerpt")

    if action == "decide":
        qs = _parse_json(questions, "questions")
        _validate_questions(qs)
        return _run_predict(st, qs, kw)

    cats = _parse_json(categories, "categories") if categories else None
    qs, field = _preset_questions(action, cats)
    _validate_questions(qs)
    if action == "email" and (subject or sender):
        try:
            import laya

            body = st if isinstance(st, str) else ""
            st = laya.email_state(str(subject or ""), body, str(sender or "") or None)
        except Exception as exc:
            return {"ok": False, "e": f"email state failed: {exc}"}
    else:
        st = _wrap_state(st, field)
    return _run_predict(st, qs, kw)


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
                model=req.get("model", "auto"),
                schema=req.get("schema"),
                states=req.get("states"),
                subject=req.get("subject", "") or "",
                sender=req.get("sender", "") or "",
                task=req.get("task", "") or "",
                lang=req.get("lang", "") or "",
                min_confidence=req.get("min_confidence"),
                max_len=req.get("max_len"),
                head_max_len=req.get("head_max_len"),
                window=req.get("window"),
                stride=req.get("stride"),
                group_size=req.get("group_size", 16)))
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
    ap.add_argument("--schema")
    ap.add_argument("--states")
    ap.add_argument("--categories")
    ap.add_argument("--subject", default="")
    ap.add_argument("--sender", default="")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--task", default="")
    ap.add_argument("--lang", default="")
    ap.add_argument("--min-confidence", dest="min_confidence")
    ap.add_argument("--max-len", dest="max_len")
    ap.add_argument("--head-max-len", dest="head_max_len")
    ap.add_argument("--window")
    ap.add_argument("--stride")
    ap.add_argument("--group-size", dest="group_size", default=16)
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
                    model=ns.model, schema=ns.schema, states=ns.states,
                    subject=ns.subject, sender=ns.sender,
                    task=ns.task, lang=ns.lang,
                    min_confidence=ns.min_confidence, max_len=ns.max_len,
                    head_max_len=ns.head_max_len,
                    window=ns.window, stride=ns.stride, group_size=ns.group_size)
    except LayaError as exc:
        print(json.dumps({"ok": False, "e": str(exc)}, separators=(",", ":")))
        return 0
    print(json.dumps(d, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
