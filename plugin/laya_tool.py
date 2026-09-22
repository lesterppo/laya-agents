"""Laya native Hermes tool — fast non-autoregressive System 1 decisions.

Wraps the `laya` pip package (Apache 2.0, NandhaKishorM/laya):
typed decisions (choice/score/noul) in a single forward pass, ~33ms on
GPU / ~200-500ms on CPU, no text generation so nothing to parse and
nothing to hallucinate. Router picks english / multilingual /
typed-decisions per request from sub-ms script detection.

Actions: decide (default), route, triage, guard, moderation, routemodel,
email, status. Preset actions run the bundled question schemas so the
agent does not hand-craft them.

Author: Peter/lesterppo
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

_QTYPES = ("choice", "score", "noul")
_MODELS = ("auto", "english", "multilingual", "typed-decisions")
_PRESETS = ("triage", "guard", "moderation", "routemodel", "email")
_MAX_Q = 20
_MAX_OPTS = 50
_INLINE_CAP = 6000

_ROUTER = None


def _get_router():
    global _ROUTER
    if _ROUTER is None:
        from laya import Router

        _ROUTER = Router()
    return _ROUTER


def _check() -> bool:
    try:
        import laya  # noqa: F401

        return True
    except Exception:
        return False


def _ok(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"))


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "e": msg}, separators=(",", ":"))


def _parse_json(s: Any, name: str) -> tuple[bool, Any, str]:
    if isinstance(s, dict):
        return True, s, ""
    if not isinstance(s, str) or not s.strip():
        return False, None, f"{name} must be a non-empty JSON object string"
    try:
        v = json.loads(s)
    except Exception as exc:
        return False, None, f"{name} is not valid JSON: {exc}"
    if not isinstance(v, dict) or not v:
        return False, None, f"{name} must be a non-empty JSON object"
    return True, v, ""


def _validate_questions(qs: dict) -> tuple[bool, str]:
    if len(qs) > _MAX_Q:
        return False, f"too many questions ({len(qs)} > {_MAX_Q})"
    for qid, q in qs.items():
        if not isinstance(q, dict):
            return False, f"question '{qid}' must be an object"
        if q.get("type") not in _QTYPES:
            return False, f"question '{qid}' needs type one of {list(_QTYPES)}"
        if not str(q.get("instructions", "")).strip():
            return False, f"question '{qid}' needs non-empty instructions"
        crit = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(crit, dict) or not crit:
                return False, f"choice question '{qid}' needs criteria object"
            if len(crit) > _MAX_OPTS:
                return False, (
                    f"choice question '{qid}' has {len(crit)} options "
                    f"(> {_MAX_OPTS}); split or shortlist first"
                )
    return True, ""


def _compact_answers(answers: dict) -> dict:
    out: dict = {}
    for qid, a in answers.items():
        if not isinstance(a, dict):
            out[qid] = a
            continue
        c: dict = {}
        for k in ("choice", "confidence", "noul", "score", "probs",
                  "distribution", "p", "label"):
            if k in a:
                v = a[k]
                if isinstance(v, dict):
                    v = {kk: round(float(vv), 4) for kk, vv in
                         list(v.items())[:_MAX_OPTS]}
                elif isinstance(v, float):
                    v = round(v, 4)
                elif isinstance(v, list):
                    v = [round(float(x), 4) if isinstance(x, (int, float))
                         else x for x in v[:_MAX_OPTS]]
                c[k] = v
        out[qid] = c or {k: str(v)[:200] for k, v in list(a.items())[:6]}
    return out


def _run_predict(state: Any, questions: dict,
                 model: str) -> str:
    try:
        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    try:
        kw: Dict[str, Any] = {}
        if model != "auto":
            kw["model"] = model
        res = router.predict(state, questions, **kw)
    except Exception as exc:
        return _err(f"predict failed: {str(exc)[:300]}")
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {})
    out: dict = {
        "ok": True,
        "a": _compact_answers(res.get("answers", {})),
        "r": {"m": routing.get("model"), "why": routing.get("reason")},
        "ms": ms,
    }
    s = _ok(out)
    if len(s) > _INLINE_CAP:
        out["a"] = {k: {"choice": v.get("choice"), "confidence": v.get("confidence"),
                        "noul": v.get("noul"), "score": v.get("score")}
                    for k, v in out["a"].items() if isinstance(v, dict)}
        s = _ok(out)
        if len(s) > _INLINE_CAP:
            return _err(f"answer payload too large ({len(s)} chars); ask fewer questions")
    return s


def _action_route(state_raw: Any) -> str:
    if isinstance(state_raw, str):
        st: Any = state_raw.strip() or None
        if not st:
            return _err("state must be non-empty text or a JSON object string")
        try:
            maybe = json.loads(state_raw)
            if isinstance(maybe, dict):
                st = maybe
        except Exception:
            pass
    elif isinstance(state_raw, dict):
        st = state_raw
    else:
        return _err("state must be text or a JSON object")
    if isinstance(state_raw, str) and len(state_raw) > 8000:
        return _err("state too long for route probe (>8000 chars)")
    try:
        router = _get_router()
        d = router.route(st, {})
    except Exception as exc:
        return _err(f"route failed: {exc}")
    return _ok({"ok": True, "m": d.get("model"), "repo": d.get("repo"),
                "why": d.get("reason")})


def _preset_questions(name: str, categories: Optional[dict] = None) -> dict:
    import laya

    if name == "triage":
        return laya.triage_questions()
    if name == "guard":
        return laya.guard_questions()
    if name == "moderation":
        return laya.moderation_questions()
    if name == "routemodel":
        return laya.router_questions()
    if name == "email":
        return laya.email_questions(categories or None)
    raise ValueError(f"unknown preset '{name}'")


def _handler(args: dict) -> str:
    if not _check():
        return _err("laya not installed: pip install laya (needs python>=3.10, torch, transformers)")
    action = str(args.get("action", "decide") or "decide").lower()
    model = str(args.get("model", "auto") or "auto").lower()
    if model not in _MODELS:
        return _err(f"unknown model '{model}'; use one of {list(_MODELS)}")

    if action == "status":
        try:
            import laya

            router = _get_router()
            loaded = router.loaded
        except Exception as exc:
            return _err(f"status failed: {exc}")
        from pathlib import Path

        hf = Path.home() / ".cache" / "huggingface"
        return _ok({"ok": True, "v": getattr(laya, "__version__", "?"),
                    "loaded": loaded, "hf_cache": hf.exists()})

    if action == "route":
        return _action_route(args.get("state", args.get("text", "")))

    if action in _PRESETS or action == "decide":
        raw_state = args.get("state", args.get("text", ""))
        if isinstance(raw_state, dict):
            state = raw_state
        elif isinstance(raw_state, str) and raw_state.strip():
            try:
                maybe = json.loads(raw_state)
                state = maybe if isinstance(maybe, dict) else raw_state
            except Exception:
                state = raw_state
        else:
            return _err("state/text must be non-empty (text or JSON object)")
        if isinstance(state, str) and len(state) > 8000:
            return _err("state too long (>8000 chars); pass the relevant excerpt")
        if action == "decide":
            ok, qs, emsg = _parse_json(args.get("questions", ""), "questions")
            if not ok:
                return _err(emsg)
        else:
            try:
                cats = None
                if args.get("categories"):
                    okc, cats, emsg = _parse_json(args.get("categories"), "categories")
                    if not okc:
                        return _err(emsg)
                qs = _preset_questions(action if action != "routemodel" else "routemodel", cats)
            except Exception as exc:
                return _err(str(exc))
        okv, vmsg = _validate_questions(qs)
        if not okv:
            return _err(vmsg)
        return _run_predict(state, qs, model)

    return _err(f"unknown action '{action}'; use decide|route|triage|guard|moderation|routemodel|email|status")


def _runner(args):
    try:
        return _handler(args if isinstance(args, dict) else {})
    except Exception as exc:  # noqa: BLE001 — tool errors are values
        return _err(f"laya tool error: {exc}")


LAYA_SCHEMA: dict[str, Any] = {
    "name": "laya",
    "description": "Fast typed decisions (choice/score/noul) in one forward pass, no"
    " hallucination. decide=text+questions JSON; route=which checkpoint, no model"
    " load; triage/guard/moderation/routemodel/email=preset schemas; status=health.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "decide (default), route, triage, guard, moderation,"
                " routemodel, email, status.",
            },
            "state": {
                "type": "string",
                "description": "Text or JSON object to decide over (required except status).",
            },
            "text": {
                "type": "string",
                "description": "Alias for state (plain text).",
            },
            "questions": {
                "type": "string",
                "description": "JSON object of typed questions for decide (choice/score/noul"
                " + instructions + criteria).",
            },
            "categories": {
                "type": "string",
                "description": "Optional JSON object overriding email preset categories.",
            },
            "model": {
                "type": "string",
                "description": "auto (default, Router picks), english, multilingual,"
                " typed-decisions.",
            },
        },
    },
}


try:  # module-level self-registration, same pattern as the sibling tools
    from tools.registry import registry as _registry

    _registry.register(
        name="laya",
        toolset="laya",
        schema=LAYA_SCHEMA,
        handler=lambda args, **kw: _runner(args),
        check_fn=_check,
        emoji="⚡",
    )
except ImportError:
    pass


def register(registry):
    registry.register(
        name="laya",
        toolset="laya",
        schema=LAYA_SCHEMA,
        handler=lambda args, **kw: _runner(args),
        check_fn=_check,
        emoji="⚡",
    )
