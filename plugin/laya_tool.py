"""Laya native Hermes tool — fast non-autoregressive System 1 decisions.

Wraps the `laya` pip package (Apache 2.0, NandhaKishorM/laya), written
against 0.4.14: typed decisions (choice/score/noul) in a single forward
pass, no text generation so nothing to parse and nothing to hallucinate.
Router picks english / multilingual / typed-decisions per request from
sub-ms script detection.

Actions: decide (default), schema, batch, long, tournament, detect, route,
triage, guard, moderation, routemodel, email, status.

Author: Peter/lesterppo
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# --- runtime network fixes (must run before anything imports laya) ---
# Some sandboxed/proxied environments ship a no_proxy/NO_PROXY with bracketed
# IPv6 literals (e.g. "[::1]") that crash httpx inside huggingface_hub with
# "Invalid port: ':1]'" on the first checkpoint download. Replace only a
# value that looks broken; a clean value is left untouched. Same fix as the
# Muse module (muse/laya_muse.py) — the plugin needs it too: a plain Linux
# box behind such a proxy failed every live call this way (2026-10-24).
def _sanitize_proxy_env() -> None:
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var, "")
        if ("[" in val or "]" in val) and val.strip():
            os.environ[var] = "localhost,127.0.0.1"


_sanitize_proxy_env()
# Plain-HTTP resume for checkpoint downloads when hf-xet stalls behind a proxy.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_QTYPES = ("choice", "score", "noul")
_MODELS = ("auto", "english", "multilingual", "typed-decisions")
_PRESETS = ("triage", "guard", "moderation", "routemodel", "email")
_MAX_Q = 20
_MAX_OPTS = 50
_MAX_TOURNAMENT_OPTS = 400
_MAX_BATCH = 64
_INLINE_CAP = 6000
_STATE_CAP = 8000
# Long scans materialise many windows per call. On a small box (8 GB) the
# multilingual checkpoint plus a wide window set exhausts RAM and the process
# is OOM-killed (exit 137), so the state budget here is deliberately modest.
# Raise it only on a machine with headroom.
_LONG_STATE_CAP = 40_000
# States/windows per forward pass on the batch and long paths. Upstream
# defaults are unbounded — predict_batch(batch_size=None) sends EVERY state
# in one pass and predict_long does the same with every window — so peak RAM
# grows with the item count instead of the item size: 37 english windows in
# one pass peaked at the 2.8 GB checkpoint-load plateau on a 7.7 GB box, and
# the same shape on the multilingual checkpoint (max_len 8192, ~16x the
# sequence width) is what OOM-killed the test box. Chunking at 8 keeps peak
# activation memory flat; total time is roughly unchanged.
_FORWARD_BATCH = 8

_ROUTER = None

# SHA-256 pins for the upstream checkpoint weights, verified by laya >= 0.3.22
# before any weight file is parsed or executed (fail-closed: a tampered or
# silently-updated checkpoint raises instead of loading). Digests computed from
# convaiinnovations/laya @ 55cf4c4e (2026-09-30). typed-decisions ships from a
# separate repo and is left unpinned — it is never auto-loaded. setdefault so a
# user-provided LAYA_SHA256_DIGESTS always wins.
import json as _json
import os as _os

_os.environ.setdefault(
    "LAYA_SHA256_DIGESTS",
    _json.dumps(
        {
            "english": {
                "model.safetensors": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
            },
            "multilingual": {
                "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
            },
        }
    ),
)

# ---------------------------------------------------------------------------
# dependency bootstrap
# ---------------------------------------------------------------------------
# laya + torch are deliberately NOT declared in plugin.yaml: torch's PyPI
# wheel drags the whole CUDA runtime in, and the CPU wheel only exists on
# download.pytorch.org. They live in a dedicated venv instead
# (scripts/install_laya_env.sh), and this adds its site-packages to
# sys.path so `import laya` works inside the Hermes interpreter. That keeps
# the install durable across `hermes update` / managed-env regeneration,
# which silently drops undeclared third-party packages.
_LAYA_VENV = os.path.expanduser("~/.hermes/laya-venv")


def _venv_site_packages() -> List[str]:
    """site-packages dirs of the dedicated laya venv, best match first."""
    lib = os.path.join(_LAYA_VENV, "lib")
    if not os.path.isdir(lib):
        return []
    cands: List[str] = []
    for entry in sorted(os.listdir(lib)):
        if entry.startswith("python"):
            cands.append(os.path.join(lib, entry, "site-packages"))
    # running interpreter's version first — a torch wheel from another
    # CPython minor version cannot be imported.
    mine = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cands.sort(key=lambda p: (not p.startswith(os.path.join(lib, mine)), p))
    return [p for p in cands if os.path.isdir(p)]


def _bootstrap_laya_path() -> bool:
    """Make the dedicated laya venv importable from the running interpreter."""
    if "laya" in sys.modules:
        return True
    try:
        import laya  # noqa: F401

        return True
    except Exception:
        pass
    for sp in _venv_site_packages():
        if sp not in sys.path:
            # append, not insert: packages already present in the Hermes
            # environment keep priority, so numpy & friends are never clobbered.
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
            raise ImportError(
                "laya is not importable; run scripts/install_laya_env.sh to create "
                f"{_LAYA_VENV} with laya + CPU torch"
            )
        from laya import Router

        _ROUTER = Router()
    return _ROUTER


def _check() -> bool:
    try:
        return _bootstrap_laya_path()
    except Exception:
        return False


def _ok(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "e": msg}, separators=(",", ":"), ensure_ascii=False)


def _parse_json(s: Any, name: str) -> Tuple[bool, Any, str]:
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


def _parse_json_any(s: Any, name: str) -> Tuple[bool, Any, str]:
    """Like _parse_json but accepts arrays too (batch states)."""
    if isinstance(s, (dict, list)):
        return True, s, ""
    if not isinstance(s, str) or not s.strip():
        return False, None, f"{name} must be a non-empty JSON string"
    try:
        v = json.loads(s)
    except Exception as exc:
        return False, None, f"{name} is not valid JSON: {exc}"
    if not isinstance(v, (dict, list)) or not v:
        return False, None, f"{name} must be a non-empty JSON object or array"
    return True, v, ""


def _validate_questions(qs: dict, max_opts: int = _MAX_OPTS) -> Tuple[bool, str]:
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
            if len(crit) > max_opts:
                return (
                    False,
                    f"choice question '{qid}' has {len(crit)} options "
                    f"(> {max_opts}); use action=tournament to narrow it by elimination",
                )
        elif q["type"] == "score":
            if not isinstance(crit, list) or not crit:
                return False, (
                    f"score question '{qid}' needs criteria as a non-empty list of "
                    "level descriptions (index 0 = lowest)"
                )
    return True, ""


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
                    v = {kk: _round(vv) for kk, vv in v.items()}
                else:
                    v = _round(v)
                c[k] = v
        probs = a.get("probabilities")
        if isinstance(probs, dict):
            items = sorted(probs.items(), key=lambda kv: -float(kv[1]))[:12]
            c["p"] = {str(kk): _round(vv) for kk, vv in items}
        out[qid] = c or {k: str(v)[:200] for k, v in list(a.items())[:6]}
    return out


def _predict_kwargs(model: str, task: str = "", lang: str = "",
                    min_confidence: Optional[float] = None,
                    max_len: Optional[int] = None) -> Dict[str, Any]:
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
    return kw


def _slim(a: dict) -> dict:
    """Fallback projection when a payload exceeds the inline cap."""
    return {k: {kk: v.get(kk) for kk in ("choice", "noul", "score", "confidence",
                                         "answer_confidence")}
            for k, v in a.items() if isinstance(v, dict)}


def _run_predict(state: Any, questions: dict, kw: Dict[str, Any]) -> str:
    try:
        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    try:
        res = router.predict(state, questions, **kw)
    except TypeError:
        # older laya without min_confidence / max_len
        trimmed = {k: v for k, v in kw.items() if k not in ("min_confidence", "max_len")}
        try:
            res = router.predict(state, questions, **trimmed)
        except Exception as exc2:
            return _err(f"predict failed: {str(exc2)[:300]}")
    except Exception as exc:
        return _err(f"predict failed: {str(exc)[:300]}")
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
    s = _ok(out)
    if len(s) > _INLINE_CAP:
        out["a"] = _slim(out["a"])
        s = _ok(out)
        if len(s) > _INLINE_CAP:
            return _err(f"answer payload too large ({len(s)} chars); ask fewer questions")
    return s


def _state_arg(args: dict) -> Tuple[bool, Any, str]:
    raw = args.get("state", args.get("text", ""))
    if isinstance(raw, (dict, list)):
        return True, raw, ""
    if not isinstance(raw, str) or not raw.strip():
        return False, None, "state/text must be non-empty (text or JSON object)"
    stripped = raw.strip()
    if stripped[:1] in ("{", "["):
        try:
            v = json.loads(stripped)
        except Exception:
            v = None
        if isinstance(v, (dict, list)):
            return True, v, ""
    return True, raw, ""


def _preset_questions(name: str, categories: Optional[dict] = None) -> Tuple[dict, str]:
    """Return (questions, state_field) for a preset.

    laya presets name the state field they read (`message` for triage,
    `body` for email, `prompt` for guard, `post` for moderation,
    `request` for routemodel). Handing raw text over under the wrong key
    silently degrades the answer, so the field is resolved from the preset
    itself via laya.presets.state_field().
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
        raise ValueError(f"unknown preset '{name}'")
    return q, _presets.state_field(q) or "text"


def _wrap_state(state: Any, field: str) -> Any:
    """Put plain text under the field the preset reads; pass dicts through."""
    if isinstance(state, dict):
        return state
    return {field: state}


def _action_route(state_raw: Any) -> str:
    ok, st, emsg = _state_arg({"state": state_raw})
    if not ok:
        return _err(emsg)
    if isinstance(st, str) and len(st) > _STATE_CAP:
        return _err(f"state too long for route probe (>{_STATE_CAP} chars)")
    try:
        router = _get_router()
        d = router.route(st, {})
    except Exception as exc:
        return _err(f"route failed: {exc}")
    det = d.get("detection") or {}
    return _ok({"ok": True, "m": d.get("model"), "repo": d.get("repo"),
                "why": d.get("reason"),
                "lang": det.get("language"), "script": det.get("script")})


def _action_detect(state_raw: Any) -> str:
    ok, st, emsg = _state_arg({"state": state_raw})
    if not ok:
        return _err(emsg)
    try:
        import laya

        d = laya.detect_language(st, explain=True)
    except Exception as exc:
        return _err(f"detect failed: {exc}")
    return _ok({"ok": True, "script": d.get("script"), "lang": d.get("language"),
                "en": d.get("is_english"), "non_latin": d.get("non_latin_fraction"),
                "undecided": d.get("language_undecided"),
                "why": (d.get("diagnostic") or {}).get("reason")})


def _action_schema(state: Any, schema: Any, kw: Dict[str, Any]) -> str:
    """JSON-schema-driven typed values (enum choice / integer level / bool)."""
    try:
        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    try:
        res = router.decide(state, schema=schema, return_details=True, **kw)
    except Exception as exc:
        return _err(f"schema decide failed: {str(exc)[:300]}")
    ms = int(1000 * (time.time() - t0))
    conf = getattr(res, "confidence", {}) or {}
    aconf = getattr(res, "answer_confidence", {}) or {}
    return _ok({"ok": True, "v": getattr(res, "values", None),
                "c": {k: _round(v) for k, v in conf.items()},
                "ac": {k: _round(v) for k, v in aconf.items()},
                "ms": ms})


def _action_batch(states: List[Any], questions: dict, kw: Dict[str, Any]) -> str:
    try:
        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    # predict_batch takes routing hints per request, not per call.
    per_req = {k: v for k, v in kw.items()
               if k in ("model", "task", "lang", "max_len", "head_max_len")}
    # batch_size bounds states per forward pass (see _FORWARD_BATCH): without
    # it the router sends all _MAX_BATCH states in a single pass.
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
        return _err(f"batch failed: {str(exc)[:300]}")
    ms = int(1000 * (time.time() - t0))
    out = []
    for res in results:
        routing = (res or {}).get("routing", {}) or {}
        out.append({"a": _compact_answers((res or {}).get("answers", {}) or {}),
                    "m": routing.get("model")})
    payload = {"ok": True, "n": len(out), "batch": out, "ms": ms}
    s = _ok(payload)
    if len(s) > _INLINE_CAP:
        for item in payload["batch"]:
            item["a"] = _slim(item["a"])
        s = _ok(payload)
        if len(s) > _INLINE_CAP:
            return _err(f"batch payload too large ({len(s)} chars); fewer states or questions")
    return s


def _action_long(state: Any, questions: dict, kw: Dict[str, Any],
                 window: Optional[int], stride: Optional[int]) -> str:
    try:
        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    # predict_long takes routing hints + window/stride; it has no min_confidence.
    # batch_size caps windows per forward pass (see _FORWARD_BATCH): upstream's
    # window_batch_cap only chunks when a small window multiplied the count, so
    # the default-window scan would otherwise run every window in one pass.
    ckw = {k: v for k, v in kw.items() if k in ("model", "task", "lang")}
    try:
        wkw: Dict[str, Any] = {"batch_size": _FORWARD_BATCH}
        if window:
            wkw["window"] = window
        if stride:
            wkw["stride"] = stride
        res = router.predict_long(state, questions, **wkw, **ckw)
    except Exception as exc:
        return _err(f"long predict failed: {str(exc)[:300]}")
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
    return _ok({"ok": True, "a": answers, "r": {"m": routing.get("model")},
                "win": usage.get("windows"), "ms": ms})


def _action_tournament(state: Any, questions: dict, kw: Dict[str, Any],
                       group_size: int) -> str:
    try:
        import laya

        router = _get_router()
    except Exception as exc:
        return _err(f"router init failed: {exc}")
    t0 = time.time()
    try:
        res = laya.predict_tournament(router, state, questions,
                                      group_size=group_size, **kw)
    except Exception as exc:
        return _err(f"tournament failed: {str(exc)[:300]}")
    ms = int(1000 * (time.time() - t0))
    routing = res.get("routing", {}) or {}
    tour = res.get("tournament") or {}
    narrow = {qid: {"n": m.get("n"), "rounds": m.get("rounds"), "labels": m.get("labels")}
              for qid, m in tour.items()}
    return _ok({"ok": True, "a": _compact_answers(res.get("answers", {}) or {}),
                "r": {"m": routing.get("model")}, "t": narrow, "ms": ms})


def _handler(args: dict) -> str:
    if not _check():
        return _err("laya not installed: run scripts/install_laya_env.sh "
                    "(creates ~/.hermes/laya-venv with laya + CPU torch)")
    action = str(args.get("action", "decide") or "decide").lower()
    model = str(args.get("model", "auto") or "auto").lower()
    if model not in _MODELS:
        return _err(f"unknown model '{model}'; use one of {list(_MODELS)}")
    task = str(args.get("task", "") or "")
    lang = str(args.get("lang", "") or "")
    mc = args.get("min_confidence")
    try:
        mc = float(mc) if mc is not None else None
    except Exception:
        return _err("min_confidence must be a number")
    max_len = args.get("max_len")
    try:
        max_len = int(max_len) if max_len is not None else None
    except Exception:
        return _err("max_len must be an integer")

    if action == "status":
        try:
            import laya

            router = _get_router()
        except Exception as exc:
            return _err(f"status failed: {exc}")
        from pathlib import Path

        hf = Path.home() / ".cache" / "huggingface"
        return _ok({"ok": True, "v": getattr(laya, "__version__", "?"),
                    "loaded": list(router.loaded or []),
                    "revs": dict(router.loaded_revisions or {}),
                    "hf_cache": hf.exists()})

    if action == "detect":
        return _action_detect(args.get("state", args.get("text", "")))

    if action == "route":
        return _action_route(args.get("state", args.get("text", "")))

    if action == "batch":
        ok, states, emsg = _parse_json_any(args.get("states"), "states")
        if not ok:
            return _err(emsg)
        if not isinstance(states, list) or not states:
            return _err("states must be a non-empty JSON array")
        if len(states) > _MAX_BATCH:
            return _err(f"too many states ({len(states)} > {_MAX_BATCH})")
        for s in states:
            slen = len(s) if isinstance(s, str) else len(json.dumps(s, ensure_ascii=False))
            if slen > _STATE_CAP:
                return _err(f"batch state too long (>{_STATE_CAP} chars); "
                            "pass the relevant excerpt per state")
        okq, qs, emsg = _parse_json(args.get("questions", ""), "questions")
        if not okq:
            return _err(emsg)
        okv, vmsg = _validate_questions(qs)
        if not okv:
            return _err(vmsg)
        return _action_batch(states, qs, _predict_kwargs(model, task, lang, mc, max_len))

    ok, state, emsg = _state_arg(args)
    if not ok:
        return _err(emsg)

    if action == "schema":
        oks, schema, emsg = _parse_json(args.get("schema", ""), "schema")
        if not oks:
            return _err(emsg)
        return _action_schema(state, schema, _predict_kwargs(model, task, lang, mc, max_len))

    if action == "long":
        okq, qs, emsg = _parse_json(args.get("questions", ""), "questions")
        if not okq:
            return _err(emsg)
        okv, vmsg = _validate_questions(qs)
        if not okv:
            return _err(vmsg)
        if isinstance(state, str) and len(state) > _LONG_STATE_CAP:
            return _err(f"state too long for long scan (>{_LONG_STATE_CAP} chars)")
        window = args.get("window")
        stride = args.get("stride")
        try:
            window = int(window) if window is not None else None
            stride = int(stride) if stride is not None else None
        except Exception:
            return _err("window/stride must be integers")
        return _action_long(state, qs, _predict_kwargs(model, task, lang, mc, max_len),
                            window, stride)

    if action == "tournament":
        okq, qs, emsg = _parse_json(args.get("questions", ""), "questions")
        if not okq:
            return _err(emsg)
        okv, vmsg = _validate_questions(qs, max_opts=_MAX_TOURNAMENT_OPTS)
        if not okv:
            return _err(vmsg)
        group_size = args.get("group_size", 16)
        try:
            group_size = int(group_size)
        except Exception:
            return _err("group_size must be an integer")
        if group_size < 2:
            return _err("group_size must be >= 2")
        return _action_tournament(state, qs, _predict_kwargs(model, task, lang, mc, max_len),
                                  group_size)

    if action in _PRESETS or action == "decide":
        if isinstance(state, str) and len(state) > _STATE_CAP:
            return _err(f"state too long (>{_STATE_CAP} chars); pass the relevant excerpt")
        if action == "decide":
            okq, qs, emsg = _parse_json(args.get("questions", ""), "questions")
            if not okq:
                return _err(emsg)
            okv, vmsg = _validate_questions(qs)
            if not okv:
                return _err(vmsg)
            return _run_predict(state, qs, _predict_kwargs(model, task, lang, mc, max_len))
        try:
            cats = None
            if args.get("categories"):
                okc, cats, emsg = _parse_json(args.get("categories"), "categories")
                if not okc:
                    return _err(emsg)
            qs, field = _preset_questions(action, cats)
        except Exception as exc:
            return _err(str(exc))
        okv, vmsg = _validate_questions(qs)
        if not okv:
            return _err(vmsg)
        st = state
        if action == "email":
            subject = str(args.get("subject", "") or "")
            sender = str(args.get("sender", "") or "")
            if subject or sender:
                try:
                    import laya

                    body = state if isinstance(state, str) else ""
                    st = laya.email_state(subject, body, sender or None)
                except Exception as exc:
                    return _err(f"email state failed: {exc}")
        else:
            st = _wrap_state(state, field)
        return _run_predict(st, qs, _predict_kwargs(model, task, lang, mc, max_len))

    return _err(f"unknown action '{action}'; use decide|schema|batch|long|tournament|"
                "detect|route|triage|guard|moderation|routemodel|email|status")


def _runner(args):
    try:
        return _handler(args if isinstance(args, dict) else {})
    except Exception as exc:  # noqa: BLE001 — tool errors are values
        return _err(f"laya tool error: {exc}")


LAYA_SCHEMA: Dict[str, Any] = {
    "name": "laya",
    "description": "Fast typed decisions (choice/score/noul) in one forward pass, no"
    " hallucination. decide=text+questions JSON; schema=JSON-schema typed values;"
    " batch=many states one pass; long=scan long docs; tournament=narrow >50 options;"
    " detect=language/script (no model load); route=which checkpoint;"
    " triage/guard/moderation/routemodel/email=preset schemas; status=health.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "decide (default), schema, batch, long, tournament, detect,"
                " route, triage, guard, moderation, routemodel, email, status.",
            },
            "state": {
                "type": "string",
                "description": "Text or JSON object to decide over (required except"
                " status/route/detect).",
            },
            "text": {
                "type": "string",
                "description": "Alias for state (plain text).",
            },
            "questions": {
                "type": "string",
                "description": "JSON object of typed questions (choice/score/noul +"
                " instructions + criteria).",
            },
            "schema": {
                "type": "string",
                "description": "JSON Schema for action=schema; returns typed values"
                " (enum/integer level/boolean).",
            },
            "states": {
                "type": "string",
                "description": "JSON array of states for action=batch (one forward pass).",
            },
            "categories": {
                "type": "string",
                "description": "Optional JSON object overriding email preset categories.",
            },
            "subject": {
                "type": "string",
                "description": "Optional email subject (action=email).",
            },
            "sender": {
                "type": "string",
                "description": "Optional email sender (action=email).",
            },
            "model": {
                "type": "string",
                "description": "auto (default, Router picks), english, multilingual,"
                " typed-decisions.",
            },
            "task": {
                "type": "string",
                "description": "Optional routing hint (e.g. typed_decisions).",
            },
            "lang": {
                "type": "string",
                "description": "Optional ISO language code that skips detection.",
            },
            "min_confidence": {
                "type": "number",
                "description": "Confidence gate; answers report abstention"
                " (passed/abstained/unevaluated).",
            },
            "max_len": {
                "type": "integer",
                "description": "Token budget override (multilingual reads up to 8192).",
            },
            "window": {
                "type": "integer",
                "description": "action=long: state tokens per window.",
            },
            "stride": {
                "type": "integer",
                "description": "action=long: token step between windows.",
            },
            "group_size": {
                "type": "integer",
                "description": "action=tournament: labels per elimination round (default 16).",
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
