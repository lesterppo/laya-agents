#!/usr/bin/env python3
"""Offline validation battery for the Laya Hermes tool.

No model download and no network: only the validation, dispatch and guard paths
are exercised. Live inference is in tests/test_laya_live.py.

    python tests/test_laya_tool.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

import laya_tool as t  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append(("PASS" if ok else "FAIL", name, str(detail)[:90]))


def call(args: dict) -> dict:
    return json.loads(t._runner(args))


def _q(obj) -> str:
    return json.dumps(obj)


NOUL = {"type": "noul", "instructions": "Is this true?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which bucket?",
    "criteria": {"a": "first", "b": "second"},
}

# --- validation: state ------------------------------------------------------
for bad in ["", "   ", None]:
    d = call({"action": "decide", "state": bad, "questions": _q({"x": NOUL})})
    check(f"decide rejects state {bad!r}", d.get("ok") is False, d.get("e"))

d = call({"action": "route", "state": ""})
check("route rejects empty state", d.get("ok") is False, d.get("e"))

d = call({"action": "detect", "state": "   "})
check("detect rejects empty state", d.get("ok") is False, d.get("e"))

d = call({"action": "route", "state": "z" * 9000})
check("route rejects oversized state", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "x" * 9000, "questions": _q({"x": NOUL})})
check("decide rejects oversized state", d.get("ok") is False, d.get("e"))

# --- validation: questions --------------------------------------------------
for bad, label in [("", "empty"), ("   ", "whitespace"), ("not json", "non-JSON"),
                   ("[]", "JSON array"), ("{}", "empty object")]:
    d = call({"action": "decide", "state": "hello", "questions": bad})
    check(f"decide rejects questions ({label})", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello",
          "questions": _q({"x": {"type": "bogus", "instructions": "hi"}})})
check("rejects unknown question type", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello", "questions": _q({"x": {"type": "noul"}})})
check("rejects missing instructions", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello",
          "questions": _q({"x": {"type": "noul", "instructions": "   "}})})
check("rejects whitespace-only instructions", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello",
          "questions": _q({"x": {"type": "choice", "instructions": "hi"}})})
check("rejects choice without criteria", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello",
          "questions": _q({f"q{i}": NOUL for i in range(21)})})
check("rejects >20 questions", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "hello",
          "questions": _q({"x": {"type": "choice", "instructions": "hi",
                                 "criteria": {f"o{i}": None for i in range(77)}}})})
check("rejects >50 options (token budget)", d.get("ok") is False, d.get("e"))
check("decide >50 options points at tournament",
      "tournament" in str(d.get("e", "")), d.get("e"))

# tournament accepts the wide set decide rejects
d = call({"action": "tournament", "state": "hello",
          "questions": _q({"x": {"type": "choice", "instructions": "hi",
                                 "criteria": {f"o{i}": None for i in range(77)}}})})
check("tournament accepts >50 options (validation only)",
      d.get("ok") is False and "router init failed" in str(d.get("e", "")) or d.get("ok") is True,
      d.get("e"))

d = call({"action": "tournament", "state": "hello",
          "questions": _q({"x": CHOICE}), "group_size": 1})
check("tournament rejects group_size < 2", d.get("ok") is False, d.get("e"))

# --- validation: new actions ------------------------------------------------
d = call({"action": "schema", "state": "hello", "schema": "not json"})
check("schema rejects bad schema JSON", d.get("ok") is False, d.get("e"))

d = call({"action": "schema", "state": "hello", "schema": "[]"})
check("schema rejects non-object schema", d.get("ok") is False, d.get("e"))

d = call({"action": "batch", "state": "hello", "questions": _q({"x": NOUL})})
check("batch rejects missing states", d.get("ok") is False, d.get("e"))

d = call({"action": "batch", "states": "{}", "questions": _q({"x": NOUL})})
check("batch rejects non-array states", d.get("ok") is False, d.get("e"))

d = call({"action": "batch", "states": _q(["a"] * 100), "questions": _q({"x": NOUL})})
check("batch rejects >64 states", d.get("ok") is False, d.get("e"))

d = call({"action": "long", "state": "hello", "questions": _q({"x": NOUL}),
          "window": "abc"})
check("long rejects non-integer window", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "x", "questions": _q({"x": NOUL}),
          "min_confidence": "high"})
check("rejects non-numeric min_confidence", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "x", "questions": _q({"x": NOUL}),
          "max_len": "big"})
check("rejects non-integer max_len", d.get("ok") is False, d.get("e"))

# --- validation: action / model --------------------------------------------
d = call({"action": "frobnicate", "state": "x"})
check("rejects unknown action", d.get("ok") is False, d.get("e"))

d = call({"action": "decide", "state": "x", "questions": _q({"x": NOUL}),
          "model": "klingon"})
check("rejects unknown model", d.get("ok") is False, d.get("e"))

d = call({"action": "email", "state": "x", "categories": "not json"})
check("rejects bad categories JSON", d.get("ok") is False, d.get("e"))

# --- pure helpers -----------------------------------------------------------
check("check_fn is cheap bool", isinstance(t._check(), bool))
check("_ok yields compact JSON", t._ok({"a": 1}) == '{"a":1}')
check("_err yields ok:false", json.loads(t._err("boom"))["ok"] is False)

ok, val, _ = t._parse_json('{"a":1}', "questions")
check("_parse_json accepts dict", ok and val == {"a": 1})
ok, _, msg = t._parse_json("[1,2]", "questions")
check("_parse_json rejects array", not ok and ("JSON" in msg or "object" in msg))
ok, val, _ = t._parse_json({"a": 1}, "questions")
check("_parse_json passes through dicts", ok and val == {"a": 1})

ok, val, msg = t._parse_json_any('["a","b"]', "states")
check("_parse_json_any accepts array", ok and val == ["a", "b"], msg)
ok, _, msg = t._parse_json_any("{}", "states")
check("_parse_json_any rejects empty object", not ok, msg)

ok, msg = t._validate_questions({"a": CHOICE, "b": NOUL})
check("_validate_questions accepts valid set", ok, msg)
ok, msg = t._validate_questions({"a": {"type": "choice", "instructions": "x", "criteria": []}})
check("_validate_questions rejects list criteria", not ok, msg)
ok, msg = t._validate_questions({"a": {"type": "score", "instructions": "x", "criteria": {}}})
check("_validate_questions rejects dict criteria for score", not ok, msg)
ok, msg = t._validate_questions({"a": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]}})
check("_validate_questions accepts list criteria for score", ok, msg)
ok, msg = t._validate_questions(
    {"a": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(77)}}},
    max_opts=t._MAX_TOURNAMENT_OPTS)
check("_validate_questions honours max_opts override", ok, msg)

# 0.4.x answer shape: probabilities / answer_confidence / legend / abstention
comp = t._compact_answers({"q": {"type": "choice", "choice": "a",
                                 "confidence": 0.123456,
                                 "answer_confidence": 0.98765,
                                 "probabilities": {"a": 0.9, "b": 0.1},
                                 "legend": {"0": "low"},
                                 "abstention": "passed",
                                 "action": {"act_probability": 1.0},
                                 "extra_ignored": "noise"}})
check("_compact_answers keeps known keys only",
      comp["q"].get("choice") == "a" and "extra_ignored" not in comp["q"]
      and "action" not in comp["q"])
check("_compact_answers rounds floats", comp["q"]["confidence"] == 0.1235)
check("_compact_answers keeps answer_confidence",
      comp["q"]["answer_confidence"] == 0.9877)
check("_compact_answers keeps probabilities",
      comp["q"].get("p", {}).get("a") == 0.9)
check("_compact_answers keeps legend", comp["q"].get("legend") == {"0": "low"})
check("_compact_answers keeps abstention", comp["q"].get("abstention") == "passed")

# --- preset resolution (imports laya, no model load) ------------------------
try:
    for preset in ("triage", "guard", "moderation", "routemodel", "email"):
        qs, field = t._preset_questions(preset)
        check(f"preset {preset} resolves", isinstance(qs, dict) and bool(qs))
        check(f"preset {preset} names its state field",
              isinstance(field, str) and bool(field), field)
    ok, msg = t._validate_questions(t._preset_questions("triage")[0])
    check("triage preset passes validation", ok, msg)
    ok, msg = t._validate_questions(t._preset_questions("guard")[0])
    check("guard preset passes validation", ok, msg)
    # the field is what the preset actually reads
    _, triage_field = t._preset_questions("triage")
    check("triage reads `message`", triage_field == "message", triage_field)
    _, email_field = t._preset_questions("email")
    check("email reads `body`", email_field == "body", email_field)
except ImportError:
    check("presets resolve (laya installed)", False, "laya not importable — skipped")

# --- memory bounds: batch/long chunking (OOM regression) --------------------
# Upstream predict_batch / predict_long default to ONE forward pass over
# every state/window, so peak RAM grows with the item count and OOM-kills
# small boxes. The tool must always pass batch_size=_FORWARD_BATCH, and
# batch states are capped per state like a single state is.
d = call({"action": "batch", "states": _q(["x" * 9000]), "questions": _q({"x": NOUL})})
check("batch rejects per-state >8000 chars", d.get("ok") is False, d.get("e"))

check("_sanitize_proxy_env is installed", callable(t._sanitize_proxy_env))
import os as _os

_old_np = _os.environ.get("NO_PROXY")
_os.environ["NO_PROXY"] = "localhost,[::1]"
t._sanitize_proxy_env()
check("proxy sanitize replaces bracketed IPv6 no_proxy",
      _os.environ["NO_PROXY"] == "localhost,127.0.0.1", _os.environ["NO_PROXY"])
_os.environ["NO_PROXY"] = "localhost,example.internal"
t._sanitize_proxy_env()
check("proxy sanitize leaves a clean no_proxy untouched",
      _os.environ["NO_PROXY"] == "localhost,example.internal")
if _old_np is None:
    _os.environ.pop("NO_PROXY", None)
else:
    _os.environ["NO_PROXY"] = _old_np


class _StubRouter:
    """Captures the kwargs the bounded paths forward to the router."""

    loaded: list = []
    loaded_revisions: dict = {}

    def __init__(self):
        self.batch_kw: dict = {}
        self.long_kw: dict = {}

    def predict_batch(self, reqs, **kw):
        self.batch_kw = kw
        return [{"answers": {}, "routing": {"model": "english"}} for _ in reqs]

    def predict_long(self, state, questions, **kw):
        self.long_kw = kw
        return {"answers": {}, "routing": {"model": "english"},
                "usage": {"windows": 3}}


_stub = _StubRouter()
_saved_router = t._ROUTER
t._ROUTER = _stub  # type: ignore[assignment]
try:
    d = call({"action": "batch", "states": _q(["a", "b", "c"]),
              "questions": _q({"x": NOUL})})
    check("batch (stub) ok", d.get("ok") is True and d.get("n") == 3, d.get("e"))
    check("batch passes batch_size=_FORWARD_BATCH",
          _stub.batch_kw.get("batch_size") == t._FORWARD_BATCH, _stub.batch_kw)
    d = call({"action": "long", "state": "hello world", "questions": _q({"x": NOUL})})
    check("long (stub) ok", d.get("ok") is True and d.get("win") == 3, d.get("e"))
    check("long passes batch_size=_FORWARD_BATCH",
          _stub.long_kw.get("batch_size") == t._FORWARD_BATCH, _stub.long_kw)
finally:
    t._ROUTER = _saved_router

# --- report -----------------------------------------------------------------
for status, name, detail in RESULTS:
    print(f"[{status}] {name:52s} {detail}")
fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} passed")
sys.exit(1 if fails else 0)
