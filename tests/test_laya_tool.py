#!/usr/bin/env python3
"""Offline validation battery for the Laya Hermes tool.

No model download and no network: only the validation, dispatch and guard paths
are exercised. Live inference is documented in skills/laya-decisions/SKILL.md.

    python tests/test_laya_tool.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

import laya_tool as t  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
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
check("_parse_json rejects array", not ok and "JSON" in msg or "object" in msg)
ok, val, _ = t._parse_json({"a": 1}, "questions")
check("_parse_json passes through dicts", ok and val == {"a": 1})

ok, msg = t._validate_questions({"a": CHOICE, "b": NOUL})
check("_validate_questions accepts valid set", ok, msg)
ok, msg = t._validate_questions({"a": {"type": "choice", "instructions": "x", "criteria": []}})
check("_validate_questions rejects list criteria", not ok, msg)

comp = t._compact_answers({"q": {"type": "choice", "choice": "a",
                                 "confidence": 0.123456,
                                 "probabilities": {"a": 0.9, "b": 0.1},
                                 "extra_ignored": "noise"}})
check("_compact_answers keeps known keys only",
      comp["q"].get("choice") == "a" and "extra_ignored" not in comp["q"])
check("_compact_answers rounds floats", comp["q"]["confidence"] == 0.1235)

# --- preset resolution (imports laya, no model load) ------------------------
try:
    for preset in ("triage", "guard", "moderation", "routemodel", "email"):
        qs = t._preset_questions(preset)
        check(f"preset {preset} resolves", isinstance(qs, dict) and bool(qs))
    ok, msg = t._validate_questions(t._preset_questions("triage"))
    check("triage preset passes validation", ok, msg)
    ok, msg = t._validate_questions(t._preset_questions("guard"))
    check("guard preset passes validation", ok, msg)
except ImportError:
    check("presets resolve (laya installed)", False, "laya not importable — skipped")

# --- report -----------------------------------------------------------------
for status, name, detail in RESULTS:
    print(f"[{status}] {name:46s} {detail}")
fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} passed")
sys.exit(1 if fails else 0)
