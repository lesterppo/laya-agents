#!/usr/bin/env python3
"""Offline validation battery for the Muse-native Laya module.

No model download and no network: only validation, dispatch, guard and
batch-plumbing paths are exercised (live inference mirrors the Hermes
battery in tests/test_laya_live.py, which covers the shared semantics).

    python tests/test_laya_muse.py
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "muse"))

import laya_muse as lm  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append(("PASS" if ok else "FAIL", name, str(detail)[:90]))


def raises(name: str, fn, needle: str = "") -> None:
    try:
        fn()
    except lm.LayaError as exc:
        check(name, needle in str(exc) if needle else True, exc)
        return
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception: {exc}")
        return
    check(name, False, "no LayaError raised")


def _q(obj) -> str:
    return json.dumps(obj)


NOUL = {"type": "noul", "instructions": "Is this true?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which bucket?",
    "criteria": {"a": "first", "b": "second"},
}

# --- validation: state / action / model -------------------------------------
for bad in ["", "   ", None]:
    raises(f"predict rejects state {bad!r}",
           lambda b=bad: lm.predict("decide", state=b, questions=_q({"x": NOUL})))
raises("route rejects empty state", lambda: lm.predict("route", state=""))
raises("detect rejects empty state", lambda: lm.predict("detect", state="   "))
raises("route rejects oversized state", lambda: lm.predict("route", state="z" * 9000))
raises("decide rejects oversized state",
       lambda: lm.predict("decide", state="x" * 9000, questions=_q({"x": NOUL})))
raises("rejects unknown action", lambda: lm.predict("frobnicate", state="x"))
raises("rejects unknown model",
       lambda: lm.predict("decide", state="x", questions=_q({"x": NOUL}), model="klingon"))

# --- validation: questions ---------------------------------------------------
for bad, label in [("", "empty"), ("not json", "non-JSON"), ("[]", "array"), ("{}", "empty")]:
    raises(f"decide rejects questions ({label})",
           lambda b=bad: lm.predict("decide", state="hello", questions=b))
raises("rejects unknown question type",
       lambda: lm.predict("decide", state="hello",
                          questions=_q({"x": {"type": "bogus", "instructions": "hi"}})))
raises("rejects missing instructions",
       lambda: lm.predict("decide", state="hello", questions=_q({"x": {"type": "noul"}})))
raises("rejects choice without criteria",
       lambda: lm.predict("decide", state="hello",
                          questions=_q({"x": {"type": "choice", "instructions": "hi"}})))
raises("rejects >20 questions",
       lambda: lm.predict("decide", state="hello",
                          questions=_q({f"q{i}": NOUL for i in range(21)})))
raises("rejects >50 options (token budget)",
       lambda: lm.predict("decide", state="hello",
                          questions=_q({"x": {"type": "choice", "instructions": "hi",
                                              "criteria": {f"o{i}": None for i in range(77)}}})),
       "tournament")
raises("tournament rejects group_size < 2",
       lambda: lm.predict("tournament", state="hello", questions=_q({"x": CHOICE}),
                          group_size=1))
raises("schema rejects bad schema JSON",
       lambda: lm.predict("schema", state="hello", schema="not json"))
raises("schema rejects non-object schema",
       lambda: lm.predict("schema", state="hello", schema="[]"))
raises("batch rejects missing states",
       lambda: lm.predict("batch", questions=_q({"x": NOUL})))
raises("batch rejects non-array states",
       lambda: lm.predict("batch", states="{}", questions=_q({"x": NOUL})))
raises("batch rejects >64 states",
       lambda: lm.predict("batch", states=_q(["a"] * 100), questions=_q({"x": NOUL})))
raises("batch rejects per-state >8000 chars",
       lambda: lm.predict("batch", states=_q(["x" * 9000]), questions=_q({"x": NOUL})))
raises("long rejects non-integer window",
       lambda: lm.predict("long", state="hello", questions=_q({"x": NOUL}), window="abc"))
raises("long rejects oversized state",
       lambda: lm.predict("long", state="x" * 40001, questions=_q({"x": NOUL})))
raises("rejects non-numeric min_confidence",
       lambda: lm.predict("decide", state="x", questions=_q({"x": NOUL}),
                          min_confidence="high"))
raises("rejects non-integer max_len",
       lambda: lm.predict("decide", state="x", questions=_q({"x": NOUL}), max_len="big"))
raises("rejects bad categories JSON",
       lambda: lm.predict("email", state="x", categories="not json"))

# --- pure helpers --------------------------------------------------------------
comp = lm._compact_answers({"q": {"type": "choice", "choice": "a",
                                  "confidence": 0.123456,
                                  "answer_confidence": 0.98765,
                                  "probabilities": {"a": 0.9, "b": 0.1},
                                  "legend": {"0": "low"},
                                  "abstention": "passed",
                                  "extra_ignored": "noise"}})
check("_compact_answers keeps known keys only",
      comp["q"].get("choice") == "a" and "extra_ignored" not in comp["q"])
check("_compact_answers rounds floats", comp["q"]["confidence"] == 0.1235)
check("_compact_answers keeps answer_confidence", comp["q"]["answer_confidence"] == 0.9877)
check("_compact_answers keeps probabilities", comp["q"].get("p", {}).get("a") == 0.9)
check("_compact_answers keeps legend strings", comp["q"].get("legend") == {"0": "low"})
check("_compact_answers keeps abstention", comp["q"].get("abstention") == "passed")

try:
    lm._validate_questions({"a": CHOICE, "b": NOUL})
    check("_validate_questions accepts valid set", True)
except lm.LayaError as exc:
    check("_validate_questions accepts valid set", False, exc)
try:
    lm._validate_questions({"a": {"type": "score", "instructions": "x", "criteria": {}}})
    check("_validate_questions rejects dict criteria for score", False)
except lm.LayaError:
    check("_validate_questions rejects dict criteria for score", True)
try:
    lm._validate_questions({"a": {"type": "score", "instructions": "x",
                                  "criteria": ["lo", "hi"]}})
    check("_validate_questions accepts list criteria for score", True)
except lm.LayaError as exc:
    check("_validate_questions accepts list criteria for score", False, exc)
try:
    lm._validate_questions(
        {"a": {"type": "choice", "instructions": "x",
               "criteria": {f"o{i}": None for i in range(77)}}},
        max_opts=lm._MAX_TOURNAMENT_OPTS)
    check("_validate_questions honours max_opts override", True)
except lm.LayaError as exc:
    check("_validate_questions honours max_opts override", False, exc)

check("_parse_json_any accepts array", lm._parse_json_any('["a","b"]', "states") == ["a", "b"])
check("_predict_kwargs omits auto model", lm._predict_kwargs("auto") == {})
check("_predict_kwargs passes explicit model",
      lm._predict_kwargs("english", min_confidence=0.85) ==
      {"model": "english", "min_confidence": 0.85})

import os as _os

_old_np = _os.environ.get("NO_PROXY")
_os.environ["NO_PROXY"] = "localhost,[::1]"
lm._sanitize_proxy_env()
check("proxy sanitize replaces bracketed IPv6 no_proxy",
      _os.environ["NO_PROXY"] == "localhost,127.0.0.1", _os.environ["NO_PROXY"])
_os.environ["NO_PROXY"] = "localhost,example.internal"
lm._sanitize_proxy_env()
check("proxy sanitize leaves a clean no_proxy untouched",
      _os.environ["NO_PROXY"] == "localhost,example.internal")
if _old_np is None:
    _os.environ.pop("NO_PROXY", None)
else:
    _os.environ["NO_PROXY"] = _old_np

# --- preset resolution (imports laya, no model load) ---------------------------
try:
    for preset in ("triage", "guard", "moderation", "routemodel", "email"):
        qs, field = lm._preset_questions(preset)
        check(f"preset {preset} resolves", isinstance(qs, dict) and bool(qs))
        check(f"preset {preset} names its state field",
              isinstance(field, str) and bool(field), field)
    _, triage_field = lm._preset_questions("triage")
    check("triage reads `message`", triage_field == "message", triage_field)
    _, email_field = lm._preset_questions("email")
    check("email reads `body`", email_field == "body", email_field)
except Exception as exc:  # noqa: BLE001
    check("presets resolve (laya installed)", False, f"laya not importable: {exc}")

# --- stub router: bounded paths + batch plumbing --------------------------------
class _StubRouter:
    loaded: list = []
    loaded_revisions: dict = {}

    def __init__(self):
        self.batch_kw: dict = {}
        self.long_kw: dict = {}

    def route(self, state, questions):
        return {"model": "english", "repo": "stub", "reason": "stub",
                "detection": {"language": "en", "script": "latin"}}

    def predict_batch(self, reqs, **kw):
        self.batch_kw = kw
        return [{"answers": {}, "routing": {"model": "english"}} for _ in reqs]

    def predict_long(self, state, questions, **kw):
        self.long_kw = kw
        return {"answers": {}, "routing": {"model": "english"},
                "usage": {"windows": 3}}

    def decide(self, state, schema=None, return_details=False, **kw):
        class R:
            values = {"department": "billing"}
            confidence = {"department": 0.9}
            answer_confidence = {"department": 0.95}

        return R()


_stub = _StubRouter()
_saved_router = lm._ROUTER
lm._ROUTER = _stub  # type: ignore[assignment]
try:
    d = lm.predict("route", state="hello")
    check("route (stub) ok", d.get("ok") is True and d.get("m") == "english", d)
    check("route (stub) reports detection", d.get("script") == "latin", d)
    d = lm.predict("batch", states=_q(["a", "b", "c"]), questions=_q({"x": NOUL}))
    check("batch (stub) ok", d.get("ok") is True and d.get("n") == 3, d)
    check("batch passes batch_size=_FORWARD_BATCH",
          _stub.batch_kw.get("batch_size") == lm._FORWARD_BATCH, _stub.batch_kw)
    d = lm.predict("long", state="hello world", questions=_q({"x": NOUL}))
    check("long (stub) ok", d.get("ok") is True and d.get("win") == 3, d)
    check("long passes batch_size=_FORWARD_BATCH",
          _stub.long_kw.get("batch_size") == lm._FORWARD_BATCH, _stub.long_kw)
    d = lm.predict("schema", state="hello", schema=_q({"type": "object"}))
    check("schema (stub) returns typed values",
          d.get("ok") is True and d.get("v", {}).get("department") == "billing", d)
    d = lm.predict("tournament", state="hello",
                   questions=_q({"x": {"type": "choice", "instructions": "hi",
                                       "criteria": {f"o{i}": None for i in range(77)}}}))
    check("tournament accepts >50 options (validation only)",
          d.get("ok") is False and "tournament failed" in str(d.get("e", "")), d.get("e"))

    res = lm.run_batch([
        {"action": "route", "state": "hello"},
        {"action": "bogus", "state": "x"},
        "not-a-dict",
        {"action": "decide", "state": "x"},
    ])
    check("run_batch preserves order and length", len(res) == 4, len(res))
    check("run_batch item 0 ok", res[0].get("ok") is True, res[0])
    check("run_batch bad action is per-item error", res[1].get("ok") is False, res[1])
    check("run_batch non-dict is per-item error", res[2].get("ok") is False, res[2])
    check("run_batch validation error is per-item", res[3].get("ok") is False, res[3])

    # CLI batch mode: invalid JSON lines stay in place, order preserved
    lines = "\n".join([
        _q({"action": "route", "state": "hello"}),
        "not json",
        _q({"action": "bogus", "state": "x"}),
    ])
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(lines)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = lm.main(["--batch", "-"])
    finally:
        sys.stdin = old_stdin
    out_lines = [json.loads(x) for x in buf.getvalue().strip().splitlines()]
    check("CLI --batch returns 0", rc == 0, rc)
    check("CLI --batch emits one line per input", len(out_lines) == 3, len(out_lines))
    check("CLI --batch order preserved",
          out_lines[0].get("ok") is True and out_lines[1].get("ok") is False
          and out_lines[2].get("ok") is False, out_lines)
finally:
    lm._ROUTER = _saved_router

# --- report ----------------------------------------------------------------------
for status, name, detail in RESULTS:
    print(f"[{status}] {name:52s} {detail}")
fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} passed")
sys.exit(1 if fails else 0)
