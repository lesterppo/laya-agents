---
name: laya-decisions
description: Use for fast typed text decisions without an LLM call.
version: 2.0.0
author: Peter (lesterppo)
license: MIT
metadata:
  hermes:
    tags: [decisions, classification, guardrails, triage]
    related_skills: [tool-reliability-audit]
---

# Laya System 1 Decisions (native `laya` tool, toolset `laya`)

## When to Use

Use when a task needs a fast typed judgment (classify, triage, moderate,
guardrail-check, route) over text without spending an LLM call — ticket
triage, prompt-guardrails, content moderation, model routing, email sorting.

Non-autoregressive decision engine (NandhaKishorM/laya, Apache 2.0): one
forward pass returns calibrated choice/score/noul answers — nothing to
parse, nothing hallucinated. Router picks english / multilingual /
typed-decisions per request from sub-ms script detection.

Targets upstream laya **0.4.14**.

## Tool actions

- `decide` — state/text + `questions` JSON (each: type choice|score|noul +
  instructions + criteria). `model` defaults to `auto` (Router picks).
- `schema` — a JSON Schema in, typed values out: enum → label, bounded int →
  level, boolean → bool, with per-field confidence.
- `batch` — many states, ONE forward pass, input order preserved.
- `long` — windowed scan of a long document; reports the deciding window index.
- `tournament` — narrows a wide choice (>50 options) by elimination rounds.
- `detect` — language + script, NO model load (sub-ms).
- `route` — which checkpoint, NO model load (sub-ms). Always safe, works
  before any download.
- Presets (no schema needed): `triage` `guard` `moderation` `routemodel`
  `email`. `status` = version + loaded models + pinned revisions.

## Answer shape (0.4.x)

Two confidences per answer, answering different questions:

- `confidence` — confidence in the **decision**. Gate on this.
- `answer_confidence` — probability of the **label that was chosen**.

`score` answers carry `legend` (level index → description) and a weak
`confidence` (0.05–0.19) with a stronger `answer_confidence` (0.36–0.72).

Pass `min_confidence` and each answer reports `abstention`:
`passed` / `abstained` / `unevaluated`.

## Measured on this box (CPU-only, no GPU, live run on 0.4.14)

- `route` / `detect`: 0.4 ms. Cold disk load: ~10 s per checkpoint (one-time
  per process; HF cache ~/.cache/huggingface ~1.5 GB both).
- Warm predict: english 1.8–3.0 s (5-question presets), batch of 4 states 1.0 s.
- `guard`: jailbreak 1.0, prompt injection 1.0. `triage`: intent=refund 0.9935.
  `routemodel`: domain=code 0.9823. `email`: category=security 0.9703,
  is_phishing 0.7576. `moderation`: toxic 0.8095, harassment 0.788, threat 0.7161.
  Hindi refund request 0.9818 on the multilingual checkpoint.
- `schema`: `{"department":"billing","urgency":2,"needs_human":false}`.
- Output ~300–400 chars (~100 tokens); schema ~245 tokens. An equivalent
  LLM classification costs 500–2000+ tokens + 3–30 s + API quota.

## Rules

- Gate on confidence: `>= 0.85` auto-act, else escalate. Ambiguous input
  returns near-zero confidence honestly (measured 0.23 on "hmm, maybe?") —
  that IS the signal.
- `choice`/`noul` are strong (jailbreak 1.0, intent=refund 0.9935,
  domain=code 0.9823, Hindi refund 0.9818 live-verified). `score` is the
  weakest primitive — never auto-act on a score alone.
- Max 20 questions/call, 50 options/choice for `decide`. For wider sets use
  `action=tournament` (elimination rounds, no embedder needed) — `decide`
  rejects >50 loudly and says so.
- Non-Latin script MUST go multilingual: the english checkpoint scores
  0.000 at 0.95 confidence on Khmer. Never rely on confidence to catch a
  wrong-checkpoint call — route BEFORE the forward pass (`auto` does this).
  German also routes to multilingual (Latin script, but not English).
- `long` caps the state at 40 000 chars: wide window sets on a small box
  exhaust RAM and the process gets OOM-killed.
- `typed-decisions` checkpoint is fine-tune-specific; base checkpoints are
  near-chance on its workflows zero-shot. Do not use unless fine-tuned.
- Shipped temperatures warning ("outside [0.5, 5]... clamping") is benign;
  treat affected-bucket confidence as uncalibrated.
- Presets read a NAMED state field (`message`, `body`, `prompt`, `post`,
  `request`). The tool wraps plain text under it automatically; pass a dict
  only if you control the key.

## Install

laya + torch live in `~/.hermes/laya-venv` (dedicated venv, because torch's
PyPI wheel drags in CUDA and `hermes update` drops undeclared packages).
`scripts/install_laya_env.sh` creates it; the tool appends its site-packages
to `sys.path`. If `import laya` fails, run that script and restart the gateway.

## Tests

```bash
python tests/test_laya_tool.py   # offline: 65 cases, no model needed
python tests/test_laya_live.py   # live: real model, every action
```
