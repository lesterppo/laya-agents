---
name: laya-decisions
description: Use for fast typed text decisions without an LLM call.
version: 1.0.0
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

## Tool actions

- `decide` — state/text + `questions` JSON (each: type choice|score|noul +
  instructions + criteria). `model` defaults to `auto` (Router picks).
- `route` — which checkpoint, NO model load (sub-ms). Always safe, works
  before any download.
- Presets (no schema needed): `triage` `guard` `moderation` `routemodel`
  `email`. `status` = version + loaded models.

## Measured on this box (CPU-only, no GPU)

- `route`: 0.06 ms. Cold disk load: english ~34 s, multilingual ~40 s
  (one-time per process; HF cache ~/.cache/huggingface ~1.5 GB both).
- Warm predict: multilingual ~140 ms, english ~1.2–2.4 s (3–5 questions).
- Output ~300–400 chars (~100 tokens); schema ~245 tokens. An equivalent
  LLM classification costs 500–2000+ tokens + 3–30 s + API quota.

## Rules

- Gate on confidence: `>= 0.85` auto-act, else escalate. Ambiguous input
  returns near-zero confidence honestly (measured 0.03) — that IS the signal.
- `choice`/`noul` are strong (billing 0.82–0.999, jailbreak 1.0, threat 0.92
  live-verified, incl. Hindi/German/Traditional Chinese). `score` is the
  weakest primitive (confidences 0.05–0.24 measured) — never auto-act on a
  score alone.
- Max 20 questions/call, 50 options/choice (Banking77-scale sets must be
  shortlisted or split first — architectural token-budget limit).
- Non-Latin script MUST go multilingual: the english checkpoint scores
  0.000 at 0.95 confidence on Khmer. Never rely on confidence to catch a
  wrong-checkpoint call — route BEFORE the forward pass (`auto` does this).
- `typed-decisions` checkpoint is fine-tune-specific; base checkpoints are
  near-chance on its workflows zero-shot. Do not use unless fine-tuned.
- Shipped temperatures warning ("outside [0.5, 5]... clamping") is benign;
  treat affected-bucket confidence as uncalibrated.
