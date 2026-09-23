# laya-agents — agent guide

Native Hermes tool wrapping [Laya](https://github.com/NandhaKishorM/laya), a
multilingual non-autoregressive System 1 decision engine. Decisions
(`choice` / `score` / `noul`) come back from **one forward pass** — no text
generation, so no parsing and no hallucination. Fully local: no API key, no
quota, no network at inference time.

## When to use this tool

Use it when a task needs a fast typed judgment over text and an LLM call would be
wasteful — classify, triage, moderate, guardrail-check, or route.

Prefer it over an LLM call when:

- the output is a **label, an ordinal level, or a yes/no** (not prose);
- the same judgment repeats per item (batch of tickets, messages, papers);
- latency or cost matters (Laya is ~140 ms warm on CPU vs 3–30 s and quota);
- the input language is unknown or non-English (the Router picks the checkpoint).

Do **not** use it when the task needs reasoning, generation, or nuance — that is
what the LLM is for. Laya is the cheap first pass, not an answer generator.

## Actions

| action | what it returns |
|---|---|
| `decide` | answers to your own questions: `choice` (label + probabilities + confidence), `score` (ordinal + distribution), `noul` (calibrated P(true)) |
| `route` | which checkpoint this input needs, with the reason. **No model load** — sub-ms, safe to call first |
| `triage` | intent, is_urgent, frustration, refund_requested, churn_risk |
| `guard` | jailbreak, prompt_injection, sensitive_data, harm_severity, topic |
| `moderation` | toxic, harassment, threat, spam, severity |
| `routemodel` | difficulty, domain, needs_tools, is_sensitive |
| `email` | category, is_spam, is_phishing, urgency, needs_reply |
| `status` | laya version, loaded checkpoints, HF cache presence |

`model` defaults to `auto` (Router decides). Override with `english`,
`multilingual`, or `typed-decisions` only when you have a reason.

## Rules that keep results trustworthy

1. **Gate on confidence.** `>= 0.85` → act; below → escalate to the LLM or a human.
   Ambiguous input returns near-zero confidence by design (measured 0.03) — that is
   the escalation signal, not a bug.
2. **Never auto-act on a `score` alone.** Ordinal questions are the weakest
   primitive (measured confidences 0.05–0.24).
3. **Know the cardinality limit.** One `choice` question tops out at ~50 options
   before the per-label token budget collapses; shortlist or split beyond that.
   The tool rejects >50 options instead of silently degrading.
4. **Keep questions to 20 per call** and batch them: one forward pass answers all of
   them, which is where the 7 ms/question throughput comes from.
5. **Trust the Router for script detection.** The English checkpoint does not
   degrade gracefully off Latin script — it stays confident while being wrong.
6. **Confidence from the affected buckets is uncalibrated** when laya prints its
   shipped-temperature clamp warning; treat those as directional only.
7. **Do not use for medical or financial decisions.** Triage and sorting only.

## Cost of ownership

- First call in a process loads the checkpoint from disk (~35–40 s, disk-bound).
  Keep one long-lived session/router; later calls are ~140 ms (multilingual) or
  ~1–2 s (English).
- Checkpoints occupy ~1.5 GB under `~/.cache/huggingface`.
- `typed-decisions` is fine-tune-specific: base checkpoints are near-chance on its
  workflows zero-shot. Fine-tune before relying on it.

## Layout

```
plugin/    Hermes plugin (__init__.py, plugin.yaml, laya_tool.py)
skills/    Hermes skill: laya-decisions
tests/     offline validation battery (python tests/test_laya_tool.py)
```
