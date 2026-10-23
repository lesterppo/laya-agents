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
| `schema` | typed values from a JSON Schema: enum → label, bounded int → level, bool → bool, plus per-field confidence |
| `batch` | many states in one forward pass, input order preserved |
| `long` | windowed scan of a long document, with the deciding window index |
| `tournament` | a wide choice (>50 options) narrowed by elimination rounds |
| `detect` | language + script, with the reason. **No model load** — sub-ms |
| `route` | which checkpoint this input needs, with the reason. **No model load** — sub-ms, safe to call first |
| `triage` | intent, is_urgent, frustration, refund_requested, churn_risk |
| `guard` | jailbreak, prompt_injection, sensitive_data, harm_severity, topic |
| `moderation` | toxic, harassment, threat, spam, severity |
| `routemodel` | difficulty, domain, needs_tools, is_sensitive |
| `email` | category, is_spam, is_phishing, urgency, needs_reply |
| `status` | laya version, loaded checkpoints + pinned revisions, HF cache presence |

Targets upstream laya **0.4.14**. Two confidences come back per answer:
`confidence` (in the decision — gate on this) and `answer_confidence` (in the
label that was chosen). With `min_confidence` set, each answer also reports
`abstention`: `passed` / `abstained` / `unevaluated`.

`model` defaults to `auto` (Router decides). Override with `english`,
`multilingual`, or `typed-decisions` only when you have a reason.

## Rules that keep results trustworthy

1. **Gate on confidence.** `>= 0.85` → act; below → escalate to the LLM or a human.
   Ambiguous input returns near-zero confidence by design (measured 0.03) — that is
   the escalation signal, not a bug.
2. **Never auto-act on a `score` alone.** Ordinal questions are the weakest
   primitive (measured confidences 0.05–0.24).
3. **Know the cardinality limit.** One `choice` question tops out at ~50 options
   before the per-label token budget collapses; `decide` rejects >50 instead of
   silently degrading and points you at `action=tournament`, which narrows the
   field by elimination rounds (no embedder needed).
4. **Keep questions to 20 per call** and batch them: one forward pass answers all of
   them, which is where the 7 ms/question throughput comes from.
5. **Trust the Router for script detection.** The English checkpoint does not
   degrade gracefully off Latin script — it stays confident while being wrong.
6. **Confidence from the affected buckets is uncalibrated** when laya prints its
   shipped-temperature clamp warning; treat those as directional only.
7. **Do not use for medical or financial decisions.** Triage and sorting only.

## Cost of ownership

- First call in a process loads the checkpoint from disk (~10 s, disk-bound).
  Keep one long-lived session/router; later calls are ~1.8–3.0 s for the
  5-question English presets, ~1.0 s for a batch of 4 states.
- Checkpoints occupy ~1.5 GB under `~/.cache/huggingface`.
- `typed-decisions` is fine-tune-specific: base checkpoints are near-chance on its
  workflows zero-shot. Fine-tune before relying on it.
- `long` caps the state at 40 000 chars and chunks its windows 8 per
  forward pass (upstream's single-pass default OOM-killed a 7 GB box —
  exit 137 — on a wide multilingual scan). `batch` chunks the same way
  and caps each state at 8 000 chars.
- laya + torch live in `~/.hermes/laya-venv` (dedicated venv), NOT in the
  Hermes managed environment: torch's PyPI wheel drags in CUDA, and
  `hermes update` drops undeclared packages. `scripts/install_laya_env.sh`
  builds it; the tool appends its site-packages to `sys.path`.

## Layout

```
plugin/    Hermes plugin (__init__.py, plugin.yaml, laya_tool.py)
skills/    Hermes skill: laya-decisions
scripts/   install_laya_env.sh — laya + CPU torch into ~/.hermes/laya-venv
tests/     offline battery (tests/test_laya_tool.py) + live battery
           (tests/test_laya_live.py)
muse/      Muse-native module + batch CLI (no Hermes)
```

Presets read a NAMED state field (`message`, `body`, `prompt`, `post`,
`request`); the tool wraps plain text under it automatically via
`laya.presets.state_field()`.
