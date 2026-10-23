# Laya × Hermes — state (2026-10-24)

Repo: `lesterppo/laya-agents` (renamed from `hermes-laya` on GitHub).
Local clone: `~/hermes-laya`, branch `master`.

## Task

1. Update the repo from upstream `NandhaKishorM/laya` (0.3.5 → **0.4.14**).
2. Run live + deep tests.

## Status: done

### The blocker (why the tool was invisible)

`hermes tools` listed no `laya` tool: `laya` and `torch` were not in the
managed env, and `hermes update` regenerates that env and drops undeclared
third-party packages. torch could not simply be declared either — its PyPI
wheel drags in ~2.5 GB of CUDA packages, and the CPU wheel only exists on
download.pytorch.org.

Fix: dedicated venv `~/.hermes/laya-venv` (laya 0.4.14 + torch 2.14.1+cpu +
transformers 5.20.2), built by `scripts/install_laya_env.sh`. The tool appends
that venv's site-packages to `sys.path` (append, not insert, so Hermes' own
numpy wins) → `import laya` works inside the Hermes 3.14 interpreter and
survives env regeneration.

### Tool rewrite for the 0.4.x API

0.4.x changed the answer shape, so the old compactor silently dropped data:

| 0.3.5 (old tool read) | 0.4.14 (actual) |
|---|---|
| `probs` / `distribution` / `p` | `probabilities` |
| — | `answer_confidence` (separate from `confidence`) |
| — | `legend` (score level → label) |
| — | `abstention` / `abstention_threshold` |

New actions: `schema` (JSON-Schema → typed values), `batch` (many states, one
forward pass), `long` (windowed document scan), `tournament` (elimination
narrowing for >50-option choices), `detect` (language/script, no model load).

New params: `min_confidence`, `max_len`, `window`, `stride`, `group_size`,
`task`, `lang`, `subject`, `sender`.

Fixed a real bug: laya presets name the state field they read (`message` for
triage, `body` for email, `prompt` for guard, `post` for moderation,
`request` for routemodel). The old tool passed raw text under the wrong key.
Now resolved via `laya.presets.state_field()` and wrapped automatically.

Preserved from the previous upstream of this repo: the Muse integration
(`muse/`), the fail-closed `LAYA_SHA256_DIGESTS` checkpoint pins (digests
re-verified against the cached checkpoints with sha256sum), the score-criteria
validation, and the redacting privacy sweep.

### Tests

- `tests/test_laya_tool.py` — offline battery, **65/65 pass**.
- `tests/test_laya_live.py` — new live battery: **48/48 pass** (real model).

## Live results (0.4.14, CPU-only, this box)

| case | result |
|---|---|
| `decide` (3 questions) | billing 0.9563, refund 0.9172, urgency score conf 0.127 |
| `triage` | intent=refund 0.9935, refund_requested 0.9613, churn_risk 0.1906 |
| `guard` | jailbreak **1.0**, prompt injection **1.0** |
| `moderation` | toxic 0.8095, harassment 0.788, threat 0.7161 |
| `email` (+subject/sender) | category=security 0.9703, is_phishing 0.7576 |
| `routemodel` | domain=code 0.9823, difficulty score 2.008 |
| route en/de/fr/es/zh/hi | english for English; multilingual for everything else |
| `detect` | han script, `why=non_latin_script` — 0.4 ms, no model load |
| decide (Hindi) | refund **0.9818** on the multilingual checkpoint |
| gate `min_confidence=0.85` | `abstention=abstained`, threshold echoed |
| `schema` | `{"department":"billing","urgency":2,"needs_human":false}` |
| `batch` (4 states) | refund/technical/sales/refund, order preserved, 1.04 s |
| `long` | 18 windows, refund found at window 9, noul 1.0, 36.4 s |
| `tournament` (79 options) | 1 elimination round → 5 finalists, refund_request 0.9928 |

Timings: route/detect 0.4 ms; warm english presets 1.9–3.7 s; batch of 4 1.04 s;
cold checkpoint load ~10–15 s; long scan 36.4 s (18 windows).

## Findings

- 0.4.x changed the router default from `english` to `multilingual`: anything
  that is not English now goes to the multilingual checkpoint, Latin script
  included. (An earlier expectation of German→english was wrong; upstream is
  right.)
- `score` stays the weakest primitive: `confidence` 0.05–0.19 while
  `answer_confidence` (the level's own probability) is 0.36–0.72.
- A wide long-document scan OOM-killed the process (exit 137) on this 7 GB box
  (0 free, 2.3 GB swap). `_LONG_STATE_CAP` is now 40 000 chars and the live
  case scans 18 windows without dying.

## Commands

```bash
HP=$(ls -d ~/.hermes/tools/python-3.14.7*/bin/python3 | head -1)
cd ~/hermes-laya
"$HP" tests/test_laya_tool.py        # offline, 65/65
"$HP" tests/test_laya_live.py        # live, 48/48
"$HP" tests/test_laya_live.py --quick   # skips long/tournament
python privacy_sweep.py
```

## Remaining / possible follow-ups

- `muse/laya_muse.py` still targets the 0.3.x answer keys — it needs the same
  0.4.14 treatment (probabilities / answer_confidence / legend) as the Hermes
  tool, plus the new actions if the Muse side wants them.
- The runtime copy at `~/.hermes/plugins/hermes_local_tools/laya_tool.py` is
  synced to the repo version (md5-identical); restart the gateway to pick it up.
