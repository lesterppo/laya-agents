# Laya × Hermes — state (2026-10-24)

Repo: `lesterppo/laya-agents` (renamed from `hermes-laya` on GitHub).
Local clone: `~/hermes-laya`, branch `master`.

## 2026-10-24 (Muse VM, 7.7 GB / 2 cores, no swap) — OOM fix + Muse parity

Peter: the test on another Linux computer went OOM; continue development
and testing on this VM. Done:

### OOM root cause (measured, not guessed)

Upstream `Agent.predict_batch(batch_size=None)` sends **every** state in
one forward pass, and `Router.predict_long` does the same with **every**
window (its `window_batch_cap` only chunks when a small window multiplied
the count). Peak RAM therefore grows with the item count, not the item
size. Checkpoint residency is the other half: english weights 804 MB /
multilingual 615 MB (fp32), one checkpoint load + one predict peaked at
2,826 MB RSS, both checkpoints resident + an english long scan at
4,105 MB.

Reproduction (pre-fix code, `systemd-run -p MemoryMax=3.5G`): a
40 000-char Traditional-Chinese `long` scan — inside the tool's own cap —
tokenised to 27,202 tokens / 71 windows (multilingual width 1024) and was
**OOM-killed, exit 137**, exactly the reported failure. The same scan on
the fixed code completes under the same cap: `ok, 71 windows, 384 s`.

### Fixes

- `plugin/laya_tool.py` + `muse/laya_muse.py`: `batch` and `long` now
  always pass `batch_size=8` (`_FORWARD_BATCH`) to the router, so peak
  activation memory is flat in the item count. `batch` also caps each
  state at 8 000 chars (previously uncapped).
- Plugin gained the Muse module's proxy fix at import time:
  bracketed-IPv6 `no_proxy` (e.g. `[::1]`) crashes httpx inside
  huggingface_hub (`Invalid port: ':1]'`) on first download — the
  offline battery failed exactly this way on this VM before the fix —
  plus `HF_HUB_DISABLE_XET=1` fallback.
- **`muse/laya_muse.py` brought to full 0.4.14 parity** (the standing
  follow-up below): it previously exposed only 8 of the 13 actions with
  the old semantics. Now: `schema`, `batch`, `long`, `tournament`,
  `detect` added; `min_confidence` / `max_len` / `task` / `lang` /
  `window` / `stride` / `group_size` parameters; email `subject`/`sender`
  via `laya.email_state()` (previously silently ignored); route reports
  detection language/script; status reports pinned revisions; batch-mode
  request objects pass every field through.

### Tests (this VM)

- `tests/test_laya_tool.py` — offline, **73/73** (was 65; +batch per-state
  cap, proxy sanitize, stub-router chunking assertions).
- `tests/test_laya_muse.py` — **new** offline battery for the Muse
  module, **73/73**.
- `tests/test_laya_live.py` — live, **56/56** (was 48; +multilingual long
  scan — 15 windows, refund 0.9999 — and a 16-state chunked batch).
  Whole-battery peak RSS 3,887 MB. Timings on this shared 2-core VM are
  far slower than the Hermes box (decide cold 27.6 s, long 77.7 s) and
  are not comparable; correctness values match the Hermes run exactly.
- Muse module also live-verified directly (guard 1.0, Hindi 0.9818,
  schema/email identical to the plugin's).

## Task (previous session)

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

- ~~`muse/laya_muse.py` still targets the 0.3.x answer keys~~ — done
  2026-10-24 (Muse VM session above): full 0.4.14 parity + its own
  offline battery.
- The runtime copy at `~/.hermes/plugins/hermes_local_tools/laya_tool.py` is
  synced to the repo version (md5-identical); restart the gateway to pick it up.
  NOTE: after this session's plugin changes, the Hermes box must re-sync
  its runtime copy from the repo (and its `~/.hermes/laya-venv` is the
  install the plugin bootstraps from).
- A single `decide` on the multilingual checkpoint still runs one
  sequence up to width 1024 (or a caller's `max_len` override) — that
  peak is inherent to one forward pass and cannot be chunked; on a
  <4 GB box prefer `english` routing or a smaller `max_len`.
