#!/usr/bin/env python3
"""Pre-push privacy sweep for hermes-laya.

Fails (exit 1) if the tracked files contain home paths, personal emails,
account handles, tokens, or private key material. Run before every push:

    python privacy_sweep.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATTERNS = [
    (r"/home/[a-z0-9_-]+/", "absolute home path"),
    (r"/mnt/[a-z]/Users/[A-Za-z]+/", "Windows user path"),
    (r"\b[\w.+-]+@(?:gmail|outlook|hotmail|yahoo|icloud)\.com\b", "personal email"),
    (r"\b(?:ppoppo|cyc236|cycptr|cyc236ha|cyc236de|cyc236es|lxp1169|lesteryannes)\b", "private handle"),
    (r"\bghp_[A-Za-z0-9]{20,}\b", "GitHub token"),
    (r"\bsk-[A-Za-z0-9_-]{20,}\b", "API key"),
    (r"\bnvapi-[A-Za-z0-9_-]{20,}\b", "NVIDIA key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"\b(?:10|172|192\.168)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "private IP"),
    (r"\b(?:NandhaKishor|convaiinnovations)\b", None),  # upstream attribution: allowed
]

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
SKIP_FILES = {"privacy_sweep.py"}

hits = 0
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
        continue
    if path.name in SKIP_FILES:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for pattern, label in PATTERNS:
        if label is None:
            continue
        for m in re.finditer(pattern, text):
            line = text[: m.start()].count("\n") + 1
            print(f"{path.relative_to(ROOT)}:{line}: {label}: {m.group(0)!r}")
            hits += 1

if hits:
    print(f"\nFAIL: {hits} privacy hit(s) — fix before pushing")
    sys.exit(1)
print("PASS: no privacy hits")
