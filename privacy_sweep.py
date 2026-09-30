#!/usr/bin/env python3
"""Pre-push privacy sweep for laya-agents.

Fails (exit 1) if the tracked files contain home paths, personal emails,
account handles, tokens, or private key material. Run before every push:

    python privacy_sweep.py

Security note: matches are REDACTED in output. The scanner must never echo
the credential it finds — stdout, shell scrollback, and CI logs are more
widely readable than the repo itself.
"""
import re
import subprocess
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

# Files larger than this are not read (they would just be slow, not secret-bearing).
MAX_BYTES = 1 << 20  # 1 MiB


def redact(match_text: str) -> str:
    """Identify a match without ever echoing the secret itself.

    Only the first two characters and the length are shown — enough to
    tell *which* credential is involved (e.g. which of two NVIDIA keys)
    without leaking it into stdout, scrollback, or CI logs.
    """
    if len(match_text) <= 4:
        return "<redacted>"
    return f"{match_text[:2]}…(len={len(match_text)})"


def iter_candidate_paths(root: Path = ROOT):
    """Yield the files the sweep should inspect.

    Prefers `git ls-files` so untracked/ignored files (e.g. a local `.env`
    that will never be pushed) are never even read — the scanner cannot
    echo secrets from files that are not going anywhere. Falls back to a
    full tree walk only when `git` is unavailable or this is not a repo.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.split(b"\x00")
        return [root / f.decode("utf-8", "surrogateescape") for f in out if f]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return sorted(p for p in root.rglob("*") if p.is_file())


def read_scan_text(path: Path):
    """Return file text for scanning, or None if the file should be skipped."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None  # binary
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_file(path: Path):
    """Return the list of (line, label, redacted_match) hits in one file."""
    text = read_scan_text(path)
    if text is None:
        return []
    hits = []
    for pattern, label in PATTERNS:
        if label is None:
            continue
        for m in re.finditer(pattern, text):
            line = text[: m.start()].count("\n") + 1
            hits.append((line, label, redact(m.group(0))))
    return hits


def main(root: Path = ROOT) -> int:
    """Run the sweep. Returns 1 if any privacy hits were found, else 0."""
    hits = 0
    for path in iter_candidate_paths(root):
        if not path.is_file():
            continue
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        if any(p in SKIP_DIRS for p in rel.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        for line, label, redacted in scan_file(path):
            print(f"{rel}:{line}: {label}: {redacted}")
            hits += 1

    if hits:
        print(f"\nFAIL: {hits} privacy hit(s) — fix before pushing")
        return 1
    print("PASS: no privacy hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
