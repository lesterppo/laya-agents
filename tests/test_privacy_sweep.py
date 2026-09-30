"""Regression tests for privacy_sweep.py.

The scanner's job is to stop secrets from reaching the repo. It must never
become a leak vector itself: matched secrets may not be echoed into stdout,
shell scrollback, or CI logs, and untracked files (e.g. a local `.env`) must
not be read at all.

Runs with the stdlib only:  python3 tests/test_privacy_sweep.py
"""
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import privacy_sweep  # noqa: E402

# Fake secrets used only to verify redaction. Never real.
FAKE_GITHUB_TOKEN = "ghp_faketoken1234567890abcdef"
FAKE_NVIDIA_KEY = "nvapi-faketoken1234567890abcdef"
FAKE_OPENAI_KEY = "sk-faketoken1234567890abcdef12"


def _git_init(root: Path):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)


def _run_main(root: Path):
    """Run privacy_sweep.main against a root and return (exit_code, stdout)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = privacy_sweep.main(root)
    return code, buf.getvalue()


class TestRedaction(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_redact_never_echoes_match(self):
        for secret in (FAKE_GITHUB_TOKEN, FAKE_NVIDIA_KEY, FAKE_OPENAI_KEY):
            shown = privacy_sweep.redact(secret)
            self.assertNotIn(secret, shown)
            # still identifies *which* credential was involved
            self.assertTrue(shown.startswith(secret[:2]))

    def test_output_never_contains_raw_secret(self):
        """End-to-end: a fake token in a file must fail the sweep without the
        raw token appearing anywhere in the output."""
        (self.tmp / "config.txt").write_text(
            f"github = {FAKE_GITHUB_TOKEN}\nnvidia = {FAKE_NVIDIA_KEY}\n"
        )
        code, out = _run_main(self.tmp)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)
        self.assertNotIn(FAKE_GITHUB_TOKEN, out)
        self.assertNotIn(FAKE_NVIDIA_KEY, out)
        # redacted form is shown instead
        self.assertIn(
            f"{FAKE_GITHUB_TOKEN[:2]}…(len={len(FAKE_GITHUB_TOKEN)})", out)

    def test_untracked_secret_file_is_not_read(self):
        """A secret in an untracked file (e.g. a local .env that will never be
        pushed) must not be read — the scanner uses `git ls-files`."""
        _git_init(self.tmp)
        (self.tmp / "app.py").write_text("print('hello')\n")
        subprocess.run(["git", "add", "app.py"], cwd=self.tmp, check=True)
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "init"],
            cwd=self.tmp, check=True)
        # untracked: contains a fake token that would be a leak if read+echoed
        (self.tmp / ".env").write_text(f"NVIDIA_API_KEY={FAKE_NVIDIA_KEY}\n")
        code, out = _run_main(self.tmp)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        self.assertNotIn(FAKE_NVIDIA_KEY, out)

    def test_tracked_secret_file_still_fails_but_redacted(self):
        """A secret in a tracked file must still fail the sweep — redacted."""
        _git_init(self.tmp)
        (self.tmp / "leak.txt").write_text(f"key = {FAKE_OPENAI_KEY}\n")
        subprocess.run(["git", "add", "leak.txt"], cwd=self.tmp, check=True)
        code, out = _run_main(self.tmp)
        self.assertEqual(code, 1)
        self.assertNotIn(FAKE_OPENAI_KEY, out)

    def test_binary_and_oversized_files_are_skipped(self):
        (self.tmp / "blob.bin").write_bytes(
            b"\x00\x01\x02" + FAKE_GITHUB_TOKEN.encode())
        self.assertIsNone(privacy_sweep.read_scan_text(self.tmp / "blob.bin"))
        big = self.tmp / "big.txt"
        big.write_text("x" * (privacy_sweep.MAX_BYTES + 1))
        self.assertIsNone(privacy_sweep.read_scan_text(big))

    def test_clean_tree_passes(self):
        (self.tmp / "ok.py").write_text("print('clean')\n")
        code, out = _run_main(self.tmp)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
