#!/usr/bin/env bash
# Install Laya (Apache 2.0, NandhaKishorM/laya) + CPU-only torch into a
# dedicated venv, and make it importable from the Hermes interpreter.
#
# Why a dedicated venv instead of plugin.yaml `python_dependencies`:
#   - torch's PyPI wheel drags the whole CUDA runtime in (~2.5 GB of nvidia-*
#     packages); the CPU wheel only exists on download.pytorch.org, which the
#     managed-env resolver does not use.
#   - `hermes update` regenerates the managed environment and silently drops
#     undeclared third-party packages. A venv under ~/.hermes/ survives.
#
# The tool (plugin/laya_tool.py) appends this venv's site-packages to sys.path,
# so `import laya` works inside Hermes. Packages already present in the
# Hermes environment keep priority (append, not insert).
#
# Usage:  ./scripts/install_laya_env.sh [--uninstall]
set -euo pipefail

VENV="${LAYA_VENV:-$HOME/.hermes/laya-venv}"
# laya 0.4.14 is the latest upstream release as of 2026-10-24 (PyPI
# latest == upstream HEAD 0a05a450, release v0.4.14, 2026-10-23).
LAYA_VERSION="${LAYA_VERSION:-0.4.14}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if [ "${1:-}" = "--uninstall" ]; then
  rm -rf "$VENV"
  echo "removed: $VENV"
  exit 0
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "error: $PY not found" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "error: uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }

echo "==> creating venv $VENV (python: $($PY -V 2>&1))"
uv venv --python "$PY" "$VENV"

# --no-config: uv reads the Hermes checkout's uv config (index pinning +
# exclude-newer) when run under ~/.hermes, which filters out torch's CPU
# wheel and pins laya to an older release.
echo "==> installing torch (CPU) + laya==$LAYA_VERSION"
uv pip install --python "$VENV/bin/python" --no-config --index-url "$TORCH_INDEX" torch
uv pip install --python "$VENV/bin/python" --no-config \
  "laya==$LAYA_VERSION" transformers safetensors huggingface-hub

"$VENV/bin/python" - <<'PY'
import laya, torch, transformers
print("laya", laya.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__)
PY

echo
echo "OK — restart the Hermes gateway (or start a new session) for the tool to appear."
echo "Verify with:  hermes tools | grep -A2 laya"
