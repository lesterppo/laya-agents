#!/usr/bin/env bash
# Install the Laya native tool into a Hermes Agent installation.
#
# Plugin-based on purpose: ~/.hermes/plugins/ lives OUTSIDE the hermes-agent git
# tree, so `hermes update` cannot wipe the tool (unlike edits to tools/ or
# toolsets.py, which are reset on every update).
#
# Usage: ./install.sh [--hermes-home DIR] [--uninstall]
# Env:   SKIP_DEPS=1  skip the laya venv install (already set up)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --hermes-home) HERMES_HOME="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PLUGIN_DIR="$HERMES_HOME/plugins/hermes_laya"
SKILL_DIR="$HERMES_HOME/skills/laya-decisions"

if [ "$UNINSTALL" = "1" ]; then
  rm -rf "$PLUGIN_DIR" "$SKILL_DIR"
  echo "removed: $PLUGIN_DIR"
  echo "removed: $SKILL_DIR"
  exit 0
fi

if [ ! -d "$HERMES_HOME" ]; then
  echo "error: $HERMES_HOME does not exist (set HERMES_HOME or pass --hermes-home)" >&2
  exit 1
fi

# Dependency install — laya + torch live in a dedicated venv, NOT in the Hermes
# managed environment:
#   - torch's PyPI wheel drags the whole CUDA runtime in; the CPU wheel only
#     exists on download.pytorch.org, which the managed-env resolver does not use;
#   - `hermes update` regenerates the managed environment and silently drops
#     undeclared third-party packages.
# The tool appends that venv's site-packages to sys.path, so `import laya` works
# inside Hermes and survives env regeneration.
if [ "${SKIP_DEPS:-0}" != "1" ]; then
  "$HERE/scripts/install_laya_env.sh"
fi

# Dependency check — the tool gates on `import laya`, but fail loudly at install
# time so the user does not silently get a missing tool.
PY="${PYTHON:-python3}"
if ! "$PY" -c "import laya" >/dev/null 2>&1; then
  echo "warning: 'laya' is not importable by $PY"
  echo "         install it with:  ./scripts/install_laya_env.sh   (or: pip install laya)"
fi

mkdir -p "$PLUGIN_DIR" "$SKILL_DIR"
cp "$HERE/plugin/__init__.py"      "$PLUGIN_DIR/__init__.py"
cp "$HERE/plugin/plugin.yaml"      "$PLUGIN_DIR/plugin.yaml"
cp "$HERE/plugin/laya_tool.py"     "$PLUGIN_DIR/laya_tool.py"
cp "$HERE/skills/laya-decisions/SKILL.md" "$SKILL_DIR/SKILL.md"

echo "installed plugin -> $PLUGIN_DIR"
echo "installed skill  -> $SKILL_DIR"
echo
echo "Restart the Hermes gateway (or start a new session) for the tool to appear."
echo "Verify with:  hermes tools | grep -A2 laya"
