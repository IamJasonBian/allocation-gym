#!/usr/bin/env bash
# Build the allocation-agent (workflow-service) wheel and stage it under
# ./vendor/wheelhouse/, optionally installing into the active env.
#
# Usage:
#   scripts/build_workflow_service.sh                # build only
#   scripts/build_workflow_service.sh --install      # build + pip install
#   scripts/build_workflow_service.sh --src PATH     # override source repo
#   WORKFLOW_SERVICE_DIR=/path scripts/build_workflow_service.sh
#
# Defaults assume the workflow-service repo is a sibling:
#   ../allocation-agent-workflow-service

set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DEFAULT="$(cd "$GYM_ROOT/.." && pwd)/allocation-agent-workflow-service"
SRC="${WORKFLOW_SERVICE_DIR:-$SRC_DEFAULT}"
WHEELHOUSE="$GYM_ROOT/vendor/wheelhouse"
EXTRAS="ml"
INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)    INSTALL=1; shift ;;
    --src)        SRC="$2"; shift 2 ;;
    --no-extras)  EXTRAS=""; shift ;;
    -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SRC" ]]; then
  echo "workflow-service not found at: $SRC" >&2
  echo "clone it next to allocation-gym, or pass --src PATH" >&2
  exit 1
fi

mkdir -p "$WHEELHOUSE"
echo ">> source:     $SRC"
echo ">> wheelhouse: $WHEELHOUSE"

# Prefer `uv build` (no env / PEP-668 issues). Fall back to `python -m build`
# inside an active venv.
if command -v uv >/dev/null 2>&1; then
  echo ">> builder:    uv $(uv --version | awk '{print $2}')"
  uv build --wheel --sdist --out-dir "$WHEELHOUSE" "$SRC"
else
  PY="${PYTHON:-python3}"
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "no venv active and uv not installed."                    >&2
    echo "either: brew install uv  OR  source <some-venv>/bin/activate" >&2
    exit 1
  fi
  echo ">> builder:    $($PY -V) ($(command -v "$PY"))"
  "$PY" -m pip install --quiet --upgrade build
  "$PY" -m build --wheel --sdist --outdir "$WHEELHOUSE" "$SRC" >/dev/null
fi

WHL="$(ls -t "$WHEELHOUSE"/allocation_agent-*.whl | head -1)"
echo ">> built: $WHL"

if [[ "$INSTALL" -eq 1 ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]] && ! command -v uv >/dev/null 2>&1; then
    echo "--install needs an active venv (or uv). skipping." >&2
    exit 1
  fi
  TARGET="allocation-agent"
  [[ -n "$EXTRAS" ]] && TARGET="allocation-agent[$EXTRAS]"
  echo ">> installing $TARGET (find-links=$WHEELHOUSE)"
  if command -v uv >/dev/null 2>&1 && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    uv pip install --find-links "$WHEELHOUSE" --upgrade "$TARGET"
  else
    "${PY:-python3}" -m pip install --find-links "$WHEELHOUSE" --upgrade "$TARGET"
  fi
fi
