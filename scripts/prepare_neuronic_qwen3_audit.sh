#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"

uv run python scripts/audit_neuronic_qwen3_artifacts.py

echo
echo "Audit complete. Copy this bundle off Neuronic before deleting any runs:"
echo "$REPO/.cache/audits/qwen3_audit_bundle.tar.gz"
