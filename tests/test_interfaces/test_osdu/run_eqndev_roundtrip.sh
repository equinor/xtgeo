#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eqndev_roundtrip.sh
#
# Upload Grid_model + PORO to Equinor OSDU dev instance, download, and verify
# exact roundtrip preservation of geometry and properties.
#
# Uses the same k8s secrets as the ores application.
#
# Usage:
#   bash tests/test_interfaces/test_osdu/run_eqndev_roundtrip.sh
#
# Prerequisites:
#   - /home/maap/ores/k8s/secret.yaml exists (with OSDU credentials)
#   - xtgeo[osdu] installed
#   - Network access to equinorswedev.energy.azure.com
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  XTGeo → Equinor OSDU Dev Roundtrip Test                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Load secrets from ores/k8s (same pattern as the ores app) ──────────────
K8S_SCRIPT="/home/maap/ores/k8s/env_from_k8s.py"
if [[ -f "$K8S_SCRIPT" ]]; then
    echo "Loading credentials from ores/k8s secrets..."
    eval "$(python3 "$K8S_SCRIPT" 2>/dev/null)"
    echo "  ✓ Secrets loaded"
else
    echo "WARNING: $K8S_SCRIPT not found."
    echo "Set OSDU_* env vars manually or provide k8s/secret.yaml."
fi

# ── Verify grid files exist ────────────────────────────────────────────────
GRID_FILE="$REPO_ROOT/tmp/data/Grid_model.grdecl"
PORO_FILE="$REPO_ROOT/tmp/data/Grid_model__PORO.grdecl"

if [[ ! -f "$GRID_FILE" ]]; then
    echo "ERROR: Grid file not found: $GRID_FILE"
    exit 1
fi
if [[ ! -f "$PORO_FILE" ]]; then
    echo "ERROR: Property file not found: $PORO_FILE"
    exit 1
fi
echo "  Grid:  $GRID_FILE"
echo "  PORO:  $PORO_FILE"
echo ""

# ── Run the Python test ────────────────────────────────────────────────────
cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/test_eqndev_roundtrip.py"
exit $?
