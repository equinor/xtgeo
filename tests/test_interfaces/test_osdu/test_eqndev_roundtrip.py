#!/usr/bin/env python3
"""
Upload Grid_model + PORO to a cloud OSDU RDDMS, download, and verify exact roundtrip.

Usage:
    # Set env vars directly
    export OSDU_TENANT_ID=""
    export OSDU_CLIENT_ID=""
    export OSDU_CLIENT_SECRET=""
    export OSDU_SCOPE="/.default"
    export OSDU_HOSTNAME=""
    export OSDU_DATA_PARTITION=""
    python tests/test_interfaces/test_osdu/test_eqndev_roundtrip.py

Exit codes:
    0 = all checks passed (exact geometry + property roundtrip)
    1 = verification failed
    2 = connection/auth error
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Grid model files (relative to this script)
# ---------------------------------------------------------------------------
GRID_FILE = os.path.join(
    os.path.dirname(__file__), "../../../tmp/data/Grid_model.grdecl"
)
PORO_FILE = os.path.join(
    os.path.dirname(__file__), "../../../tmp/data/Grid_model__PORO.grdecl"
)


def main():
    import xtgeo
    from xtgeo.interfaces.osdu import (
        EtpProvider,
        OsduSession,
        ijk_grid_to_xtgeo,
        read_grid_properties,
        write_grid_property,
        xtgeo_grid_to_resqml,
    )

    # ── Session from env vars or profile ──────────────────────────────
    # OsduSession.from_env() picks up:
    #   OSDU_HOSTNAME → etp_url + rest_base_url
    #   OSDU_TENANT_ID → token_url
    #   OSDU_CLIENT_ID, OSDU_CLIENT_SECRET, OSDU_SCOPE → auth
    #   OSDU_DATA_PARTITION, OSDU_DATASPACE → connection
    #   OSDU_LEGAL_TAG, OSDU_ACL_OWNERS, OSDU_ACL_VIEWERS → ACL/legal
    # Also: XTGEO_OSDU_* (higher priority)
    profile = None
    if "--profile" in sys.argv:
        idx = sys.argv.index("--profile")
        if idx + 1 < len(sys.argv):
            profile = sys.argv[idx + 1]

    session = OsduSession.load(profile) if profile else OsduSession.from_env()

    # Validate we have credentials for cloud
    if session.auth_mode == "none" and session.etp_url.startswith("wss://"):
        print(
            "ERROR: Cloud connection requires credentials."
            " Set env vars or use --profile.",
            file=sys.stderr,
        )
        print("  See script docstring for required env vars.", file=sys.stderr)
        sys.exit(2)

    print("┌─ Cloud OSDU RDDMS Roundtrip Test ────────────────────────")
    print(f"│  ETP:        {session.etp_url}")
    print(f"│  Partition:  {session.data_partition}")
    print(f"│  Dataspace:  {session.dataspace}")
    print(f"│  Auth:       {session.auth_mode}")
    print(f"│  Grid:       {os.path.basename(GRID_FILE)}")
    print(f"│  Property:   {os.path.basename(PORO_FILE)}")
    print("└────────────────────────────────────────────────────────────")

    # ── 1. Load local grid data ───────────────────────────────────────
    print("\n[1/6] Loading local GRDECL grid...")
    grid_path = os.path.abspath(GRID_FILE)
    poro_path = os.path.abspath(PORO_FILE)

    if not os.path.exists(grid_path):
        print(f"  ERROR: Grid file not found: {grid_path}", file=sys.stderr)
        sys.exit(1)

    grid = xtgeo.grid_from_file(grid_path, fformat="grdecl")
    print(f"  Grid: ncol={grid.ncol}, nrow={grid.nrow}, nlay={grid.nlay}")

    poro = xtgeo.gridproperty_from_file(
        poro_path, fformat="grdecl", name="PORO", grid=grid
    )
    print(
        f"  PORO: min={poro.values.min():.4f}, max={poro.values.max():.4f}, "
        f"shape={poro.values.shape}"
    )

    # Store originals for comparison
    orig_coordsv = grid._coordsv.copy()
    orig_zcornsv = grid._zcornsv.copy()
    orig_actnumsv = grid._actnumsv.copy()
    orig_poro = poro.values.copy()

    # ── 2. Authenticate ───────────────────────────────────────────────
    print("\n[2/6] Authenticating...")
    try:
        token = session.access_token()
    except Exception as e:
        print(f"  ERROR: Authentication failed: {e}", file=sys.stderr)
        sys.exit(2)
    if token:
        print(f"  Token: {token[:20]}...{token[-10:]} ({len(token)} chars)")
    else:
        print("  No auth (local dev mode)")

    # ── 3. Connect and create dataspace ───────────────────────────────
    print(f"\n[3/6] Connecting to ETP and ensuring dataspace '{session.dataspace}'...")

    # Delete + recreate dataspace if requested (for clean testing)
    if os.environ.get("OSDU_RECREATE_DATASPACE", "0") == "1":
        print("  Deleting existing dataspace (OSDU_RECREATE_DATASPACE=1)...")
        try:
            session.delete_dataspace()
        except Exception as e:
            print(f"  (delete: {e})")

    # Create dataspace via REST with ACL/legal (cloud RDDMS needs this)
    if session.rest_base_url:
        try:
            session.create_dataspace_rest()
            print("  Dataspace created via REST")
        except Exception as e:
            print(f"  Dataspace REST: {e} (may already exist)")
    else:
        # Local dev — create via ETP
        try:
            session.create_dataspace_etp()
            print("  Dataspace created via ETP")
        except Exception as e:
            print(f"  Dataspace ETP: {e} (may already exist)")

    # Open ETP provider
    provider = EtpProvider(session.etp_config())
    try:
        provider.open()
    except Exception as e:
        print(f"  ERROR: ETP connection failed: {e}", file=sys.stderr)
        sys.exit(2)

    print("  Connected to ETP")

    # ── 4. Upload grid + property ─────────────────────────────────────
    print("\n[4/6] Uploading grid + PORO property...")
    uuids = xtgeo_grid_to_resqml(provider, grid, title="Grid_model", crs_epsg=23031)
    grid_uuid = uuids["Grid_model"]
    print(f"  Grid UUID:  {grid_uuid}")
    print(f"  CRS UUID:   {uuids['CRS']}")

    prop_uuid = write_grid_property(provider, poro, grid_uuid=grid_uuid)
    print(f"  PORO UUID:  {prop_uuid}")

    # List what's in the dataspace now
    objects = provider.list_objects()
    print(f"\n  Dataspace contents ({len(objects)} objects):")
    for obj in objects:
        print(f"    {obj['type']:35s} {obj['title']}")

    # ── 5. Download and compare ───────────────────────────────────────
    print("\n[5/6] Downloading grid + property from OSDU...")
    grid2, props2 = ijk_grid_to_xtgeo(provider, grid_uuid)
    print(f"  Grid: ncol={grid2.ncol}, nrow={grid2.nrow}, nlay={grid2.nlay}")

    # Read properties
    props_read = read_grid_properties(
        provider, grid_uuid, grid.ncol, grid.nrow, grid.nlay
    )
    print(f"  Properties read: {len(props_read)}")
    poro2 = props_read[0] if props_read else None
    if poro2:
        print(f"  PORO: min={poro2.values.min():.4f}, max={poro2.values.max():.4f}")

    # ── 6. Verify exact roundtrip ─────────────────────────────────────
    print("\n[6/6] Verifying exact geometry and property preservation...")
    errors = []

    # Grid dimensions
    if grid2.ncol != grid.ncol:
        errors.append(f"ncol mismatch: {grid.ncol} → {grid2.ncol}")
    if grid2.nrow != grid.nrow:
        errors.append(f"nrow mismatch: {grid.nrow} → {grid2.nrow}")
    if grid2.nlay != grid.nlay:
        errors.append(f"nlay mismatch: {grid.nlay} → {grid2.nlay}")

    # Coordinate pillars (exact float64)
    if not np.array_equal(grid2._coordsv, orig_coordsv):
        max_diff = np.abs(grid2._coordsv - orig_coordsv).max()
        if max_diff == 0:
            pass  # exact
        elif max_diff < 1e-10:
            errors.append(f"coordsv: near-exact (max diff {max_diff:.2e})")
        else:
            errors.append(f"coordsv: DIFFERS (max diff {max_diff:.6f})")
    exact = np.array_equal(grid2._coordsv, orig_coordsv)
    print(f"  coordsv: {'EXACT' if exact else 'DIFFERS'}")

    # Z-corners (exact float64)
    if not np.array_equal(grid2._zcornsv, orig_zcornsv):
        max_diff = np.abs(grid2._zcornsv - orig_zcornsv).max()
        if max_diff == 0:
            pass
        elif max_diff < 1e-10:
            errors.append(f"zcornsv: near-exact (max diff {max_diff:.2e})")
        else:
            errors.append(f"zcornsv: DIFFERS (max diff {max_diff:.6f})")
    exact = np.array_equal(grid2._zcornsv, orig_zcornsv)
    print(f"  zcornsv: {'EXACT' if exact else 'DIFFERS'}")

    # Activity mask (exact int32)
    if not np.array_equal(grid2._actnumsv, orig_actnumsv):
        n_diff = np.sum(grid2._actnumsv != orig_actnumsv)
        errors.append(f"actnumsv: {n_diff} cells differ")
    exact = np.array_equal(grid2._actnumsv, orig_actnumsv)
    print(f"  actnumsv: {'EXACT' if exact else 'DIFFERS'}")

    # PORO property (exact float64)
    if poro2 is not None:
        # Flatten and compare (handle masked arrays)
        p1 = np.ma.filled(orig_poro, fill_value=np.nan).ravel()
        p2 = np.ma.filled(poro2.values, fill_value=np.nan).ravel()
        if p1.shape != p2.shape:
            errors.append(f"PORO shape mismatch: {p1.shape} vs {p2.shape}")
        elif not np.allclose(p1, p2, equal_nan=True, atol=0, rtol=0):
            max_diff = np.nanmax(np.abs(p1 - p2))
            if max_diff < 1e-10:
                print(f"  PORO: near-exact (max diff {max_diff:.2e})")
            else:
                errors.append(f"PORO: max diff {max_diff:.6e}")
        else:
            print("  PORO: EXACT")
    else:
        errors.append("PORO: not returned from read")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not errors:
        print("✓ ALL CHECKS PASSED — exact geometry and property roundtrip")
        print("=" * 60)
        rc = 0
    else:
        print("✗ VERIFICATION FAILED:")
        for err in errors:
            print(f"  • {err}")
        print("=" * 60)
        rc = 1

    provider.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
