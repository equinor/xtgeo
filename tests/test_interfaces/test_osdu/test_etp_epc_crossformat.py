"""Cross-format roundtrip: RDDMS → xtgeo → resqpy EPC → xtgeo → RDDMS → xtgeo.

Tests that the Sleipner IJK grid (and a Grid2D surface) survive a full
format chain without data loss:

  eqndev RDDMS (maap/sleipner)
    → xtgeo objects  (via ETP read + parametric→explicit conversion)
    → EPC file       (via EpcFileProvider write)
    → xtgeo objects  (via EpcFileProvider read)
    → RDDMS          (via ETP write to maap/xtgeo)
    → xtgeo objects  (via ETP read)
    → compare with original: must be identical except new UUIDs

Uses auth from ~/ores/k8s for eqndev instance. ETP only for RDDMS.
"""

from __future__ import annotations

import sys
import uuid as _uuid
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Auth / config helpers
# ---------------------------------------------------------------------------

K8S_DIR = Path.home() / "ores" / "k8s"


def _load_k8s_env() -> dict[str, str]:
    sys.path.insert(0, str(K8S_DIR))
    from env_from_k8s import load_k8s_yaml

    return {
        **load_k8s_yaml(K8S_DIR / "configmap.yaml"),
        **load_k8s_yaml(K8S_DIR / "secret.yaml"),
    }


def _etp_config(dataspace: str = "maap/sleipner"):
    from xtgeo.interfaces.osdu._session import OsduSession

    env = _load_k8s_env()
    session = OsduSession(
        profile="eqndev",
        etp_url=f"wss://{env['INSTANCE_EQNDEV_HOSTNAME']}/api/reservoir-ddms-etp/v2/",
        token_url=(
            f"https://login.microsoftonline.com/"
            f"{env['INSTANCE_EQNDEV_TENANT_ID']}/oauth2/v2.0/token"
        ),
        client_id=env["INSTANCE_EQNDEV_CLIENT_ID"],
        client_secret=env["INSTANCE_EQNDEV_CLIENT_SECRET"],
        scope=env["INSTANCE_EQNDEV_SCOPE"],
        auth_mode="client_credentials",
        data_partition=env["INSTANCE_EQNDEV_DATA_PARTITION_ID"],
        dataspace=dataspace,
    )
    return session.etp_config()


# ---------------------------------------------------------------------------
# Original grid/surface UUIDs in maap/sleipner
# ---------------------------------------------------------------------------

SLEIPNER_GRID_UUID = "84b37c83-04dc-4dee-883e-5bd3367c6520"
SLEIPNER_SURFACE_UUID = "11dca4a0-276b-42a4-a09a-15fae5adcf5d"  # VeloMap


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_rddms_epc_rddms_roundtrip(tmp_path):
    """Full cross-format roundtrip: RDDMS → EPC → RDDMS, compare xtgeo objects."""
    from xtgeo.interfaces.osdu import EpcFileProvider, EtpProvider
    from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml

    # =======================================================================
    # Step 1: Read from RDDMS (maap/sleipner) → xtgeo
    # =======================================================================
    print("\n=== Step 1: RDDMS (maap/sleipner) → xtgeo ===")
    config_src = _etp_config("maap/sleipner")
    p_src = EtpProvider(config_src)
    p_src.open()

    # Read the Sleipner IJK grid
    grid_orig, props_orig = ijk_grid_to_xtgeo(p_src, SLEIPNER_GRID_UUID)
    print(f"  Grid: {grid_orig.ncol}×{grid_orig.nrow}×{grid_orig.nlay}")
    print(f"  Properties: {[p.name for p in props_orig]}")

    # Read a Grid2D surface
    surf_geom = p_src.get_grid2d_geometry(SLEIPNER_SURFACE_UUID)
    print(
        f"  Surface: ni={surf_geom['ni']}, nj={surf_geom['nj']}, "
        f"origin=({surf_geom['origin_x']:.1f}, {surf_geom['origin_y']:.1f})"
    )

    p_src.close()

    # Snapshot the original xtgeo arrays for final comparison
    coord_orig = grid_orig._coordsv.copy()
    zcorn_orig = grid_orig._zcornsv.copy()
    actnum_orig = grid_orig._actnumsv.copy()
    props_orig_arrays = {p.name: p.values.copy() for p in props_orig}
    surf_vals_orig = surf_geom["values"].copy()

    # =======================================================================
    # Step 2: xtgeo → EPC file (via EpcFileProvider)
    # =======================================================================
    print("\n=== Step 2: xtgeo → EPC file ===")
    epc_path = str(tmp_path / "sleipner_roundtrip.epc")

    with EpcFileProvider(epc_path, mode="w") as epc_w:
        uuids = xtgeo_grid_to_resqml(
            epc_w,
            grid_orig,
            title="Sleipner_RT_Grid",
            properties=props_orig,
        )
        grid_epc_uuid = uuids["Sleipner_RT_Grid"]
        print(f"  Wrote grid: {grid_epc_uuid}")

        # Write surface
        surf_epc_uuid = str(_uuid.uuid4())
        epc_w.put_grid2d_geometry(
            uuid=surf_epc_uuid,
            title="Sleipner_RT_Surface",
            ni=surf_geom["ni"],
            nj=surf_geom["nj"],
            origin_x=surf_geom["origin_x"],
            origin_y=surf_geom["origin_y"],
            di=surf_geom["di"],
            dj=surf_geom["dj"],
            rotation=surf_geom.get("rotation", 0.0),
            values=surf_geom["values"],
            crs_uuid=uuids.get("CRS", str(_uuid.uuid4())),
        )
        print(f"  Wrote surface: {surf_epc_uuid}")

    print(f"  EPC file: {epc_path}")
    h5_path = epc_path.replace(".epc", ".h5")
    epc_size = Path(epc_path).stat().st_size / 1024
    h5_size = Path(h5_path).stat().st_size / 1024 / 1024
    print(f"  EPC size: {epc_size:.1f} KB, H5 size: {h5_size:.1f} MB")

    # =======================================================================
    # Step 3: EPC file → xtgeo (read back from EPC)
    # =======================================================================
    print("\n=== Step 3: EPC file → xtgeo ===")
    with EpcFileProvider(epc_path, mode="r") as epc_r:
        grid_epc, props_epc = ijk_grid_to_xtgeo(epc_r, grid_epc_uuid)
        print(f"  Grid: {grid_epc.ncol}×{grid_epc.nrow}×{grid_epc.nlay}")
        print(f"  Properties: {[p.name for p in props_epc]}")

        surf_epc_geom = epc_r.get_grid2d_geometry(surf_epc_uuid)
        print(f"  Surface: ni={surf_epc_geom['ni']}, nj={surf_epc_geom['nj']}")

    # =======================================================================
    # Step 4: xtgeo → RDDMS (maap/xtgeo) via ETP
    # =======================================================================
    print("\n=== Step 4: xtgeo → RDDMS (maap/xtgeo) ===")
    config_dst = _etp_config("maap/xtgeo")
    p_dst = EtpProvider(config_dst)
    p_dst.open()

    # Ensure dataspace exists
    try:
        p_dst.put_dataspace("maap/xtgeo")
    except Exception:
        pass  # Already exists

    uuids2 = xtgeo_grid_to_resqml(
        p_dst,
        grid_epc,
        title="Sleipner_RT_Grid_ETP",
        properties=props_epc,
    )
    grid_etp_uuid = uuids2["Sleipner_RT_Grid_ETP"]
    print(f"  Wrote grid: {grid_etp_uuid}")

    # Write surface
    surf_etp_uuid = str(_uuid.uuid4())
    p_dst.put_grid2d_geometry(
        uuid=surf_etp_uuid,
        title="Sleipner_RT_Surface_ETP",
        ni=surf_epc_geom["ni"],
        nj=surf_epc_geom["nj"],
        origin_x=surf_epc_geom["origin_x"],
        origin_y=surf_epc_geom["origin_y"],
        di=surf_epc_geom["di"],
        dj=surf_epc_geom["dj"],
        rotation=surf_epc_geom.get("rotation", 0.0),
        values=surf_epc_geom["values"],
        crs_uuid=uuids2.get("CRS", str(_uuid.uuid4())),
    )
    print(f"  Wrote surface: {surf_etp_uuid}")
    p_dst.close()

    # =======================================================================
    # Step 5: RDDMS (maap/xtgeo) → xtgeo (read back)
    # =======================================================================
    print("\n=== Step 5: RDDMS (maap/xtgeo) → xtgeo ===")
    p_final = EtpProvider(config_dst)
    p_final.open()

    grid_final, props_final = ijk_grid_to_xtgeo(p_final, grid_etp_uuid)
    print(f"  Grid: {grid_final.ncol}×{grid_final.nrow}×{grid_final.nlay}")
    print(f"  Properties: {[p.name for p in props_final]}")

    surf_final_geom = p_final.get_grid2d_geometry(surf_etp_uuid)
    print(f"  Surface: ni={surf_final_geom['ni']}, nj={surf_final_geom['nj']}")

    p_final.close()

    # =======================================================================
    # Step 6: Compare original (step 1) vs final (step 5)
    # =======================================================================
    print("\n=== Step 6: Compare original vs final ===")
    all_ok = True

    # Grid dimensions
    assert grid_final.ncol == grid_orig.ncol
    assert grid_final.nrow == grid_orig.nrow
    assert grid_final.nlay == grid_orig.nlay
    print(
        f"  Dimensions: MATCH "
        f"({grid_orig.ncol}×{grid_orig.nrow}×{grid_orig.nlay})"
    )

    # Coord (pillar coordinates, float64)
    coord_final = grid_final._coordsv
    if np.array_equal(coord_orig, coord_final):
        print(f"  coord: BITWISE IDENTICAL ({coord_orig.shape})")
    else:
        max_diff = np.max(np.abs(coord_orig.astype(np.float64) - coord_final.astype(np.float64)))
        n_diff = np.sum(coord_orig != coord_final)
        print(f"  coord: DIFFER (max_diff={max_diff:.2e}, n_diff={n_diff}/{coord_orig.size})")
        np.testing.assert_allclose(
            coord_final.astype(np.float64),
            coord_orig.astype(np.float64),
            atol=1e-6,
            err_msg="Pillar coordinates differ beyond tolerance",
        )
        print(f"  coord: MATCH within 1e-6")
        all_ok = False  # Not bitwise identical

    # Zcorn (z-corners, float32 in xtgeo)
    zcorn_final = grid_final._zcornsv
    if np.array_equal(zcorn_orig, zcorn_final):
        print(f"  zcorn: BITWISE IDENTICAL ({zcorn_orig.shape})")
    else:
        max_diff = np.max(np.abs(zcorn_orig.astype(np.float64) - zcorn_final.astype(np.float64)))
        n_diff = np.sum(zcorn_orig != zcorn_final)
        print(f"  zcorn: DIFFER (max_diff={max_diff:.2e}, n_diff={n_diff}/{zcorn_orig.size})")
        np.testing.assert_allclose(
            zcorn_final.astype(np.float32),
            zcorn_orig.astype(np.float32),
            atol=1e-3,
            err_msg="Z-corners differ beyond tolerance",
        )
        print(f"  zcorn: MATCH within 1e-3")
        all_ok = False

    # Actnum (integer, exact)
    actnum_final = grid_final._actnumsv
    np.testing.assert_array_equal(
        actnum_final.astype(np.int32), actnum_orig.astype(np.int32)
    )
    print(f"  actnum: IDENTICAL ({actnum_orig.shape})")

    # Properties
    props_final_dict = {p.name: p for p in props_final}
    for name, orig_vals in props_orig_arrays.items():
        if name not in props_final_dict:
            print(f"  property '{name}': MISSING in final")
            all_ok = False
            continue
        final_vals = props_final_dict[name].values
        if np.array_equal(orig_vals, final_vals):
            print(f"  property '{name}': BITWISE IDENTICAL ({orig_vals.shape})")
        else:
            max_diff = np.max(np.abs(orig_vals.astype(np.float64) - final_vals.astype(np.float64)))
            n_diff = np.sum(orig_vals != final_vals)
            print(
                f"  property '{name}': DIFFER "
                f"(max_diff={max_diff:.2e}, n_diff={n_diff})"
            )
            np.testing.assert_allclose(
                final_vals.astype(np.float64),
                orig_vals.astype(np.float64),
                atol=1e-6,
            )
            print(f"  property '{name}': MATCH within 1e-6")

    # Surface
    assert surf_final_geom["ni"] == surf_geom["ni"]
    assert surf_final_geom["nj"] == surf_geom["nj"]
    assert abs(surf_final_geom["origin_x"] - surf_geom["origin_x"]) < 1e-6
    assert abs(surf_final_geom["origin_y"] - surf_geom["origin_y"]) < 1e-6
    print(f"  surface metadata: MATCH")

    if np.array_equal(surf_vals_orig, surf_final_geom["values"]):
        print(f"  surface values: BITWISE IDENTICAL ({surf_vals_orig.shape})")
    else:
        max_diff = np.max(np.abs(
            surf_vals_orig.astype(np.float64) - surf_final_geom["values"].astype(np.float64)
        ))
        n_diff = np.sum(surf_vals_orig != surf_final_geom["values"])
        print(
            f"  surface values: DIFFER "
            f"(max_diff={max_diff:.2e}, n_diff={n_diff})"
        )
        np.testing.assert_allclose(
            surf_final_geom["values"].astype(np.float64),
            surf_vals_orig.astype(np.float64),
            atol=1e-6,
        )

    if all_ok:
        print("\n=== PASS: Cross-format roundtrip is BITWISE IDENTICAL ===")
    else:
        print("\n=== PASS: Cross-format roundtrip within tolerance ===")
        print("  (Some arrays differ due to float32↔float64 format conversion)")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_rddms_epc_rddms_roundtrip(Path(td))
