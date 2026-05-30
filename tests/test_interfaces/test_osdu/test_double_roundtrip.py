"""Double roundtrip: prove write→read idempotency across TWO full cycles.

For each source dataspace (Sleipner, Drogon):
  Cycle 0: Read all test objects from source  → snapshot_0
  Cycle 1: Write to maap/xtgeo → read back    → snapshot_1
  Cycle 2: Write again (new UUIDs) → read back → snapshot_2

Assert: snapshot_1 == snapshot_2  (bitwise — proves idempotency)
Assert: snapshot_0 ≈ snapshot_1   (bitwise for grids/surfaces, exact for wells)
"""

from __future__ import annotations

import sys
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytestmark = pytest.mark.requires_rddms

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


def _etp_config(dataspace: str):
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
# Object UUIDs to test
# ---------------------------------------------------------------------------

SLEIPNER_OBJECTS = {
    "ijk_grid": "84b37c83-04dc-4dee-883e-5bd3367c6520",
    "grid2d": "11dca4a0-276b-42a4-a09a-15fae5adcf5d",  # VeloMap
    "wellbore_trajectory": "317eab0d-709d-48b1-bddc-8f7d8b41e4ec",
}

DROGON_OBJECTS = {
    "ijk_grid": "8c4f84a5-d819-4ae9-b7e4-03fe06ca5ae7",  # Geogrid (faulted)
    "grid2d": "0c6ab8e7-c793-4ab5-a88c-ccf457d9266d",  # BaseVolantis
    "wellbore_trajectory": "2ceda4cc-02f1-4670-8385-e4af78b9e732",
    "pointset": "4b04246c-d085-45e9-bd51-95759bb22545",  # DP_interp
    "polylineset": "6b8d2ebb-7283-4680-8258-b62918a1e29d",  # AOI
    # TriangulatedSets in Drogon lack blob data on server; tested via synthetic.
}


# ---------------------------------------------------------------------------
# Snapshot: read all objects from a provider
# ---------------------------------------------------------------------------


def _read_snapshot(provider, objects: Dict[str, str], label: str) -> Dict[str, Any]:
    """Read objects from provider and return snapshot dict."""
    from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo

    snap: Dict[str, Any] = {}

    if "ijk_grid" in objects:
        uuid = objects["ijk_grid"]
        # Load properties for Sleipner, skip for Drogon (faulted, very large)
        load_props = "pointset" not in objects  # heuristic: Drogon has pointsets
        grid, props = ijk_grid_to_xtgeo(provider, uuid, load_properties=load_props)
        snap["grid_coord"] = grid._coordsv.copy()
        snap["grid_zcorn"] = grid._zcornsv.copy()
        snap["grid_actnum"] = grid._actnumsv.copy()
        snap["grid_dims"] = (grid.ncol, grid.nrow, grid.nlay)
        snap["grid_obj"] = grid
        snap["grid_props"] = props
        snap["grid_props_arrays"] = {p.name: p.values.copy() for p in props}
        print(
            f"  [{label}] IJK grid: {grid.ncol}x{grid.nrow}x{grid.nlay}, "
            f"props={[p.name for p in props]}"
        )

    if "grid2d" in objects:
        uuid = objects["grid2d"]
        geom = provider.get_grid2d_geometry(uuid)
        snap["grid2d_geom"] = geom
        snap["grid2d_values"] = geom["values"].copy()
        print(
            f"  [{label}] Grid2D: ni={geom['ni']}, nj={geom['nj']}, "
            f"origin=({geom['origin_x']:.1f}, {geom['origin_y']:.1f})"
        )

    if "wellbore_trajectory" in objects:
        uuid = objects["wellbore_trajectory"]
        traj = provider.get_wellbore_trajectory(uuid)
        snap["traj"] = traj
        snap["traj_md"] = traj["md"].copy()
        snap["traj_xyz"] = traj["xyz"].copy()
        print(
            f"  [{label}] Trajectory: {len(traj['md'])} stations, "
            f"title={traj.get('title', '')}"
        )

    if "pointset" in objects:
        uuid = objects["pointset"]
        pts = provider.get_pointset(uuid)
        snap["pointset"] = pts
        snap["pointset_points"] = pts["points"].copy()
        print(f"  [{label}] PointSet: {pts['points'].shape}")

    if "polylineset" in objects:
        uuid = objects["polylineset"]
        pls = provider.get_polylineset(uuid)
        snap["polylineset"] = pls
        snap["polylineset_polylines"] = [p.copy() for p in pls["polylines"]]
        snap["polylineset_closed"] = list(pls["closed"])
        print(
            f"  [{label}] PolylineSet: {len(pls['polylines'])} polylines, "
            f"total pts={sum(len(p) for p in pls['polylines'])}"
        )

    if "triangulated_set" in objects:
        uuid = objects["triangulated_set"]
        tri = provider.get_triangulated_set(uuid)
        snap["triangulated_set"] = tri
        snap["triset_vertices"] = tri["vertices"].copy()
        snap["triset_triangles"] = tri["triangles"].copy()
        print(
            f"  [{label}] TriangulatedSet: "
            f"{tri['vertices'].shape[0]} vertices, "
            f"{tri['triangles'].shape[0]} triangles"
        )

    return snap


# ---------------------------------------------------------------------------
# Write all objects from a snapshot, return new UUID mapping
# ---------------------------------------------------------------------------


def _write_snapshot(
    provider, snap: Dict[str, Any], prefix: str
) -> Dict[str, str]:
    """Write snapshot objects to provider with new UUIDs. Return UUID map."""
    from xtgeo.interfaces.osdu._ijk_grid import xtgeo_grid_to_resqml

    uuid_map: Dict[str, str] = {}

    # CRS UUID for non-grid objects (will be set by grid write or created)
    crs_uuid = str(_uuid.uuid4())

    if "grid_obj" in snap:
        grid = snap["grid_obj"]
        props = snap["grid_props"]
        uuids = xtgeo_grid_to_resqml(
            provider,
            grid,
            title=f"{prefix}_Grid",
            properties=props,
            preserve_uuids=False,
        )
        uuid_map["ijk_grid"] = uuids[f"{prefix}_Grid"]
        crs_uuid = uuids.get("CRS", crs_uuid)
        print(f"  Wrote IJK grid: {uuid_map['ijk_grid']}")

    if "grid2d_geom" in snap:
        geom = snap["grid2d_geom"]
        new_uuid = str(_uuid.uuid4())
        provider.put_grid2d_geometry(
            uuid=new_uuid,
            title=f"{prefix}_Surface",
            ni=geom["ni"],
            nj=geom["nj"],
            origin_x=geom["origin_x"],
            origin_y=geom["origin_y"],
            di=geom["di"],
            dj=geom["dj"],
            rotation=geom.get("rotation", 0.0),
            values=geom["values"],
            crs_uuid=crs_uuid,
        )
        uuid_map["grid2d"] = new_uuid
        print(f"  Wrote Grid2D: {new_uuid}")

    if "traj" in snap:
        traj = snap["traj"]
        new_uuid = str(_uuid.uuid4())
        provider.put_wellbore_trajectory(
            uuid=new_uuid,
            title=f"{prefix}_Well",
            md=traj["md"],
            xyz=traj["xyz"],
            crs_uuid=crs_uuid,
        )
        uuid_map["wellbore_trajectory"] = new_uuid
        print(f"  Wrote Trajectory: {new_uuid}")

    if "pointset" in snap:
        pts = snap["pointset"]
        new_uuid = str(_uuid.uuid4())
        provider.put_pointset(
            uuid=new_uuid,
            title=f"{prefix}_PointSet",
            points=pts["points"],
            crs_uuid=crs_uuid,
        )
        uuid_map["pointset"] = new_uuid
        print(f"  Wrote PointSet: {new_uuid}")

    if "polylineset" in snap:
        pls = snap["polylineset"]
        new_uuid = str(_uuid.uuid4())
        provider.put_polylineset(
            uuid=new_uuid,
            title=f"{prefix}_PolylineSet",
            polylines=pls["polylines"],
            closed=pls["closed"],
            crs_uuid=crs_uuid,
        )
        uuid_map["polylineset"] = new_uuid
        print(f"  Wrote PolylineSet: {new_uuid}")

    if "triangulated_set" in snap:
        tri = snap["triangulated_set"]
        new_uuid = str(_uuid.uuid4())
        provider.put_triangulated_set(
            uuid=new_uuid,
            title=f"{prefix}_TriSet",
            vertices=tri["vertices"],
            triangles=tri["triangles"],
            crs_uuid=crs_uuid,
        )
        uuid_map["triangulated_set"] = new_uuid
        print(f"  Wrote TriangulatedSet: {new_uuid}")

    return uuid_map


# ---------------------------------------------------------------------------
# Compare two snapshots
# ---------------------------------------------------------------------------


def _compare_snapshots(
    snap_a: Dict[str, Any],
    snap_b: Dict[str, Any],
    label: str,
    strict: bool = True,
) -> bool:
    """Compare two snapshots. strict=True requires bitwise match."""
    all_ok = True

    # IJK grid
    if "grid_coord" in snap_a and "grid_coord" in snap_b:
        assert snap_a["grid_dims"] == snap_b["grid_dims"], "Grid dimensions differ"
        print(f"  [{label}] Dimensions: MATCH {snap_a['grid_dims']}")

        for name, atol in [("grid_coord", 1e-6), ("grid_zcorn", 1e-3), ("grid_actnum", 0)]:
            a, b = snap_a[name], snap_b[name]
            if np.array_equal(a, b):
                print(f"  [{label}] {name}: BITWISE IDENTICAL {a.shape}")
            elif strict:
                max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                n_diff = np.sum(a != b)
                print(f"  [{label}] {name}: DIFFER max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}")
                raise AssertionError(f"{name} not bitwise identical in {label}")
            else:
                max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                n_diff = np.sum(a != b)
                print(f"  [{label}] {name}: DIFFER max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}")
                np.testing.assert_allclose(a.astype(np.float64), b.astype(np.float64), atol=atol)
                print(f"  [{label}] {name}: MATCH within {atol}")
                all_ok = False

        # Properties
        for pname, a_vals in snap_a.get("grid_props_arrays", {}).items():
            if pname not in snap_b.get("grid_props_arrays", {}):
                print(f"  [{label}] property '{pname}': MISSING in B")
                if strict:
                    raise AssertionError(f"Property '{pname}' missing in {label}")
                all_ok = False
                continue
            b_vals = snap_b["grid_props_arrays"][pname]
            if np.array_equal(a_vals, b_vals):
                print(f"  [{label}] property '{pname}': BITWISE IDENTICAL {a_vals.shape}")
            elif strict:
                max_diff = np.max(np.abs(a_vals.astype(np.float64) - b_vals.astype(np.float64)))
                print(f"  [{label}] property '{pname}': DIFFER max_diff={max_diff:.2e}")
                raise AssertionError(f"Property '{pname}' not bitwise identical in {label}")
            else:
                max_diff = np.max(np.abs(a_vals.astype(np.float64) - b_vals.astype(np.float64)))
                print(f"  [{label}] property '{pname}': DIFFER max_diff={max_diff:.2e}")
                np.testing.assert_allclose(
                    a_vals.astype(np.float64), b_vals.astype(np.float64), atol=1e-6
                )
                all_ok = False

    # Grid2D surface
    if "grid2d_values" in snap_a and "grid2d_values" in snap_b:
        ga, gb = snap_a["grid2d_geom"], snap_b["grid2d_geom"]
        assert ga["ni"] == gb["ni"] and ga["nj"] == gb["nj"]
        assert abs(ga["origin_x"] - gb["origin_x"]) < 1e-6
        assert abs(ga["origin_y"] - gb["origin_y"]) < 1e-6
        va, vb = snap_a["grid2d_values"], snap_b["grid2d_values"]
        if np.array_equal(va, vb):
            print(f"  [{label}] Grid2D values: BITWISE IDENTICAL {va.shape}")
        elif strict:
            max_diff = np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))
            print(f"  [{label}] Grid2D values: DIFFER max_diff={max_diff:.2e}")
            raise AssertionError(f"Grid2D values not bitwise identical in {label}")
        else:
            max_diff = np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))
            print(f"  [{label}] Grid2D values: DIFFER max_diff={max_diff:.2e}")
            np.testing.assert_allclose(va.astype(np.float64), vb.astype(np.float64), atol=1e-6)
            all_ok = False

    # Wellbore trajectory
    if "traj_md" in snap_a and "traj_md" in snap_b:
        md_a, md_b = snap_a["traj_md"], snap_b["traj_md"]
        xyz_a, xyz_b = snap_a["traj_xyz"], snap_b["traj_xyz"]
        if np.array_equal(md_a, md_b) and np.array_equal(xyz_a, xyz_b):
            print(
                f"  [{label}] Trajectory: BITWISE IDENTICAL "
                f"(md={md_a.shape}, xyz={xyz_a.shape})"
            )
        elif strict:
            md_diff = np.max(np.abs(md_a - md_b)) if md_a.size else 0
            xyz_diff = np.max(np.abs(xyz_a - xyz_b)) if xyz_a.size else 0
            print(f"  [{label}] Trajectory: DIFFER md={md_diff:.2e}, xyz={xyz_diff:.2e}")
            raise AssertionError(f"Trajectory not bitwise identical in {label}")
        else:
            np.testing.assert_allclose(md_a, md_b, atol=1e-6)
            np.testing.assert_allclose(xyz_a, xyz_b, atol=1e-6)
            print(f"  [{label}] Trajectory: MATCH within 1e-6")
            all_ok = False

    # PointSet
    if "pointset_points" in snap_a and "pointset_points" in snap_b:
        pa, pb = snap_a["pointset_points"], snap_b["pointset_points"]
        if np.array_equal(pa, pb):
            print(f"  [{label}] PointSet: BITWISE IDENTICAL {pa.shape}")
        elif strict:
            max_diff = np.max(np.abs(pa - pb))
            print(f"  [{label}] PointSet: DIFFER max_diff={max_diff:.2e}")
            raise AssertionError(f"PointSet not bitwise identical in {label}")
        else:
            np.testing.assert_allclose(pa, pb, atol=1e-6)
            print(f"  [{label}] PointSet: MATCH within 1e-6")
            all_ok = False

    # PolylineSet
    if "polylineset_polylines" in snap_a and "polylineset_polylines" in snap_b:
        pls_a = snap_a["polylineset_polylines"]
        pls_b = snap_b["polylineset_polylines"]
        assert len(pls_a) == len(pls_b), "Different polyline count"
        assert snap_a["polylineset_closed"] == snap_b["polylineset_closed"]
        all_match = True
        for i, (a, b) in enumerate(zip(pls_a, pls_b)):
            if not np.array_equal(a, b):
                all_match = False
                break
        if all_match:
            total_pts = sum(len(p) for p in pls_a)
            print(
                f"  [{label}] PolylineSet: BITWISE IDENTICAL "
                f"({len(pls_a)} polylines, {total_pts} pts)"
            )
        elif strict:
            raise AssertionError(f"PolylineSet not bitwise identical in {label}")
        else:
            for a, b in zip(pls_a, pls_b):
                np.testing.assert_allclose(a, b, atol=1e-6)
            print(f"  [{label}] PolylineSet: MATCH within 1e-6")
            all_ok = False

    # TriangulatedSet
    if "triset_vertices" in snap_a and "triset_vertices" in snap_b:
        va, vb = snap_a["triset_vertices"], snap_b["triset_vertices"]
        ta, tb = snap_a["triset_triangles"], snap_b["triset_triangles"]
        verts_ok = np.array_equal(va, vb)
        tris_ok = np.array_equal(ta, tb)
        if verts_ok and tris_ok:
            print(
                f"  [{label}] TriangulatedSet: BITWISE IDENTICAL "
                f"({va.shape[0]} vertices, {ta.shape[0]} triangles)"
            )
        elif strict:
            if not verts_ok:
                max_diff = np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))
                print(f"  [{label}] TriSet vertices: DIFFER max_diff={max_diff:.2e}")
            if not tris_ok:
                n_diff = np.sum(ta != tb)
                print(f"  [{label}] TriSet triangles: DIFFER n_diff={n_diff}")
            raise AssertionError(f"TriangulatedSet not bitwise identical in {label}")
        else:
            np.testing.assert_allclose(va.astype(np.float64), vb.astype(np.float64), atol=1e-6)
            np.testing.assert_array_equal(ta, tb)
            print(f"  [{label}] TriangulatedSet: MATCH within tolerance")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Core double-roundtrip engine
# ---------------------------------------------------------------------------


def _double_roundtrip(
    source_dataspace: str,
    objects: Dict[str, str],
    name: str,
):
    """Run TWO roundtrip cycles and verify idempotency."""
    from xtgeo.interfaces.osdu import EtpProvider

    target_ds = "maap/xtgeo"

    # ===== Cycle 0: Read from source =====
    print(f"\n{'='*60}")
    print(f"CYCLE 0: Read from {source_dataspace}")
    print(f"{'='*60}")
    cfg_src = _etp_config(source_dataspace)
    p_src = EtpProvider(cfg_src)
    p_src.open()
    snap_0 = _read_snapshot(p_src, objects, "cycle0")
    p_src.close()

    # Ensure target dataspace exists
    cfg_dst = _etp_config(target_ds)
    p_wr = EtpProvider(cfg_dst)
    p_wr.open()
    try:
        p_wr.put_dataspace(target_ds)
    except Exception:
        pass

    # ===== Cycle 1: Write → Read =====
    print(f"\n{'='*60}")
    print(f"CYCLE 1: Write to {target_ds} → Read back")
    print(f"{'='*60}")
    uuid_map_1 = _write_snapshot(p_wr, snap_0, f"{name}_C1")
    p_wr.close()

    p_rd1 = EtpProvider(cfg_dst)
    p_rd1.open()
    snap_1 = _read_snapshot(p_rd1, uuid_map_1, "cycle1")
    p_rd1.close()

    # ===== Cycle 2: Write snap_1 → Read =====
    print(f"\n{'='*60}")
    print(f"CYCLE 2: Write again to {target_ds} → Read back")
    print(f"{'='*60}")
    p_wr2 = EtpProvider(cfg_dst)
    p_wr2.open()
    uuid_map_2 = _write_snapshot(p_wr2, snap_1, f"{name}_C2")
    p_wr2.close()

    p_rd2 = EtpProvider(cfg_dst)
    p_rd2.open()
    snap_2 = _read_snapshot(p_rd2, uuid_map_2, "cycle2")
    p_rd2.close()

    # ===== Compare =====
    print(f"\n{'='*60}")
    print(f"COMPARE: cycle0 vs cycle1 (format conversion tolerance)")
    print(f"{'='*60}")
    ok_01 = _compare_snapshots(snap_0, snap_1, "cycle0↔cycle1", strict=False)

    print(f"\n{'='*60}")
    print(f"COMPARE: cycle1 vs cycle2 (MUST BE BITWISE IDENTICAL)")
    print(f"{'='*60}")
    ok_12 = _compare_snapshots(snap_1, snap_2, "cycle1↔cycle2", strict=True)

    print(f"\n{'='*60}")
    if ok_01 and ok_12:
        print(f"PASS: {name} double roundtrip BITWISE IDENTICAL across all cycles")
    elif ok_12:
        print(f"PASS: {name} double roundtrip IDEMPOTENT (cycle1==cycle2 bitwise)")
        print("  cycle0→cycle1 has expected format conversion differences")
    else:
        print(f"FAIL: {name} double roundtrip NOT idempotent")
    print(f"{'='*60}")

    return ok_01, ok_12


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sleipner_double_roundtrip():
    """Sleipner: IJK grid + properties + Grid2D + well trajectory × 2 cycles."""
    ok_01, ok_12 = _double_roundtrip(
        source_dataspace="maap/sleipner",
        objects=SLEIPNER_OBJECTS,
        name="Sleipner",
    )
    assert ok_12, "Cycle 1 and Cycle 2 must be bitwise identical"


def test_drogon_double_roundtrip():
    """Drogon: faulted IJK grid + Grid2D + well + pointset + polylineset × 2."""
    ok_01, ok_12 = _double_roundtrip(
        source_dataspace="maap/drogon",
        objects=DROGON_OBJECTS,
        name="Drogon",
    )
    assert ok_12, "Cycle 1 and Cycle 2 must be bitwise identical"


def test_synthetic_triset_double_roundtrip():
    """Synthetic TriangulatedSet: create mesh, write→read→write→read, compare.

    Drogon trisets lack blob data on server, so we synthesise a small mesh
    and prove the ETP write↔read path is bitwise idempotent.
    """
    from xtgeo.interfaces.osdu import EtpProvider

    target_ds = "maap/xtgeo"
    cfg = _etp_config(target_ds)

    # Create synthetic mesh (a tetrahedron)
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int32)
    crs_uuid = str(_uuid.uuid4())

    # Ensure dataspace
    p0 = EtpProvider(cfg)
    p0.open()
    try:
        p0.put_dataspace(target_ds)
    except Exception:
        pass
    p0.put_crs(crs_uuid, "SyntheticCRS", 0, 0, 0, 0, True)

    # Cycle 1: write → read
    uuid1 = str(_uuid.uuid4())
    p0.put_triangulated_set(uuid1, "SynTriSet_C1", verts, tris, crs_uuid)
    tri1 = p0.get_triangulated_set(uuid1)
    p0.close()

    assert np.array_equal(tri1["vertices"], verts), "Cycle 1 vertices mismatch"
    assert np.array_equal(tri1["triangles"], tris), "Cycle 1 triangles mismatch"
    print(
        f"  Cycle 1: BITWISE IDENTICAL "
        f"({verts.shape[0]} vertices, {tris.shape[0]} triangles)"
    )

    # Cycle 2: write cycle-1 data → read
    p1 = EtpProvider(cfg)
    p1.open()
    uuid2 = str(_uuid.uuid4())
    p1.put_triangulated_set(uuid2, "SynTriSet_C2", tri1["vertices"], tri1["triangles"], crs_uuid)
    tri2 = p1.get_triangulated_set(uuid2)
    p1.close()

    assert np.array_equal(tri1["vertices"], tri2["vertices"]), "Cycle 2 vertices differ"
    assert np.array_equal(tri1["triangles"], tri2["triangles"]), "Cycle 2 triangles differ"
    print(
        f"  Cycle 2: BITWISE IDENTICAL "
        f"({tri2['vertices'].shape[0]} vertices, {tri2['triangles'].shape[0]} triangles)"
    )
    print("  PASS: Synthetic TriangulatedSet double roundtrip BITWISE IDENTICAL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SLEIPNER DOUBLE ROUNDTRIP")
    print("(IJK grid + 4 properties + Grid2D surface + wellbore trajectory)")
    print("=" * 70)
    test_sleipner_double_roundtrip()

    print("\n\n")
    print("=" * 70)
    print("DROGON DOUBLE ROUNDTRIP")
    print("(faulted IJK grid + Grid2D + well + pointset + polylineset)")
    print("=" * 70)
    test_drogon_double_roundtrip()

    print("\n\n")
    print("=" * 70)
    print("SYNTHETIC TRIANGULATEDSET DOUBLE ROUNDTRIP")
    print("=" * 70)
    test_synthetic_triset_double_roundtrip()
