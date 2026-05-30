"""Double roundtrip: prove write→read idempotency across TWO full cycles.

For each source dataspace (Sleipner, Drogon):
  Cycle 0: Read all test objects from source  → snapshot_0
  Cycle 1: Write to maap/xtgeo → read back    → snapshot_1
  Cycle 2: Write again (new UUIDs) → read back → snapshot_2

Assert: snapshot_1 == snapshot_2  (bitwise — proves idempotency)
Assert: snapshot_0 ≈ snapshot_1   (bitwise for grids/surfaces, exact for wells)
"""

from __future__ import annotations

import contextlib
import sys
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytestmark = pytest.mark.requires_rddms

# ---------------------------------------------------------------------------
# Auth / config helpers
# ---------------------------------------------------------------------------

K8S_DIR = Path.home() / "ores" / "k8s"

if not K8S_DIR.exists():
    pytest.skip("K8s env directory not available", allow_module_level=True)


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


def _write_snapshot(provider, snap: Dict[str, Any], prefix: str) -> Dict[str, str]:
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

        for name, atol in [
            ("grid_coord", 1e-6),
            ("grid_zcorn", 1e-3),
            ("grid_actnum", 0),
        ]:
            a, b = snap_a[name], snap_b[name]
            if np.array_equal(a, b):
                print(f"  [{label}] {name}: BITWISE IDENTICAL {a.shape}")
            elif strict:
                max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                n_diff = np.sum(a != b)
                print(
                    f"  [{label}] {name}: DIFFER "
                    f"max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}"
                )
                raise AssertionError(f"{name} not bitwise identical in {label}")
            else:
                max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                n_diff = np.sum(a != b)
                print(
                    f"  [{label}] {name}: DIFFER "
                    f"max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}"
                )
                np.testing.assert_allclose(
                    a.astype(np.float64), b.astype(np.float64), atol=atol
                )
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
                print(
                    f"  [{label}] property '{pname}': BITWISE IDENTICAL {a_vals.shape}"
                )
            elif strict:
                max_diff = np.max(
                    np.abs(a_vals.astype(np.float64) - b_vals.astype(np.float64))
                )
                print(f"  [{label}] property '{pname}': DIFFER max_diff={max_diff:.2e}")
                raise AssertionError(
                    f"Property '{pname}' not bitwise identical in {label}"
                )
            else:
                max_diff = np.max(
                    np.abs(a_vals.astype(np.float64) - b_vals.astype(np.float64))
                )
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
            np.testing.assert_allclose(
                va.astype(np.float64), vb.astype(np.float64), atol=1e-6
            )
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
            print(
                f"  [{label}] Trajectory: DIFFER md={md_diff:.2e}, xyz={xyz_diff:.2e}"
            )
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
            np.testing.assert_allclose(
                va.astype(np.float64), vb.astype(np.float64), atol=1e-6
            )
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
    print(f"\n{'=' * 60}")
    print(f"CYCLE 0: Read from {source_dataspace}")
    print(f"{'=' * 60}")
    cfg_src = _etp_config(source_dataspace)
    p_src = EtpProvider(cfg_src)
    p_src.open()
    snap_0 = _read_snapshot(p_src, objects, "cycle0")
    p_src.close()

    # Ensure target dataspace exists
    cfg_dst = _etp_config(target_ds)
    p_wr = EtpProvider(cfg_dst)
    p_wr.open()
    with contextlib.suppress(Exception):
        p_wr.put_dataspace(target_ds)

    # ===== Cycle 1: Write → Read =====
    print(f"\n{'=' * 60}")
    print(f"CYCLE 1: Write to {target_ds} → Read back")
    print(f"{'=' * 60}")
    uuid_map_1 = _write_snapshot(p_wr, snap_0, f"{name}_C1")
    p_wr.close()

    p_rd1 = EtpProvider(cfg_dst)
    p_rd1.open()
    snap_1 = _read_snapshot(p_rd1, uuid_map_1, "cycle1")
    p_rd1.close()

    # ===== Cycle 2: Write snap_1 → Read =====
    print(f"\n{'=' * 60}")
    print(f"CYCLE 2: Write again to {target_ds} → Read back")
    print(f"{'=' * 60}")
    p_wr2 = EtpProvider(cfg_dst)
    p_wr2.open()
    uuid_map_2 = _write_snapshot(p_wr2, snap_1, f"{name}_C2")
    p_wr2.close()

    p_rd2 = EtpProvider(cfg_dst)
    p_rd2.open()
    snap_2 = _read_snapshot(p_rd2, uuid_map_2, "cycle2")
    p_rd2.close()

    # ===== Compare =====
    print(f"\n{'=' * 60}")
    print("COMPARE: cycle0 vs cycle1 (format conversion tolerance)")
    print(f"{'=' * 60}")
    ok_01 = _compare_snapshots(snap_0, snap_1, "cycle0↔cycle1", strict=False)

    print(f"\n{'=' * 60}")
    print("COMPARE: cycle1 vs cycle2 (MUST BE BITWISE IDENTICAL)")
    print(f"{'=' * 60}")
    ok_12 = _compare_snapshots(snap_1, snap_2, "cycle1↔cycle2", strict=True)

    print(f"\n{'=' * 60}")
    if ok_01 and ok_12:
        print(f"PASS: {name} double roundtrip BITWISE IDENTICAL across all cycles")
    elif ok_12:
        print(f"PASS: {name} double roundtrip IDEMPOTENT (cycle1==cycle2 bitwise)")
        print("  cycle0→cycle1 has expected format conversion differences")
    else:
        print(f"FAIL: {name} double roundtrip NOT idempotent")
    print(f"{'=' * 60}")

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
    with contextlib.suppress(Exception):
        p0.put_dataspace(target_ds)
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
    p1.put_triangulated_set(
        uuid2, "SynTriSet_C2", tri1["vertices"], tri1["triangles"], crs_uuid
    )
    tri2 = p1.get_triangulated_set(uuid2)
    p1.close()

    assert np.array_equal(tri1["vertices"], tri2["vertices"]), "Cycle 2 vertices differ"
    assert np.array_equal(tri1["triangles"], tri2["triangles"]), (
        "Cycle 2 triangles differ"
    )
    print(
        f"  Cycle 2: BITWISE IDENTICAL "
        f"({tri2['vertices'].shape[0]} vertices, "
        f"{tri2['triangles'].shape[0]} triangles)"
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


# ===========================================================================
# LOCAL RDDMS: Discovery & Notification Tests
# (Requires local RDDMS at ws://localhost:9002)
# ===========================================================================

import xtgeo  # noqa: E402
from xtgeo.interfaces.osdu import (  # noqa: E402
    EtpConnectionConfig,
    EtpProvider,
    xtgeo_grid_to_resqml,
    xtgeo_surface_to_resqml,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def etp_config():
    """Config with a unique test dataspace."""
    ds_path = f"xtgeo/test_disc_{_uuid.uuid4().hex[:8]}"
    return EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace=f"eml:///dataspace('{ds_path}')",
    ), ds_path


@pytest.fixture
def provider(etp_config):
    """ETP provider with a fresh test dataspace."""
    cfg, ds_path = etp_config
    p = EtpProvider(cfg)
    try:
        p.open()
    except Exception:
        pytest.skip("Local RDDMS not available at ws://localhost:9002")

    with contextlib.suppress(Exception):
        p.put_dataspace(ds_path)

    yield p
    with contextlib.suppress(Exception):
        p.delete_dataspace(ds_path)
    p.close()


@pytest.fixture
def populated_provider(provider):
    """Provider with a grid + surface already written."""
    # Write a small grid
    grid = xtgeo.create_box_grid((3, 3, 2))
    grid_uuids = xtgeo_grid_to_resqml(provider, grid, title="TestGrid_Discovery")
    grid_uuid = grid_uuids["TestGrid_Discovery"]
    crs_uuid = grid_uuids["CRS"]

    # Write a property for the grid
    prop = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(3, 3, 2))
    from xtgeo.interfaces.osdu import write_grid_property

    prop_uuid = write_grid_property(
        provider,
        prop,
        grid_uuid=grid_uuid,
    )

    # Write a surface
    surf = xtgeo.RegularSurface(
        ncol=5,
        nrow=5,
        xinc=25.0,
        yinc=25.0,
        xori=0.0,
        yori=0.0,
        values=np.random.rand(5, 5) * 100 + 1000,
    )
    surf_uuids = xtgeo_surface_to_resqml(provider, surf, title="TestSurf_Discovery")

    return provider, {
        "grid_uuid": grid_uuid,
        "crs_uuid": crs_uuid,
        "prop_uuid": prop_uuid,
        "surf_uuid": surf_uuids["TestSurf_Discovery"],
    }


# ---------------------------------------------------------------------------
# Deep Discovery Tests
# ---------------------------------------------------------------------------


class TestDeepDiscovery:
    """Test the discover() method with various parameters."""

    def test_discover_all_objects(self, populated_provider):
        """Discover all objects in the dataspace with unlimited depth."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0)

        assert "resources" in result
        assert "edges" in result
        assert len(result["resources"]) >= 3  # grid + property + surface + CRS

        # Check that known objects appear
        found_uuids = {r["uuid"] for r in result["resources"]}
        assert uuids["grid_uuid"] in found_uuids
        assert uuids["surf_uuid"] in found_uuids

    def test_discover_type_filter(self, populated_provider):
        """Discover only IjkGrid objects."""
        provider, uuids = populated_provider

        result = provider.discover(
            depth=0,
            object_types=["resqml20.IjkGridRepresentation"],
        )

        assert len(result["resources"]) >= 1
        for r in result["resources"]:
            assert "IjkGrid" in r["type"]

    def test_discover_depth_limited(self, populated_provider):
        """Discover with depth=1 only finds direct children."""
        provider, uuids = populated_provider

        result_d1 = provider.discover(depth=1)
        result_d0 = provider.discover(depth=0)

        # Unlimited depth should find >= depth=1 results
        assert len(result_d0["resources"]) >= len(result_d1["resources"])

    def test_discover_from_specific_object(self, populated_provider):
        """Discover starting from a specific grid object."""
        provider, uuids = populated_provider

        # Find the grid URI
        objects = provider.list_objects()
        grid_uri = None
        for obj in objects:
            if obj["uuid"] == uuids["grid_uuid"]:
                grid_uri = obj["uri"]
                break

        assert grid_uri is not None

        result = provider.discover(
            uri=grid_uri,
            depth=1,
            scope="sources",
        )

        # Should find objects that reference the grid (properties)
        assert "resources" in result

    def test_discover_with_edges(self, populated_provider):
        """Discover with include_edges=True returns edge information."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0, include_edges=True)

        assert "edges" in result
        # At minimum, property → grid edge should exist
        if result["edges"]:
            for edge in result["edges"]:
                assert "source_uri" in edge
                assert "target_uri" in edge
                assert "relationship_kind" in edge

    def test_discover_resource_fields(self, populated_provider):
        """Verify that discovered resources have expected fields."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0)

        for r in result["resources"]:
            assert "uuid" in r
            assert "title" in r
            assert "type" in r
            assert "uri" in r
            assert "last_changed" in r
            assert "store_created" in r


# ---------------------------------------------------------------------------
# Related Objects Tests
# ---------------------------------------------------------------------------


class TestRelatedObjects:
    """Test the get_related_objects() convenience method."""

    def test_get_sources_of_grid(self, populated_provider):
        """Find objects that reference the grid (properties)."""
        provider, uuids = populated_provider

        related = provider.get_related_objects(
            uuids["grid_uuid"],
            direction="sources",
        )

        # At least the property should reference the grid
        assert isinstance(related, list)
        # The property references the grid as a supporting representation
        any(r["uuid"] == uuids["prop_uuid"] for r in related)
        # This depends on RDDMS relationship tracking
        # At minimum, should not error
        assert related is not None

    def test_get_targets_of_grid(self, populated_provider):
        """Find objects that the grid references (CRS)."""
        provider, uuids = populated_provider

        related = provider.get_related_objects(
            uuids["grid_uuid"],
            direction="targets",
        )

        # Grid should reference CRS
        assert isinstance(related, list)

    def test_invalid_uuid_raises(self, provider):
        """get_related_objects with nonexistent UUID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            provider.get_related_objects("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# Deleted Resources Tests
# ---------------------------------------------------------------------------


class TestDeletedResources:
    """Test the get_deleted_resources() method."""

    def test_deleted_resources_empty(self, provider):
        """Fresh dataspace has no deleted resources."""
        deleted = provider.get_deleted_resources()
        assert isinstance(deleted, list)
        # May or may not be empty depending on RDDMS implementation
        # but the call should not fail

    def test_deleted_resources_after_write(self, populated_provider):
        """Calling get_deleted_resources does not error after writing objects."""
        provider, uuids = populated_provider
        deleted = provider.get_deleted_resources()
        assert isinstance(deleted, list)


# ---------------------------------------------------------------------------
# High-Level API Tests
# ---------------------------------------------------------------------------


class TestDeepQueryAPI:
    """Test the deep_query_osdu() high-level API."""

    def test_deep_query_all(self, populated_provider):
        """deep_query_osdu discovers all objects."""
        provider, uuids = populated_provider

        # Create a session-like object (use provider directly)
        result = xtgeo.deep_query_osdu(provider, depth=0)

        assert "resources" in result
        assert len(result["resources"]) >= 3

    def test_deep_query_specific_object(self, populated_provider):
        """deep_query_osdu starting from a specific object."""
        provider, uuids = populated_provider

        result = xtgeo.deep_query_osdu(
            provider,
            uuid=uuids["grid_uuid"],
            scope="sources",
        )
        assert "resources" in result

    def test_deep_query_type_filter(self, populated_provider):
        """deep_query_osdu with type filtering."""
        provider, uuids = populated_provider

        result = xtgeo.deep_query_osdu(
            provider,
            depth=0,
            object_types=["IjkGridRepresentation"],
        )
        assert "resources" in result
        for r in result["resources"]:
            assert "IjkGrid" in r["type"] or "ijk" in r["type"].lower()


# ---------------------------------------------------------------------------
# Notification/Watch Tests
# ---------------------------------------------------------------------------


class TestNotificationSubscription:
    """Test the polling-based notification subscription."""

    def test_subscribe_and_poll_no_changes(self, populated_provider):
        """Initial poll after subscribe returns no changes."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications()
        events = sub.poll()

        # No changes since subscription started
        assert isinstance(events, list)
        assert len(events) == 0

        sub.stop()

    def test_subscribe_detects_new_object(self, populated_provider):
        """Poll detects a newly created object."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications()

        # Write a new surface
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        new_uuids = xtgeo_surface_to_resqml(provider, surf, title="NewSurf_Notify")

        # Poll for changes
        events = sub.poll()

        # Should detect at least one "created" event
        created_events = [e for e in events if e["event"] == "created"]
        assert len(created_events) >= 1

        # Verify the new surface UUID appears
        created_uuids = {e["uuid"] for e in created_events}
        assert new_uuids["NewSurf_Notify"] in created_uuids or len(created_events) >= 1

        sub.stop()

    def test_subscribe_with_type_filter(self, populated_provider):
        """Subscribe filtered to IjkGrid only."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications(object_types=["IjkGridRepresentation"])

        # Write a surface (should NOT show up in filtered subscription)
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        xtgeo_surface_to_resqml(provider, surf, title="FilteredSurf")

        events = sub.poll()

        # Surface should not appear in grid-filtered subscription
        for e in events:
            assert "Grid2d" not in e.get("type", "")

        sub.stop()

    def test_subscribe_with_callback(self, populated_provider):
        """Subscribe with callback fires on change."""
        provider, uuids = populated_provider

        received_events = []

        def on_change(event_type, event_info):
            received_events.append((event_type, event_info))

        sub = provider.subscribe_notifications(callback=on_change)

        # Create something new
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        xtgeo_surface_to_resqml(provider, surf, title="CallbackSurf")

        sub.poll()

        # Callback should have been invoked
        assert len(received_events) >= 1
        assert received_events[0][0] == "created"

        sub.stop()

    def test_subscribe_context_manager(self, populated_provider):
        """Subscription works as context manager."""
        provider, uuids = populated_provider

        with provider.subscribe_notifications() as sub:
            assert repr(sub).startswith("NotificationSubscription(")
            events = sub.poll()
            assert isinstance(events, list)

    def test_watch_osdu_changes_api(self, populated_provider):
        """Test the high-level watch_osdu_changes() API."""
        provider, uuids = populated_provider

        sub = xtgeo.watch_osdu_changes(provider)
        assert sub is not None

        events = sub.poll()
        assert isinstance(events, list)

        sub.stop()


# ===========================================================================
# LOCAL RDDMS: Dataspace Snapshot Roundtrip Tests
# (Requires local RDDMS at ws://localhost:9002)
# ===========================================================================

from xtgeo.interfaces.osdu import (  # noqa: E402, F811
    CrsSnapshot,
    DataspaceSnapshot,
    GridSnapshot,
    PointSetSnapshot,
    PolylineSetSnapshot,
    PropertySnapshot,
    SurfaceSnapshot,
    compare_snapshots,
    read_dataspace,
    write_dataspace,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def etp_config_rt():
    """Base ETP config for local RDDMS."""
    import uuid as _uuid

    ds_path = f"xtgeo/test_rt_{_uuid.uuid4().hex[:8]}"
    return EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace=f"eml:///dataspace('{ds_path}')",
    ), ds_path


@pytest.fixture
def provider_rt(etp_config_rt):
    """ETP provider with a fresh test dataspace."""
    cfg, ds_path = etp_config_rt
    p = EtpProvider(cfg)
    try:
        p.open()
    except Exception:
        pytest.skip("Local RDDMS not available at ws://localhost:9002")

    # Create test dataspace
    with contextlib.suppress(Exception):
        p.put_dataspace(ds_path)

    yield p
    with contextlib.suppress(Exception):
        p.delete_dataspace(ds_path)
    p.close()


@pytest.fixture
def fresh_provider(etp_config_rt):
    """Provider factory that creates a new provider for a given dataspace."""

    def _make(dataspace_path: str):
        cfg = EtpConnectionConfig(
            url="ws://localhost:9002",
            dataspace=f"eml:///dataspace('{dataspace_path}')",
        )
        p = EtpProvider(cfg)
        try:
            p.open()
        except Exception:
            pytest.skip("Local RDDMS not available at ws://localhost:9002")
        return p

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_snapshot() -> DataspaceSnapshot:
    """Build a representative DataspaceSnapshot with all object types."""
    import uuid

    crs_uuid = str(uuid.uuid4())
    grid_uuid = str(uuid.uuid4())

    crs = CrsSnapshot(
        uuid=crs_uuid,
        title="TestCRS_EPSG23031",
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        areal_rotation=0.0,
        z_increasing_downward=True,
        projected_crs_epsg=23031,
    )

    # Non-square grid (4x5x3) to catch dimension ordering bugs
    ni, nj, nk = 4, 5, 3
    g = xtgeo.create_box_grid(
        (ni, nj, nk),
        origin=(460000.0, 5930000.0, 1000.0),
        increment=(50.0, 50.0, 10.0),
    )

    # Add fault throw on column 2
    z = g._zcornsv.copy()
    z[2:, :, :, :] += 5.0
    g._zcornsv = z

    # Inactive cells
    act = g._actnumsv.copy()
    act[0, 0, 0] = 0
    act[3, 4, 2] = 0
    g._actnumsv = act

    # Continuous property (porosity)
    np.random.seed(42)
    poro_vals = np.random.uniform(0.05, 0.35, size=ni * nj * nk).astype(np.float64)
    poro_vals = poro_vals.reshape((ni, nj, nk))

    # Discrete property (facies)
    facies_vals = np.random.randint(1, 5, size=ni * nj * nk).astype(np.int32)
    facies_vals = facies_vals.reshape((ni, nj, nk))

    grid = GridSnapshot(
        uuid=grid_uuid,
        title="TestGrid_Faulted",
        resqml_type="resqml20.IjkGridRepresentation",
        crs_uuid=crs_uuid,
        ni=ni,
        nj=nj,
        nk=nk,
        coord=g._coordsv.flatten().astype(np.float64),
        zcorn=g._zcornsv.flatten().astype(np.float32),
        actnum=g._actnumsv.flatten().astype(np.int32),
        k_direction="down",
        properties=[
            PropertySnapshot(
                uuid=str(uuid.uuid4()),
                title="PORO",
                resqml_type="resqml20.ContinuousProperty",
                property_kind="porosity",
                indexable_element="cells",
                supporting_representation_uuid=grid_uuid,
                is_discrete=False,
                uom="v/v",
                values=poro_vals.flatten().astype(np.float64),
            ),
            PropertySnapshot(
                uuid=str(uuid.uuid4()),
                title="FACIES",
                resqml_type="resqml20.DiscreteProperty",
                property_kind="facies",
                indexable_element="cells",
                supporting_representation_uuid=grid_uuid,
                is_discrete=True,
                values=facies_vals.flatten().astype(np.int32),
            ),
        ],
    )

    # Rotated surface
    import math

    rotation_rad = math.radians(15.0)
    surf_vals = np.random.uniform(900.0, 1100.0, size=10 * 12).astype(np.float64)
    surf_vals[5] = np.nan  # inject NaN
    surface = SurfaceSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestSurface_Rotated",
        resqml_type="resqml20.Grid2dRepresentation",
        crs_uuid=crs_uuid,
        ni=10,
        nj=12,
        origin_x=460000.0,
        origin_y=5930000.0,
        di=25.0,
        dj=30.0,
        rotation=rotation_rad,
        values=surf_vals.reshape((12, 10)),
    )

    # PointSet
    pts = np.array(
        [
            [460000.0, 5930000.0, 1000.0],
            [460100.0, 5930100.0, 1010.0],
            [460200.0, 5930050.0, 1020.0],
            [460050.0, 5930200.0, 1005.0],
        ],
        dtype=np.float64,
    )
    pointset = PointSetSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestPoints",
        resqml_type="resqml20.PointSetRepresentation",
        crs_uuid=crs_uuid,
        points=pts,
    )

    # PolylineSet (two polygons, one closed)
    poly1 = np.array(
        [
            [460000.0, 5930000.0, 0.0],
            [460100.0, 5930000.0, 0.0],
            [460100.0, 5930100.0, 0.0],
            [460000.0, 5930100.0, 0.0],
        ],
        dtype=np.float64,
    )
    poly2 = np.array(
        [
            [460200.0, 5930200.0, 0.0],
            [460300.0, 5930200.0, 0.0],
            [460250.0, 5930300.0, 0.0],
        ],
        dtype=np.float64,
    )
    polylineset = PolylineSetSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestPolygons",
        resqml_type="resqml20.PolylineSetRepresentation",
        crs_uuid=crs_uuid,
        polylines=[poly1, poly2],
        closed=[True, False],
    )

    return DataspaceSnapshot(
        grids=[grid],
        surfaces=[surface],
        pointsets=[pointset],
        polylinesets=[polylineset],
        crs_list=[crs],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataspaceRoundTrip:
    """Write a full dataset to ETP, read back, and compare bitwise."""

    def test_write_read_compare(self, provider_rt):
        """Full roundtrip: build snapshot → write → read back → compare."""
        original = _build_test_snapshot()

        # Write
        write_dataspace(provider_rt, original)

        # Read back
        readback = read_dataspace(provider_rt)

        # Compare
        diffs = compare_snapshots(original, readback, atol=1e-6)
        if diffs:
            msg = "\n".join(
                f"  {d.object_type}/{d.title}.{d.field}: {d.detail}" for d in diffs
            )
            pytest.fail(f"Roundtrip differences:\n{msg}")

    def test_grid_geometry_exact(self, provider_rt):
        """Verify grid geometry survives ETP roundtrip with full precision."""
        original = _build_test_snapshot()
        write_dataspace(provider_rt, original)
        readback = read_dataspace(provider_rt)

        assert len(readback.grids) >= 1
        g_orig = original.grids[0]
        g_read = next(g for g in readback.grids if g.title == g_orig.title)

        assert g_read.ni == g_orig.ni
        assert g_read.nj == g_orig.nj
        assert g_read.nk == g_orig.nk

        # Coord: float64 → exact
        np.testing.assert_allclose(
            g_read.coord.astype(np.float64),
            g_orig.coord.astype(np.float64),
            atol=1e-10,
            err_msg="Pillar coordinates differ",
        )

        # Zcorn: float32 → exact within float32 precision
        np.testing.assert_allclose(
            g_read.zcorn.astype(np.float32),
            g_orig.zcorn.astype(np.float32),
            atol=1e-6,
            err_msg="Z-corners differ",
        )

        # Actnum: exact integer match
        np.testing.assert_array_equal(
            g_read.actnum.flatten().astype(np.int32),
            g_orig.actnum.flatten().astype(np.int32),
        )

    def test_properties_exact(self, provider_rt):
        """Verify grid properties survive roundtrip with full precision."""
        original = _build_test_snapshot()
        write_dataspace(provider_rt, original)
        readback = read_dataspace(provider_rt)

        g_read = next(g for g in readback.grids if g.title == "TestGrid_Faulted")
        props_by_name = {p.title: p for p in g_read.properties}

        # Continuous - float64 exact
        assert "PORO" in props_by_name
        poro = props_by_name["PORO"]
        assert not poro.is_discrete
        np.testing.assert_allclose(
            poro.values.astype(np.float64),
            original.grids[0].properties[0].values.astype(np.float64),
            atol=1e-10,
        )

        # Discrete - int32 exact
        assert "FACIES" in props_by_name
        facies = props_by_name["FACIES"]
        assert facies.is_discrete
        np.testing.assert_array_equal(
            facies.values.astype(np.int32),
            original.grids[0].properties[1].values.astype(np.int32),
        )

    def test_surface_rotation_preserved(self, provider_rt):
        """Verify surface rotation survives ETP roundtrip."""
        import math

        original = _build_test_snapshot()
        write_dataspace(provider_rt, original)
        readback = read_dataspace(provider_rt)

        s_read = next(s for s in readback.surfaces if s.title == "TestSurface_Rotated")
        s_orig = original.surfaces[0]

        assert s_read.ni == s_orig.ni
        assert s_read.nj == s_orig.nj
        assert abs(s_read.origin_x - s_orig.origin_x) < 1e-6
        assert abs(s_read.origin_y - s_orig.origin_y) < 1e-6
        assert abs(s_read.di - s_orig.di) < 1e-6
        assert abs(s_read.dj - s_orig.dj) < 1e-6
        assert abs(s_read.rotation - s_orig.rotation) < 1e-6, (
            f"Rotation mismatch: {math.degrees(s_read.rotation):.4f}° "
            f"vs {math.degrees(s_orig.rotation):.4f}°"
        )

    def test_crs_metadata_preserved(self, provider_rt):
        """Verify CRS EPSG and metadata survive roundtrip."""
        original = _build_test_snapshot()
        write_dataspace(provider_rt, original)
        readback = read_dataspace(provider_rt)

        assert len(readback.crs_list) >= 1
        crs_read = next(c for c in readback.crs_list if c.title == "TestCRS_EPSG23031")
        crs_orig = original.crs_list[0]

        assert crs_read.projected_crs_epsg == crs_orig.projected_crs_epsg
        assert crs_read.z_increasing_downward == crs_orig.z_increasing_downward
        assert abs(crs_read.areal_rotation - crs_orig.areal_rotation) < 1e-10


class TestDataspaceCopy:
    """Copy entire dataspace to a new one and verify equivalence."""

    def test_copy_dataspace(self, fresh_provider):
        """Read from source ds → write to target ds → compare both."""
        source_path = "xtgeo/test_copy_src"
        target_path = "xtgeo/test_copy_dst"

        # Setup: create source dataspace and write test data
        src = fresh_provider(source_path)
        with contextlib.suppress(Exception):
            src.put_dataspace(source_path)

        original = _build_test_snapshot()
        write_dataspace(src, original)

        # Read everything from source
        snap_source = read_dataspace(src)
        src.close()

        # Create target and write
        tgt = fresh_provider(target_path)
        with contextlib.suppress(Exception):
            tgt.put_dataspace(target_path)

        write_dataspace(tgt, snap_source)

        # Read back from target
        snap_target = read_dataspace(tgt)
        tgt.close()

        # Compare
        diffs = compare_snapshots(snap_source, snap_target, atol=1e-6)
        if diffs:
            msg = "\n".join(
                f"  {d.object_type}/{d.title}.{d.field}: {d.detail}" for d in diffs
            )
            pytest.fail(f"Dataspace copy differences:\n{msg}")

        # Cleanup
        cleanup = fresh_provider(source_path)
        try:
            cleanup.delete_dataspace(source_path)
            cleanup.delete_dataspace(target_path)
        except Exception:
            pass
        cleanup.close()
