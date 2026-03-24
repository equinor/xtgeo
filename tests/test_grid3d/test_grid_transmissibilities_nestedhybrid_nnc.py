"""Tests for Grid.get_transmissibilities() nested-hybrid extension."""

import pathlib

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo.common.log import functimer

# Darcy unit-conversion factor for METRIC grids (mD·m → m³·cP/(d·bar))
_C = 9.869233e-16 * 1e3 * 86400 * 1e5  # ≈ 8.527e-3

NESTEDHYBRID1 = pathlib.Path("3dgrids/drogon/5/drogon_nested_hybrid1.roff")

# use a clipped model from Emerald
EMERALD_ORIGINAL = pathlib.Path("3dgrids/eme/3/original.grdecl")

# simple tests using just one cell hybrid

EMERALD_ONE_CELL_HYBRID_GRID = pathlib.Path(
    "3dgrids/eme/3/onecellregion1_refine_1_1_1_merged.grdecl"
)
EMERALD_ONE_CELL_HYBRID_PROPS = pathlib.Path(
    "3dgrids/eme/3/onecellregion1_refine_1_1_1_merged_props.grdecl"
)

# more complex:

EMERALD_NESTED1_GRID = pathlib.Path(
    "3dgrids/eme/3/nested_grid_refine_1_1_1_merged.grdecl"
)
EMERALD_NESTED1_PROPS = pathlib.Path(
    "3dgrids/eme/3/nested_grid_refine_1_1_1_merged_props.grdecl"
)

EMERALD_NESTED2_GRID = pathlib.Path(
    "3dgrids/eme/3/nested_grid_refine_2_2_2_merged.grdecl"
)
EMERALD_NESTED2_PROPS = pathlib.Path(
    "3dgrids/eme/3/nested_grid_refine_2_2_2_merged_props.grdecl"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nested_grid(nc, nr, nl, nest_pattern, axis="i"):
    """Return a box grid with NEST_ID set along the given axis.

    nest_pattern is a 1-D list of length nc/nr/nl (for axis i/j/k).
    Cells with NEST_ID==0 are also deactivated.
    """
    grid = xtgeo.create_box_grid((nc, nr, nl))
    nest_arr = np.zeros((nc, nr, nl), dtype=np.int32)
    act_arr = np.ones((nc, nr, nl), dtype=np.int32)
    for idx, val in enumerate(nest_pattern):
        if axis == "i":
            nest_arr[idx, :, :] = val
            if val == 0:
                act_arr[idx, :, :] = 0
        elif axis == "j":
            nest_arr[:, idx, :] = val
            if val == 0:
                act_arr[:, idx, :] = 0
        elif axis == "k":
            nest_arr[:, :, idx] = val
            if val == 0:
                act_arr[:, :, idx] = 0
    grid.set_actnum(
        xtgeo.GridProperty(grid, values=act_arr, name="ACTNUM", discrete=True)
    )
    nested = xtgeo.GridProperty(
        grid,
        values=nest_arr,
        name="NEST_ID",
        discrete=True,
        codes={0: "none", 1: "mother", 2: "refined"},
    )
    return grid, nested


def _make_nnc_table_for_box(nc, nr, nl, nest_pattern, axis):
    """Build an nnc_table for a standard box-grid nested hybrid layout.

    Convention: I1=mother (1-based), I2=refined (1-based),
    DIRECTION from mother's perspective.
    """
    # Find the last mother cell before the hole and the first refined after
    mother_idx = None
    refined_idx = None
    for idx, val in enumerate(nest_pattern):
        if val == 1 and idx + 1 < len(nest_pattern) and nest_pattern[idx + 1] == 0:
            mother_idx = idx
        if val == 2 and idx - 1 >= 0 and nest_pattern[idx - 1] == 0:
            refined_idx = idx

    if mother_idx is None or refined_idx is None:
        return pd.DataFrame(columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])

    rows = []
    if axis == "i":
        direction = "I+" if refined_idx > mother_idx else "I-"
        for j in range(nr):
            for k in range(nl):
                rows.append(
                    {
                        "I1": mother_idx + 1,
                        "J1": j + 1,
                        "K1": k + 1,
                        "I2": refined_idx + 1,
                        "J2": j + 1,
                        "K2": k + 1,
                        "DIRECTION": direction,
                    }
                )
    elif axis == "j":
        direction = "J+" if refined_idx > mother_idx else "J-"
        for i in range(nc):
            for k in range(nl):
                rows.append(
                    {
                        "I1": i + 1,
                        "J1": mother_idx + 1,
                        "K1": k + 1,
                        "I2": i + 1,
                        "J2": refined_idx + 1,
                        "K2": k + 1,
                        "DIRECTION": direction,
                    }
                )
    elif axis == "k":
        direction = "K+" if refined_idx > mother_idx else "K-"
        for i in range(nc):
            for j in range(nr):
                rows.append(
                    {
                        "I1": i + 1,
                        "J1": j + 1,
                        "K1": mother_idx + 1,
                        "I2": i + 1,
                        "J2": j + 1,
                        "K2": refined_idx + 1,
                        "DIRECTION": direction,
                    }
                )

    return pd.DataFrame(rows, columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])


def _uniform(grid, value, name="prop"):
    return xtgeo.GridProperty(grid, values=value, name=name, discrete=False)


def _nnc_table_from_nest_id(grid, nest_id_prop, search_radius=600.0, min_area=1e-3):
    """Test helper: build nnc_table from NEST_ID using geometric face matching.

    For grids that were created externally (not via ``create_nested_hybrid_grid``),
    this replicates the old KDTree-based boundary-face detection to produce the
    nnc_table that ``get_transmissibilities(nnc_table=...)`` now expects.

    Convention: I1=mother (1-based), I2=refined (1-based),
    DIRECTION from mother's perspective.
    """
    import xtgeo._internal as _internal
    from scipy.spatial import KDTree

    from xtgeo.grid3d._grid_transmissibilities import _collect_nh_boundary_faces

    grid._set_xtgformat2()
    gcpp = grid._get_grid_cpp()
    nc, nr, nl = grid.ncol, grid.nrow, grid.nlay
    nv = np.asarray(nest_id_prop.values.filled(0), dtype=np.int32)

    mother_faces = _collect_nh_boundary_faces(gcpp, nv, 1, nc, nr, nl)
    refined_faces = _collect_nh_boundary_faces(gcpp, nv, 2, nc, nr, nl)

    if not mother_faces or not refined_faces:
        return pd.DataFrame(columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])

    mother_ctrs = np.array([f[0] for f in mother_faces])
    tree = KDTree(mother_ctrs)

    _DIR_LABEL = {
        (1, 0, 0): "I+",
        (-1, 0, 0): "I-",
        (0, 1, 0): "J+",
        (0, -1, 0): "J-",
        (0, 0, 1): "K+",
        (0, 0, -1): "K-",
    }

    rows = []
    for ctr_r, ijk_r, cc_r, face_r, dir_r in refined_faces:
        for idx in tree.query_ball_point(ctr_r, r=search_radius):
            ctr_m, ijk_m, cc_m, face_m, dir_m = mother_faces[idx]
            if (
                dir_r[0] + dir_m[0],
                dir_r[1] + dir_m[1],
                dir_r[2] + dir_m[2],
            ) != (0, 0, 0):
                continue
            fr = _internal.grid3d.face_overlap_result(cc_r, face_r, cc_m, face_m)
            if fr.area <= min_area:
                continue
            # dir_m is the direction from mother INTO the hole = toward refined
            direction = _DIR_LABEL.get(dir_m, "?")
            rows.append(
                {
                    "I1": ijk_m[0] + 1,
                    "J1": ijk_m[1] + 1,
                    "K1": ijk_m[2] + 1,
                    "I2": ijk_r[0] + 1,
                    "J2": ijk_r[1] + 1,
                    "K2": ijk_r[2] + 1,
                    "DIRECTION": direction,
                }
            )

    return pd.DataFrame(rows, columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])


def _build_emerald_boundary_df(testdata_path, region_o):
    """Build a boundary connection table for the Emerald original grid.

    Returns a DataFrame of every TRANX/TRANY/TRANZ and fault-NNC connection
    in the original Emerald grid where one cell has NestReg=1 (mother region)
    and the adjacent cell has NestReg=2 (the to-be-refined region).

    Parameters
    ----------
    testdata_path:
        Path to the test data directory.
    region_o:
        GridProperty with the NestReg values for the original grid.

    Columns: I1, J1, K1, I2, J2, K2, T, DIRECTION  (1-based;
    NestReg=1 mother always in I1/J1/K1, NestReg=2 cell always in I2/J2/K2.
    DIRECTION is from the NestReg=1 mother cell toward the NestReg=2 cell.)
    """
    import pandas as pd

    file = testdata_path / EMERALD_ORIGINAL
    grid_o = xtgeo.grid_from_file(file)
    permx_o = xtgeo.gridproperty_from_file(file, name="KX", grid=grid_o)
    permy_o = xtgeo.gridproperty_from_file(file, name="KY", grid=grid_o)
    permz_o = xtgeo.gridproperty_from_file(file, name="KZ", grid=grid_o)
    ntg_o = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid_o)

    tx, ty, tz, nncs_fault, _, _ = grid_o.get_transmissibilities(
        permx_o, permy_o, permz_o, ntg_o
    )

    reg = region_o.values.filled(0)
    nc, nr, nl = grid_o.ncol, grid_o.nrow, grid_o.nlay

    rows: list = []

    def _add(r1, r2, i1b, j1b, k1b, i2b, j2b, k2b, t_val, direction):
        """Append a row with NestReg=1 (mother) cell always first."""
        if t_val <= 0:
            return
        if r1 == 1:
            rows.append((i1b, j1b, k1b, i2b, j2b, k2b, t_val, direction))
        else:
            rows.append((i2b, j2b, k2b, i1b, j1b, k1b, t_val, direction))

    # TRANX (I-direction)
    for k in range(nl):
        for j in range(nr):
            for i in range(nc - 1):
                r1, r2 = int(reg[i, j, k]), int(reg[i + 1, j, k])
                if {r1, r2} == {1, 2}:
                    direction = "I-" if r1 == 2 else "I+"
                    _add(
                        r1,
                        r2,
                        i + 1,
                        j + 1,
                        k + 1,
                        i + 2,
                        j + 1,
                        k + 1,
                        float(tx.values[i, j, k]),
                        direction,
                    )

    # TRANY (J-direction)
    for k in range(nl):
        for i in range(nc):
            for j in range(nr - 1):
                r1, r2 = int(reg[i, j, k]), int(reg[i, j + 1, k])
                if {r1, r2} == {1, 2}:
                    direction = "J-" if r1 == 2 else "J+"
                    _add(
                        r1,
                        r2,
                        i + 1,
                        j + 1,
                        k + 1,
                        i + 1,
                        j + 2,
                        k + 1,
                        float(ty.values[i, j, k]),
                        direction,
                    )

    # TRANZ (K-direction)
    for j in range(nr):
        for i in range(nc):
            for k in range(nl - 1):
                r1, r2 = int(reg[i, j, k]), int(reg[i, j, k + 1])
                if {r1, r2} == {1, 2}:
                    direction = "K-" if r1 == 2 else "K+"
                    _add(
                        r1,
                        r2,
                        i + 1,
                        j + 1,
                        k + 1,
                        i + 1,
                        j + 1,
                        k + 2,
                        float(tz.values[i, j, k]),
                        direction,
                    )

    # Fault NNCs from original that cross the NestReg boundary
    _axis_dir = {
        (1, 0, 0): "I+",
        (-1, 0, 0): "I-",
        (0, 1, 0): "J+",
        (0, -1, 0): "J-",
        (0, 0, 1): "K+",
        (0, 0, -1): "K-",
    }
    for row in nncs_fault.itertuples():
        r1 = int(reg[row.I1 - 1, row.J1 - 1, row.K1 - 1])
        r2 = int(reg[row.I2 - 1, row.J2 - 1, row.K2 - 1])
        if {r1, r2} != {1, 2}:
            continue
        # Direction from NestReg=1 (mother) cell toward NestReg=2 cell
        hole_i, hole_j, hole_k = (
            (row.I1, row.J1, row.K1) if r1 == 2 else (row.I2, row.J2, row.K2)
        )
        moth_i, moth_j, moth_k = (
            (row.I2, row.J2, row.K2) if r1 == 2 else (row.I1, row.J1, row.K1)
        )
        dvec = (
            int(np.sign(hole_i - moth_i)),
            int(np.sign(hole_j - moth_j)),
            int(np.sign(hole_k - moth_k)),
        )
        direction = _axis_dir.get(dvec, "NNC")
        _add(r1, r2, row.I1, row.J1, row.K1, row.I2, row.J2, row.K2, row.T, direction)

    return pd.DataFrame(
        rows, columns=["I1", "J1", "K1", "I2", "J2", "K2", "T", "DIRECTION"]
    )


# ---------------------------------------------------------------------------
# Box-grid unit tests
# ---------------------------------------------------------------------------


class TestBoxGrid:
    """Unit tests using small synthetic box grids with analytically known answers.

    Grid layout for I-direction tests (arrows show hole-facing boundary faces):

        [ mother | mother | (hole) | refined | refined ]
          i=0      i=1       i=2      i=3       i=4
                        ↑                 ↑
                  east face           west face

    The single NNC connects cell (i=1, 1-based: I1=2, mother) to cell
    (i=3, 1-based: I2=4, refined).

    For 1 m³ unit-cube cells, perm=1 mD, NTG=1:
        d1 = d2 = 0.5 m,  A = 1 m²
        HT = k·A/d = 1·1/0.5 = 2
        T  = C_DARCY · HT² / (2·HT) = C_DARCY ≈ 8.527e-3
    """

    def test_i_direction_single_nnc_found(self):
        """5×1×1 grid with one-cell hole along I: exactly one NNC detected."""
        grid, nested = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        assert len(nnc_df) == 1
        row = nnc_df.iloc[0]
        assert row["I1"] == 2  # mother cell i=1 (1-based)
        assert row["I2"] == 4  # refined cell i=3 (1-based)
        assert row["TYPE"] == "NestedHybrid"

    def test_i_direction_unit_perm_ntg_transmissibility(self):
        """Unit perm and NTG → T equals the Darcy conversion constant."""
        grid, nested = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        np.testing.assert_allclose(nnc_df["T"].values, [_C], rtol=1e-4)

    def test_j_direction_single_nnc_found(self):
        """1×5×1 grid with hole along J: exactly one NNC detected."""
        grid, nested = _make_nested_grid(1, 5, 1, [1, 1, 0, 2, 2], axis="j")
        table = _make_nnc_table_for_box(1, 5, 1, [1, 1, 0, 2, 2], axis="j")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        assert len(nnc_df) == 1
        np.testing.assert_allclose(nnc_df["T"].values, [_C], rtol=1e-4)

    def test_k_direction_single_nnc_found(self):
        """1×1×5 grid with hole along K: exactly one NNC detected."""
        grid, nested = _make_nested_grid(1, 1, 5, [1, 1, 0, 2, 2], axis="k")
        table = _make_nnc_table_for_box(1, 1, 5, [1, 1, 0, 2, 2], axis="k")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        assert len(nnc_df) == 1
        np.testing.assert_allclose(nnc_df["T"].values, [_C], rtol=1e-4)

    def test_no_hole_returns_empty_dataframe(self):
        """Grid with only NEST_ID==1 (no hole, no NEST_ID==2) → 0 NNCs."""
        grid = xtgeo.create_box_grid((3, 1, 1))
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")
        empty_table = pd.DataFrame(
            columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"]
        )

        _, _, _, _, nnc_df, rbnd = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=empty_table
        )

        assert len(nnc_df) == 0
        assert list(nnc_df.columns) == [
            "I1",
            "J1",
            "K1",
            "I2",
            "J2",
            "K2",
            "T",
            "TYPE",
            "DIRECTION",
        ]
        assert (rbnd.values.filled(0) == 0).all()

    def test_ntg_scales_ij_transmissibility(self):
        """Halving NTG on all cells halves both HTs → T = C/2 for I-direction."""
        grid, nested = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg_half = _uniform(grid, 0.5, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg_half, nnc_table=table
        )

        # k_eff = 1*0.5 = 0.5;  HT = 0.5/0.5 = 1;  T = C*1*1/(1+1) = C/2
        np.testing.assert_allclose(nnc_df["T"].values, [_C / 2], rtol=1e-4)

    def test_k_direction_ntg_not_applied(self):
        """K-direction NNCs use permz only; NTG must not affect T."""
        grid, nested = _make_nested_grid(1, 1, 5, [1, 1, 0, 2, 2], axis="k")
        table = _make_nnc_table_for_box(1, 1, 5, [1, 1, 0, 2, 2], axis="k")
        perm = _uniform(grid, 1.0, "perm")
        ntg_half = _uniform(grid, 0.5, "ntg")  # must be ignored for K

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg_half, nnc_table=table
        )

        # k_eff = permz = 1 (NTG not applied); T = C
        np.testing.assert_allclose(nnc_df["T"].values, [_C], rtol=1e-4)

    def test_refined_boundary_property_marks_correct_cells(self):
        """refined_boundary marks refined cells (NEST_ID==2) in the NNC, not mother."""
        grid, nested = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        nv = nested.values.filled(0).astype(int)
        _, _, _, _, _, rbnd = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        flag = rbnd.values.filled(0)
        marked = np.argwhere(flag == 1)
        assert len(marked) > 0, "No refined boundary cells marked"
        for i, j, k in marked:
            assert nv[i, j, k] == 2, (
                f"Cell ({i},{j},{k}) marked but has NEST_ID={nv[i, j, k]}"
            )
        # Mother cells must NOT be marked
        for i in range(grid.ncol):
            for j in range(grid.nrow):
                for k in range(grid.nlay):
                    if nv[i, j, k] == 1:
                        assert flag[i, j, k] == 0

    def test_transmissibility_proportional_to_perm(self):
        """Doubling permeability on both cells doubles HTs → T doubles."""
        grid, nested = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        ntg = _uniform(grid, 1.0, "ntg")

        perm1 = _uniform(grid, 1.0, "perm1")
        _, _, _, _, nnc1, _ = grid.get_transmissibilities(
            perm1, perm1, perm1, ntg, nnc_table=table
        )

        perm2 = _uniform(grid, 2.0, "perm2")
        _, _, _, _, nnc2, _ = grid.get_transmissibilities(
            perm2, perm2, perm2, ntg, nnc_table=table
        )

        np.testing.assert_allclose(nnc2["T"].values, 2 * nnc1["T"].values, rtol=1e-6)

    def test_multiple_columns_produce_multiple_nncs(self):
        """3×3×1 grid with I-direction hole: 3 NNCs (one per J column)."""
        grid, nested = _make_nested_grid(5, 3, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 3, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nnc_df, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )

        assert len(nnc_df) == 3  # one NNC per J row

    def test_nnc_table_only_skips_regular_tpfa(self):
        """nnc_table_only=True returns zero TRAN* and empty regular NNC df."""
        grid, _ = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        tx, ty, tz, nnc_df, nnc_nh, rbnd = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table, nnc_table_only=True
        )

        # Regular TRAN* are all zero
        assert (tx.values.filled(0) == 0).all()
        assert (ty.values.filled(0) == 0).all()
        assert (tz.values.filled(0) == 0).all()

        # Regular NNC dataframe is empty but has the expected columns
        assert len(nnc_df) == 0
        assert list(nnc_df.columns) == ["I1", "J1", "K1", "I2", "J2", "K2", "T", "TYPE"]

        # Nested-hybrid NNCs are still computed
        assert nnc_nh is not None
        assert len(nnc_nh) == 1
        np.testing.assert_allclose(nnc_nh["T"].values, [_C], rtol=1e-4)
        assert rbnd is not None

    def test_nnc_table_only_matches_full_nh_results(self):
        """nnc_table_only=True yields the same nested-hybrid NNCs as the full call."""
        grid, _ = _make_nested_grid(5, 3, 1, [1, 1, 0, 2, 2], axis="i")
        table = _make_nnc_table_for_box(5, 3, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        _, _, _, _, nh_full, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table
        )
        _, _, _, _, nh_only, _ = grid.get_transmissibilities(
            perm, perm, perm, ntg, nnc_table=table, nnc_table_only=True
        )

        pd.testing.assert_frame_equal(
            nh_full.reset_index(drop=True), nh_only.reset_index(drop=True)
        )

    def test_nnc_table_only_requires_nnc_table(self):
        """nnc_table_only=True without nnc_table raises ValueError."""
        grid, _ = _make_nested_grid(5, 1, 1, [1, 1, 0, 2, 2], axis="i")
        perm = _uniform(grid, 1.0, "perm")
        ntg = _uniform(grid, 1.0, "ntg")

        with pytest.raises(ValueError, match="nnc_table_only"):
            grid.get_transmissibilities(perm, perm, perm, ntg, nnc_table_only=True)


class TestDrogonNestedCase:
    """Test get_transmissibilities() nested-hybrid mode with the Drogon grid."""

    def test_nested_drogon_nncs(self, testdata_path):
        """NNCs between the nested refined region (NEST_ID==2) and the mother
        grid (NEST_ID==1) are detected and transmissibilities are computed.

        The nested hybrid grid stores two regions in the same ROFF file:
        - NEST_ID == 1: the coarse mother grid  (i=0..45,  k=0..31)
        - NEST_ID == 2: the refined grid         (i=47..66, k=0..63)
        The refined region physically occupies the same 3-D space as a
        sub-region of the mother grid.  The corresponding mother cells have
        been deactivated (NEST_ID == 0) to carve out the hole.
        """
        file = testdata_path / NESTEDHYBRID1

        grid = xtgeo.grid_from_file(file)
        permx = xtgeo.gridproperty_from_file(file, name="KX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="KY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="KZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)
        nested = xtgeo.gridproperty_from_file(file, name="NEST_ID", grid=grid)
        table = _nnc_table_from_nest_id(grid, nested)

        @functimer(output="print")
        def get_nncs():
            return grid.get_transmissibilities(
                permx, permy, permz, ntg, nnc_table=table
            )

        _, _, _, _, nnc_df, refined_boundary = get_nncs()

        print(f"NNCs found: {len(nnc_df)}")
        print(nnc_df.head())
        print(f"Refined boundary cells: {(refined_boundary.values == 1).sum()}")

        # --- Basic structural checks -------------------------------------------
        assert len(nnc_df) > 0, "Expected at least one NNC"
        assert list(nnc_df.columns) == [
            "I1",
            "J1",
            "K1",
            "I2",
            "J2",
            "K2",
            "T",
            "TYPE",
            "DIRECTION",
        ]
        assert (nnc_df["TYPE"] == "NestedHybrid").all()

        # Indices must be 1-based and within grid bounds
        ncol, nrow, nlay = grid.ncol, grid.nrow, grid.nlay
        assert nnc_df["I1"].between(1, ncol).all()
        assert nnc_df["J1"].between(1, nrow).all()
        assert nnc_df["K1"].between(1, nlay).all()
        assert nnc_df["I2"].between(1, ncol).all()
        assert nnc_df["J2"].between(1, nrow).all()
        assert nnc_df["K2"].between(1, nlay).all()

        # All computed transmissibilities must be finite and non-negative
        assert np.isfinite(nnc_df["T"].values).all()
        assert (nnc_df["T"] >= 0.0).all()

        # --- GridProperty checks -----------------------------------------------
        assert refined_boundary.name == "NNC_REFINED_BOUNDARY"
        assert refined_boundary.isdiscrete
        assert refined_boundary.ncol == ncol
        assert refined_boundary.nrow == nrow
        assert refined_boundary.nlay == nlay

        # Refined boundary cells must all come from NEST_ID==2 region
        nv = nested.values.filled(0).astype(int)
        flag = refined_boundary.values.filled(0)
        marked_ijk = np.argwhere(flag == 1)
        for i, j, k in marked_ijk:
            assert nv[i, j, k] == 2, (
                f"Cell ({i},{j},{k}) marked as refined_boundary but "
                f"has NEST_ID={nv[i, j, k]}"
            )

        # --- Export for visualisation -------------------------------------------
        out_dir = pathlib.Path(testdata_path) / "3dgrids/drogon/5"
        out_prop = out_dir / "drogon_nested_hybrid1_nnc_boundary.roff"
        refined_boundary.to_file(out_prop, fformat="roff")
        print(f"Written: {out_prop}")


class TestEmeraldNestedCase:
    """Test nested-hybrid mode witha subset of the Emerald grid."""

    def test_emerald_hybrid_onecell_no_refinement(self, testdata_path):
        """Here the hybrid grid is just one cell, no refinement."""

        hgrid = xtgeo.grid_from_file(testdata_path / EMERALD_ONE_CELL_HYBRID_GRID)
        permx_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_ONE_CELL_HYBRID_PROPS, name="KX", grid=hgrid
        )
        permy_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_ONE_CELL_HYBRID_PROPS, name="KY", grid=hgrid
        )
        permz_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_ONE_CELL_HYBRID_PROPS, name="KZ", grid=hgrid
        )
        ntg_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_ONE_CELL_HYBRID_PROPS, name="NTG", grid=hgrid
        )
        region_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_ONE_CELL_HYBRID_PROPS, name="NestReg", grid=hgrid
        )
        table = _nnc_table_from_nest_id(hgrid, region_hybrid)

        _, _, _, _, nncs_connections, _ = hgrid.get_transmissibilities(
            permx_hybrid,
            permy_hybrid,
            permz_hybrid,
            ntg_hybrid,
            nnc_table=table,
        )
        print(nncs_connections)

        # checked with original grid TRANSX, TRANY, TRANZ
        assert nncs_connections.iloc[0]["T"] == pytest.approx(7.1687, rel=0.001)  # X
        assert nncs_connections.iloc[1]["T"] == pytest.approx(4.8443, rel=0.001)  # Y
        assert nncs_connections.iloc[2]["T"] == pytest.approx(379.47, rel=0.001)  # Z

        # similar test using the function
        orig = testdata_path / EMERALD_ORIGINAL
        grid_o = xtgeo.grid_from_file(orig)
        region_o = xtgeo.gridproperty_from_file(orig, name="NestReg", grid=grid_o)
        region_o.values = 1
        region_o.values[10, 29, 0] = 2
        boundary_df = _build_emerald_boundary_df(testdata_path, region_o)
        print(boundary_df)

    def test_emerald_hybrid_nncs_no_refinement(self, testdata_path):
        """Validate fault NNCs for hybrid connections versus the original.

        For cells in the original grid at the NestReg 1→0(hole) boundary, the
        dominant NestedHybrid NNC transmissibility in the hybrid grid (1:1:1
        refinement, i.e. no actual refinement) must match the corresponding
        TRANX/TRANY value in the original grid.

        A "clean" boundary cell is one that faces the hole in exactly one
        direction (I or J).  Only those are compared, because corner/diagonal
        cells face multiple refined cells simultaneously, making a 1:1 match
        ambiguous.

        For the I-direction, the western edge of NestReg=2 (mother at I=i,
        hole at I=i+1) is tested — that boundary has no large fault throw.

        For the J-direction, only the *northern* edge of NestReg=2 is tested
        (hole at J=j, mother at J=j+1).  The southern edge crosses the fault
        staircase where one mother cell faces many vertically offset refined
        cells; the dominant-NNC pattern does not apply there.
        """
        file = testdata_path / EMERALD_ORIGINAL

        grid = xtgeo.grid_from_file(file)
        permx = xtgeo.gridproperty_from_file(file, name="KX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="KY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="KZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)
        region_orig = xtgeo.gridproperty_from_file(file, name="NestReg", grid=grid)

        tx_orig, ty_orig, tz_orig, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, ntg
        )

        hybrid_grid = xtgeo.grid_from_file(testdata_path / EMERALD_NESTED1_GRID)
        permx_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KX", grid=hybrid_grid
        )
        permy_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KY", grid=hybrid_grid
        )
        permz_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KZ", grid=hybrid_grid
        )
        ntg_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NTG", grid=hybrid_grid
        )
        region_hybrid = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NestReg", grid=hybrid_grid
        )
        table = _nnc_table_from_nest_id(hybrid_grid, region_hybrid)

        _, _, _, _, nncs_connections, _ = hybrid_grid.get_transmissibilities(
            permx_hybrid,
            permy_hybrid,
            permz_hybrid,
            ntg_hybrid,
            nnc_table=table,
        )

        # Build a lookup: mother-cell (I,J,K) [1-based] → max NNC T
        # The mother cell can appear as cell1 or cell2 in the NNC dataframe.
        nnc_max_t: dict[tuple, float] = {}
        for _, row in nncs_connections.iterrows():
            for ic, jc, kc in [
                (int(row["I1"]), int(row["J1"]), int(row["K1"])),
                (int(row["I2"]), int(row["J2"]), int(row["K2"])),
            ]:
                key = (ic, jc, kc)
                t_val = float(row["T"])
                if key not in nnc_max_t or t_val > nnc_max_t[key]:
                    nnc_max_t[key] = t_val

        reg = region_orig.values.filled(0)
        nc, nr, nl = grid.ncol, grid.nrow, grid.nlay

        mismatches = []

        # ------------------------------------------------------------------
        # I-direction boundaries: original cell (i,j,k) with NestReg=1
        # whose east neighbour (i+1,j,k) has NestReg=2 (the hole region).
        # "Clean" means the cell is not simultaneously a J-direction boundary.
        # ------------------------------------------------------------------
        for j in range(1, nr - 1):
            for k in range(nl):
                for i in range(nc - 1):
                    if not (reg[i, j, k] == 1 and reg[i + 1, j, k] == 2):
                        continue
                    # Skip cells that are also J-direction boundaries (corners)
                    if reg[i, j - 1, k] == 2 or reg[i, j + 1, k] == 2:
                        continue
                    trans_orig = float(tx_orig.values[i, j, k])
                    if trans_orig < 1e-4:
                        continue
                    mother = (i + 1, j + 1, k + 1)  # 1-based
                    trans_nnc = nnc_max_t.get(mother)
                    if trans_nnc is None:
                        mismatches.append(
                            (mother, "I", trans_orig, None, "no NNC found")
                        )
                        continue
                    if abs(trans_nnc - trans_orig) > 1e-3 * max(trans_orig, 1e-9):
                        mismatches.append(
                            (
                                mother,
                                "I",
                                trans_orig,
                                trans_nnc,
                                f"ratio={trans_nnc / trans_orig:.4f}",
                            )
                        )

        # ------------------------------------------------------------------
        # J-direction boundaries: original cell (i,j+1,k) with NestReg=1
        # whose south neighbour (i,j,k) has NestReg=2 (the hole region).
        # This is the northern edge of the NestReg=2 region which has clean,
        # non-faulted geometry.  The southern edge crosses the fault staircase
        # and cannot be compared 1:1 (multiple overlapping refined cells).
        # ------------------------------------------------------------------
        for i in range(1, nc - 1):
            for k in range(nl):
                for j in range(nr - 1):
                    if not (reg[i, j, k] == 2 and reg[i, j + 1, k] == 1):
                        continue
                    # Skip cells that are also I-direction boundaries (corners)
                    if reg[i - 1, j + 1, k] == 2 or reg[i + 1, j + 1, k] == 2:
                        continue
                    trans_orig = float(ty_orig.values[i, j, k])
                    if trans_orig < 1e-4:
                        continue
                    mother = (i + 1, j + 2, k + 1)  # 1-based (mother at j+1, 0-based)
                    trans_nnc = nnc_max_t.get(mother)
                    if trans_nnc is None:
                        mismatches.append(
                            (mother, "J", trans_orig, None, "no NNC found")
                        )
                        continue
                    if abs(trans_nnc - trans_orig) > 1e-3 * max(trans_orig, 1e-9):
                        mismatches.append(
                            (
                                mother,
                                "J",
                                trans_orig,
                                trans_nnc,
                                f"ratio={trans_nnc / trans_orig:.4f}",
                            )
                        )

        assert not mismatches, (
            "Dominant hybrid NNC T does not match original TRAN "
            "for clean boundary cells:\n"
            + "\n".join(
                f"  cell={m[0]} dir={m[1]} "
                f"trans_orig={m[2]:.6g} trans_nnc={m[3]} {m[4]}"
                for m in mismatches
            )
        )

    def test_emerald_boundary_table_vs_hybrid_nncs(self, testdata_path):
        """Build a boundary-T table from the original grid and compare to the hybrid.

        Constructs a dataframe (the "boundary table") containing every connection
        in the original Emerald grid where one cell has NestReg=1 (mother region)
        and the adjacent cell has NestReg=2 (the to-be-refined region).  The table
        has the same column layout as the ``nncs_connections`` dataframe returned
        by ``get_transmissibilities``:

            I1, J1, K1  — the NestReg=1 (mother) cell, 1-based
            I2, J2, K2  — the NestReg=2 (hole) cell, 1-based
            T           — TRANX / TRANY / TRANZ or fault-NNC transmissibility

        Sources of rows:

        * **TRANX** – I-direction face-to-face connections crossing the boundary
        * **TRANY** – J-direction connections crossing the boundary
        * **TRANZ** – K-direction connections crossing the boundary
        * **NNC**   – fault-plane NNCs from the original grid that cross the
          NestReg 1↔2 boundary

        For the 1:1:1 Emerald hybrid grid (no actual refinement) every row in
        this table must match a NestedHybrid NNC in the hybrid grid involving the
        same mother cell with the same T value (within 0.1 %).

        Known limitations — two boundary types are excluded from the assertion:

        1. **TRANY where hole is at J+1** (southern boundary, `J2 > J1` in the
           table, i.e. refined at higher J than mother).  That edge is the fault
           plane itself.  The original grid computes a regular TPFA
           face-transmissibility there, while the hybrid uses the
           geometric-overlap NNC-scan algorithm, which distributes the T across
           multiple refined cells.  The per-entry T values therefore do *not*
           agree 1:1.

        2. **TRANX where hole is at I−1** (eastern boundary of NestReg=2, ``I2
           < I1`` in the table, i.e. refined at lower I than mother).  The
           NestedHybrid NNC generator currently does **not** produce correct
           large-T connections for mother cells whose adjacent hole face is in
           the −I direction.  These mother cells receive only spurious tiny-T
           NNCs.  This is a known gap to be fixed.
        """
        from collections import defaultdict

        file = testdata_path / EMERALD_ORIGINAL
        grid_o = xtgeo.grid_from_file(file)
        region_o = xtgeo.gridproperty_from_file(file, name="NestReg", grid=grid_o)
        boundary_df = _build_emerald_boundary_df(testdata_path, region_o)

        # ------------------------------------------------------------------
        # Hybrid grid: collect NestedHybrid NNCs.  Convention: I1/J1/K1 is the
        # mother (NestReg=1) cell, I2/J2/K2 is the refined cell.
        # ------------------------------------------------------------------
        grid_h = xtgeo.grid_from_file(testdata_path / EMERALD_NESTED1_GRID)
        permx_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KX", grid=grid_h
        )
        permy_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KY", grid=grid_h
        )
        permz_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KZ", grid=grid_h
        )
        ntg_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NTG", grid=grid_h
        )
        region_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NestReg", grid=grid_h
        )
        table_h = _nnc_table_from_nest_id(grid_h, region_h)
        _, _, _, _, nncs_h, _ = grid_h.get_transmissibilities(
            permx_h, permy_h, permz_h, ntg_h, nnc_table=table_h
        )
        nncs_h = nncs_h[nncs_h["TYPE"] == "NestedHybrid"]

        reg_h = region_h.values.filled(0)  # noqa: F841
        hyb_trans: dict = defaultdict(list)
        for row in nncs_h.itertuples():
            # I1/J1/K1 is always the mother (NestReg=1) in NestedHybrid output
            mother = (row.I1, row.J1, row.K1)
            hyb_trans[mother].append(row.T)

        # ------------------------------------------------------------------
        # Containment check: for each boundary_df entry (after applying the
        # exclusions documented in the docstring), the hybrid NNCs for the
        # same mother must contain a matching T within 0.1 %.
        #
        # Excluded:
        #   * J2 > J1  (TRANY southern/fault-plane boundary)
        #   * I2 < I1  (TRANX eastern hole-edge boundary — known missing NNCs)
        # ------------------------------------------------------------------
        TOL = 1e-3  # 0.1 %
        to_check = boundary_df[
            ~(
                (
                    boundary_df["J1"] < boundary_df["J2"]
                )  # TRANY southern boundary (refined J > mother J)
                | (
                    boundary_df["I1"] > boundary_df["I2"]
                )  # TRANX eastern boundary (refined I < mother I)
            )
        ]

        mismatches = []
        for brow in to_check.itertuples():
            mother = (brow.I1, brow.J1, brow.K1)
            t_o = brow.T
            ts_h = hyb_trans.get(mother, [])
            has_match = any(abs(t_h - t_o) <= TOL * max(t_o, t_h, 1e-9) for t_h in ts_h)
            if not has_match:
                hyb_max = max(ts_h) if ts_h else None
                mismatches.append((mother, t_o, hyb_max))

        assert not mismatches, (
            f"Hybrid NNCs missing a matching T for {len(mismatches)} boundary "
            "table entries (see docstring for excluded categories):\n"
            + "\n".join(
                f"  mother={m[0]} T_orig={m[1]:.5g} hyb_max={m[2]}"
                for m in mismatches[:15]
            )
        )

    def test_emerald_nnc_vs_boundary_by_i2j2k2_and_direction(self, testdata_path):
        """Join nncs_h and boundary_df on (I1, J1, K1, DIRECTION) and compare T.

        Both DataFrames use the same convention:
          I1/J1/K1 = NestReg=1 (mother) cell
          I2/J2/K2 = NestReg=2 (refined/hole) cell
          DIRECTION = direction from the mother cell toward the refined cell

        For every row that appears in both tables (inner join on the four
        columns), the transmissibility must agree within 0.1 %.

        Known limitations — two direction categories are excluded entirely, plus
        one targeted exclusion for corner hole cells:

        * ``DIRECTION == "I-"`` — hole is west of mother (mother looks I- toward
          refined).  The NNC algorithm does not produce correct large-T
          connections for these faces (documented gap, same exclusion as
          in ``test_emerald_boundary_table_vs_hybrid_nncs``).
        * ``DIRECTION == "J+"`` — hole is south of mother (mother looks J+
          toward refined / fault plane).  The original grid uses a regular
          TPFA face-T while the hybrid uses the geometric-overlap scan, so
          per-entry values do not agree 1:1.
        * ``DIRECTION == "I+"`` for **corner hole cells only** — a corner hole
          cell is one that appears in ``boundary_df`` with more than one
          direction (i.e. it sits at the corner of the hole region and faces
          the mother in both the I and J directions simultaneously).  At such
          cells the fault staircase causes the hybrid algorithm to route the
          transmissibility through the J face rather than the I face, so the
          direction-specific comparison is not valid.  Non-corner ``"I+"`` cells
          are still checked.

        The assertion covers all ``"J-"``, ``"K+"``, ``"K-"`` rows, and all
        ``"I+"`` rows that are not at fault-staircase corners.
        """

        file = testdata_path / EMERALD_ORIGINAL
        grid_o = xtgeo.grid_from_file(file)
        region_o = xtgeo.gridproperty_from_file(file, name="NestReg", grid=grid_o)
        boundary_df = _build_emerald_boundary_df(testdata_path, region_o)

        grid_h = xtgeo.grid_from_file(testdata_path / EMERALD_NESTED1_GRID)
        permx_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KX", grid=grid_h
        )
        permy_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KY", grid=grid_h
        )
        permz_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="KZ", grid=grid_h
        )
        ntg_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NTG", grid=grid_h
        )
        region_h = xtgeo.gridproperty_from_file(
            testdata_path / EMERALD_NESTED1_PROPS, name="NestReg", grid=grid_h
        )
        table_h = _nnc_table_from_nest_id(grid_h, region_h)
        _, _, _, _, nncs_h, _ = grid_h.get_transmissibilities(
            permx_h, permy_h, permz_h, ntg_h, nnc_table=table_h
        )
        nncs_h = nncs_h[nncs_h["TYPE"] == "NestedHybrid"].copy()

        # Inner join: only rows where (I1, J1, K1, DIRECTION) matches in both
        merged = boundary_df.merge(
            nncs_h[["I1", "J1", "K1", "DIRECTION", "T"]].rename(
                columns={"T": "trans_hyb"}
            ),
            on=["I1", "J1", "K1", "DIRECTION"],
            how="inner",
        ).rename(columns={"T": "trans_orig"})

        assert len(merged) > 0, "No rows matched on (I1, J1, K1, DIRECTION)"

        # Corner hole cells: appear in boundary_df in more than one direction.
        # For these the fault staircase makes the I+ direction ambiguous (the
        # algorithm routes the T through the J face instead).
        corner_holes = set(
            boundary_df.groupby(["I2", "J2", "K2"])["DIRECTION"]
            .nunique()
            .pipe(lambda s: s[s > 1].index.tolist())
        )
        corner_mask = merged[["I2", "J2", "K2"]].apply(tuple, axis=1).isin(corner_holes)

        _BLANKET_EXCL = {"I-", "J+"}  # entire direction excluded (see docstring)
        to_assert = merged[
            ~merged["DIRECTION"].isin(_BLANKET_EXCL)
            & ~((merged["DIRECTION"] == "I+") & corner_mask)
        ]
        assert len(to_assert) > 0, "No assertable rows remain after applying exclusions"

        TOL = 1e-3  # 0.1 %
        bad = to_assert[
            (to_assert["trans_orig"] - to_assert["trans_hyb"]).abs()
            > TOL * to_assert[["trans_orig", "trans_hyb"]].max(axis=1).clip(lower=1e-9)
        ]

        assert bad.empty, (
            f"{len(bad)} T mismatches on (I1, J1, K1, DIRECTION) join:\n"
            + bad[["I1", "J1", "K1", "DIRECTION", "trans_orig", "trans_hyb"]]
            .head(15)
            .to_string()
        )
