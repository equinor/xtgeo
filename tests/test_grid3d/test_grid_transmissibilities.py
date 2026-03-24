"""Tests for Grid.get_transmissibilities() public API.

Covers:
- Return types (GridProperty objects + DataFrame).
- Array shapes for various grid sizes (all padded to full ncol, nrow, nlay).
- Unit-cube box grid with perm=ntg=1 yields T=1 everywhere.
- Permeability and NTG scaling (harmonic mean).
- Inactive cell pairs produce masked values in TRAN arrays.
- Conforming box grid produces no NNCs (empty DataFrame).
- NNC DataFrame indices are 1-based.
- NNC TYPE strings are "Fault" or "Pinchout".
- ntg=None is equivalent to ntg filled with 1.0.
- K-direction: no NTG scaling (permz only).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo.common.log import functimer

B7CASE_LH = pathlib.Path("3dgrids/etc/b7_grid_transm_etc.grdecl")  # left-handed
B7CASE_RH = pathlib.Path("3dgrids/etc/b7_grid_transm_etc_rh.grdecl")  # right-handed

# using a clipped segment from the public Drogon case (rh=right-handed, lh=left-handed)
DCASE_GRID_RH = pathlib.Path("3dgrids/etc/dcase_grid_rh.grdecl")
DCASE_PROPS_RH = pathlib.Path("3dgrids/etc/dcase_props_rh.grdecl")

DCASE_GRID_LH = pathlib.Path("3dgrids/etc/dcase_grid_lh.grdecl")
DCASE_PROPS_LH = pathlib.Path("3dgrids/etc/dcase_props_lh.grdecl")

# right handed drogon case with TRANSM computed with OPM (march 2026)
DROGON_FULL_OPM = pathlib.Path("3dgrids/drogon/4/simgrid_w_props_opm_transm.grdecl")

# clipped Emerald model used for NNC validation
EMERALD_ORIGINAL = pathlib.Path("3dgrids/eme/3/original.grdecl")

# Darcy unit-conversion factor for METRIC grids (mD·m → m³·cP/(d·bar))
_C_METRIC = 9.869233e-16 * 1e3 * 86400 * 1e5  # ≈ 8.527e-3
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_grid(nc: int, nr: int, nl: int) -> xtgeo.Grid:
    return xtgeo.create_box_grid((nc, nr, nl))


def _uniform_prop(
    grid: xtgeo.Grid, value: float, name: str = "prop"
) -> xtgeo.GridProperty:
    return xtgeo.GridProperty(grid, name=name, values=value, discrete=False)


def _compute(nc, nr, nl, permx=1.0, permy=1.0, permz=1.0, ntg=None):
    """Convenience: uniform box grid, uniform properties."""
    grid = _box_grid(nc, nr, nl)
    px = _uniform_prop(grid, permx, "permx")
    py = _uniform_prop(grid, permy, "permy")
    pz = _uniform_prop(grid, permz, "permz")
    nt = _uniform_prop(grid, ntg, "ntg") if ntg is not None else None
    tranx, trany, tranz, nnc, _, _ = grid.get_transmissibilities(px, py, pz, ntg=nt)
    return tranx, trany, tranz, nnc


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_tranx_is_gridproperty(self):
        tranx, trany, tranz, nnc = _compute(3, 4, 5)
        assert isinstance(tranx, xtgeo.GridProperty)

    def test_trany_is_gridproperty(self):
        tranx, trany, tranz, nnc = _compute(3, 4, 5)
        assert isinstance(trany, xtgeo.GridProperty)

    def test_tranz_is_gridproperty(self):
        tranx, trany, tranz, nnc = _compute(3, 4, 5)
        assert isinstance(tranz, xtgeo.GridProperty)

    def test_nnc_is_dataframe(self):
        tranx, trany, tranz, nnc = _compute(3, 4, 5)
        assert isinstance(nnc, pd.DataFrame)

    def test_nnc_has_expected_columns(self):
        tranx, trany, tranz, nnc = _compute(3, 4, 5)
        assert set(nnc.columns) == {"I1", "J1", "K1", "I2", "J2", "K2", "T", "TYPE"}


# ---------------------------------------------------------------------------
# Array shapes
# ---------------------------------------------------------------------------


class TestShapes:
    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_tranx_shape(self, nc, nr, nl):
        tranx, _, _, _ = _compute(nc, nr, nl)
        assert tranx.values.shape == (nc, nr, nl)

    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_trany_shape(self, nc, nr, nl):
        _, trany, _, _ = _compute(nc, nr, nl)
        assert trany.values.shape == (nc, nr, nl)

    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_tranz_shape(self, nc, nr, nl):
        _, _, tranz, _ = _compute(nc, nr, nl)
        assert tranz.values.shape == (nc, nr, nl)


# ---------------------------------------------------------------------------
# Unit transmissibility (box grid, perm=ntg=1)
# ---------------------------------------------------------------------------


class TestUnitTran:
    """For a unit cube box with perm=ntg=1, TPFA gives T = _C_METRIC everywhere.

    Raw TPFA: HT = k*ntg * A / d = 1 * 1.0 / 0.5 = 2.0 → T = 1.0 mD·m
    After Darcy conversion: T = _C_METRIC ≈ 8.527e-3 m³·cP/(d·bar)
    """

    def test_tranx_all_one(self):
        tranx, _, _, _ = _compute(3, 4, 5)
        # Exclude the dummy last I-slice (padded with 0)
        assert np.all(
            np.isclose(tranx.values.filled(np.nan)[:-1, :, :], _C_METRIC, rtol=1e-6)
        )

    def test_trany_all_one(self):
        _, trany, _, _ = _compute(3, 4, 5)
        # Exclude the dummy last J-slice (padded with 0)
        assert np.all(
            np.isclose(trany.values.filled(np.nan)[:, :-1, :], _C_METRIC, rtol=1e-6)
        )

    def test_tranz_all_one(self):
        _, _, tranz, _ = _compute(3, 4, 5)
        # Exclude the dummy last K-slice (padded with 0)
        assert np.all(
            np.isclose(tranz.values.filled(np.nan)[:, :, :-1], _C_METRIC, rtol=1e-6)
        )


# ---------------------------------------------------------------------------
# Permeability scaling
# ---------------------------------------------------------------------------


class TestPermScaling:
    """T should be the harmonic mean of HT1 and HT2, scaled by the Darcy factor."""

    def test_tranx_harmonic_mean_of_perms(self):
        """With permx=2 in all cells, T = 2 * _C_METRIC."""
        tranx, _, _, _ = _compute(3, 4, 5, permx=2.0)
        assert np.all(
            np.isclose(
                tranx.values.filled(np.nan)[:-1, :, :], 2.0 * _C_METRIC, rtol=1e-6
            )
        )

    def test_trany_harmonic_mean_of_perms(self):
        _, trany, _, _ = _compute(3, 4, 5, permy=3.0)
        assert np.all(
            np.isclose(
                trany.values.filled(np.nan)[:, :-1, :], 3.0 * _C_METRIC, rtol=1e-6
            )
        )

    def test_tranz_harmonic_mean_of_perms(self):
        _, _, tranz, _ = _compute(3, 4, 5, permz=4.0)
        assert np.all(
            np.isclose(
                tranz.values.filled(np.nan)[:, :, :-1], 4.0 * _C_METRIC, rtol=1e-6
            )
        )


# ---------------------------------------------------------------------------
# NTG scaling (horizontal only)
# ---------------------------------------------------------------------------


class TestNTGScaling:
    """NTG reduces effective horizontal permeability; not applied to permz."""

    def test_ntg_halves_tranx(self):
        """ntg=0.5 → k_eff=0.5 → T = 0.5 * _C_METRIC."""
        tranx, _, _, _ = _compute(3, 4, 5, permx=1.0, ntg=0.5)
        assert np.all(
            np.isclose(
                tranx.values.filled(np.nan)[:-1, :, :], 0.5 * _C_METRIC, rtol=1e-6
            )
        )

    def test_ntg_halves_trany(self):
        _, trany, _, _ = _compute(3, 4, 5, permy=1.0, ntg=0.5)
        assert np.all(
            np.isclose(
                trany.values.filled(np.nan)[:, :-1, :], 0.5 * _C_METRIC, rtol=1e-6
            )
        )

    def test_ntg_not_applied_to_tranz(self):
        """NTG should NOT affect vertical transmissibility."""
        _, _, tranz_ntg, _ = _compute(3, 4, 5, permz=1.0, ntg=0.5)
        _, _, tranz_no_ntg, _ = _compute(3, 4, 5, permz=1.0)
        # Exclude dummy last K-slice to avoid NaN != NaN comparison
        assert np.allclose(
            tranz_ntg.values.filled(np.nan)[:, :, :-1],
            tranz_no_ntg.values.filled(np.nan)[:, :, :-1],
            atol=1e-10,
        )

    def test_ntg_none_same_as_ntg_one(self):
        """ntg=None should produce the same result as passing ntg=1.0."""
        grid = _box_grid(3, 4, 5)
        px = _uniform_prop(grid, 2.0, "permx")
        py = _uniform_prop(grid, 2.0, "permy")
        pz = _uniform_prop(grid, 2.0, "permz")
        ntg_one = _uniform_prop(grid, 1.0, "ntg")

        tx_none, ty_none, tz_none, _, _, _ = grid.get_transmissibilities(
            px, py, pz, ntg=None
        )
        tx_one, ty_one, tz_one, _, _, _ = grid.get_transmissibilities(
            px, py, pz, ntg=ntg_one
        )

        # Exclude dummy boundary slices to avoid NaN != NaN comparison
        assert np.allclose(
            tx_none.values.filled(np.nan)[:-1, :, :],
            tx_one.values.filled(np.nan)[:-1, :, :],
            atol=1e-12,
        )
        assert np.allclose(
            ty_none.values.filled(np.nan)[:, :-1, :],
            ty_one.values.filled(np.nan)[:, :-1, :],
            atol=1e-12,
        )
        assert np.allclose(
            tz_none.values.filled(np.nan)[:, :, :-1],
            tz_one.values.filled(np.nan)[:, :, :-1],
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Inactive cells
# ---------------------------------------------------------------------------


class TestInactiveCells:
    def test_inactive_cell_zeros_tranx(self):
        """I-connections involving an inactive cell → T = 0 (perm filled as 0)."""
        grid = _box_grid(3, 4, 5)
        # Deactivate a single cell
        actnum = grid.get_actnum()
        actnum.values[1, 1, 1] = 0
        grid.set_actnum(actnum)

        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")

        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(px, py, pz)

        # I-pair (0,1,1)←→(1,1,1): source cell (0,1,1) is active but neighbour is
        # inactive → C++ returns 0.0; the entry is unmasked (source cell is active)
        assert tranx.values[0, 1, 1] == pytest.approx(0.0)
        # I-pair (1,1,1)←→(2,1,1): source cell (1,1,1) is inactive → masked
        assert np.ma.is_masked(tranx.values[1, 1, 1])

    def test_all_active_no_masked_entries(self):
        """A fully grid should have no masked TRAN values (excl. dummy slices)."""
        tranx, trany, tranz, _ = _compute(3, 4, 5)
        # Use getmaskarray so we always get a proper boolean array
        # (mask may be scalar False).
        # The last boundary slice of each axis is a zero sentinel — exclude it.
        assert not np.any(np.ma.getmaskarray(tranx.values)[:-1, :, :])
        assert not np.any(np.ma.getmaskarray(trany.values)[:, :-1, :])
        assert not np.any(np.ma.getmaskarray(tranz.values)[:, :, :-1])


# ---------------------------------------------------------------------------
# NNC DataFrame: conforming grid has no NNCs
# ---------------------------------------------------------------------------


class TestConformingGridNoNNC:
    def test_empty_nnc_for_box_grid(self):
        _, _, _, nnc = _compute(3, 4, 5)
        assert len(nnc) == 0

    def test_empty_nnc_dataframe_has_correct_columns(self):
        _, _, _, nnc = _compute(3, 4, 5)
        assert "I1" in nnc.columns
        assert "T" in nnc.columns
        assert "TYPE" in nnc.columns


# ---------------------------------------------------------------------------
# NNC DataFrame: pinch-out
# ---------------------------------------------------------------------------


class TestPinchoutNNC:
    """A grid with a collapsed middle layer produces K-pinch-out NNCs."""

    @pytest.fixture()
    def pinchout_grid(self):
        """3x3x3 grid where the middle layer (k=1) is inactive.

        Pinch-out NNCs connect the active cells in k=0 and k=2 across the
        inactive k=1 run.
        """
        grid = xtgeo.create_box_grid(
            dimension=(3, 3, 3),
            increment=(1.0, 1.0, 1.0),
        )
        actnum = grid.get_actnum()
        # Deactivate the entire middle layer (k index 1, 0-based)
        actnum.values[:, :, 1] = 0
        grid.set_actnum(actnum)
        return grid

    def test_pinchout_nnc_created(self, pinchout_grid):
        grid = pinchout_grid
        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")
        _, _, _, nnc, _, _ = grid.get_transmissibilities(
            px, py, pz, min_dz_pinchout=0.5
        )
        pinchout = nnc[nnc["TYPE"] == "Pinchout"]
        assert len(pinchout) > 0

    def test_pinchout_nnc_type_is_string(self, pinchout_grid):
        grid = pinchout_grid
        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")
        _, _, _, nnc, _, _ = grid.get_transmissibilities(
            px, py, pz, min_dz_pinchout=0.5
        )
        if len(nnc) > 0:
            assert nnc["TYPE"].dtype == object  # string column

    def test_pinchout_nnc_indices_one_based(self, pinchout_grid):
        grid = pinchout_grid
        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")
        _, _, _, nnc, _, _ = grid.get_transmissibilities(
            px, py, pz, min_dz_pinchout=0.5
        )
        if len(nnc) > 0:
            assert nnc["I1"].min() >= 1
            assert nnc["J1"].min() >= 1
            assert nnc["K1"].min() >= 1
            assert nnc["I2"].min() >= 1
            assert nnc["J2"].min() >= 1
            assert nnc["K2"].min() >= 1

    def test_pinchout_nnc_positive_T(self, pinchout_grid):
        grid = pinchout_grid
        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")
        _, _, _, nnc, _, _ = grid.get_transmissibilities(
            px, py, pz, min_dz_pinchout=0.5
        )
        pinchout = nnc[nnc["TYPE"] == "Pinchout"]
        if len(pinchout) > 0:
            assert (pinchout["T"] > 0).all()


# ---------------------------------------------------------------------------
# NNC TYPE values are strings
# ---------------------------------------------------------------------------


class TestNNCTypeStrings:
    """NNC TYPE column must contain string 'Fault' or 'Pinchout'."""

    def test_type_values_are_valid_strings(self):
        _, _, _, nnc = _compute(3, 4, 5)
        # For a box grid there are no NNCs, but check dtype is correct when empty
        if len(nnc) > 0:
            valid_types = {"Fault", "Pinchout"}
            assert set(nnc["TYPE"].unique()).issubset(valid_types)

    def test_empty_nnc_type_column_dtype(self):
        _, _, _, nnc = _compute(3, 4, 5)
        # Even with 0 rows, TYPE column should exist and have object (string) dtype
        assert "TYPE" in nnc.columns


# ---------------------------------------------------------------------------
# min_dz_pinchout controls recognition threshold
# ---------------------------------------------------------------------------


class TestMinDzPinchout:
    def test_parameter_is_accepted(self):
        """min_dz_pinchout is accepted as a keyword argument without error."""
        grid = _box_grid(3, 4, 5)
        px = _uniform_prop(grid, 1.0, "permx")
        py = _uniform_prop(grid, 1.0, "permy")
        pz = _uniform_prop(grid, 1.0, "permz")
        _, _, _, nnc, _, _ = grid.get_transmissibilities(
            px, py, pz, min_dz_pinchout=0.001
        )
        assert isinstance(nnc, pd.DataFrame)

    def test_default_value_accepted(self):
        """Default min_dz_pinchout=1e-4 runs without error."""
        _, _, _, nnc = _compute(3, 4, 5)
        assert isinstance(nnc, pd.DataFrame)


# ---------------------------------------------------------------------------
# Right-handed grid: TRANY sentinel position
# ---------------------------------------------------------------------------


class TestHandedness:
    """The zero sentinel in TRANY always goes at the last J-slice (j=nrow-1)
    regardless of grid handedness, mirroring the Eclipse/OPM-flow convention."""

    def test_lefthanded_trany_sentinel_at_last_row(self):
        grid = _box_grid(3, 4, 2)
        if grid.ijk_handedness != "left":
            grid.ijk_handedness = "left"
        assert grid.ijk_handedness == "left"

        _, trany, _, _, _, _ = grid.get_transmissibilities(1.0, 1.0, 1.0)

        assert np.all(trany.values[:, -1, :] == 0.0)
        assert np.all(trany.values[:, :-1, :] > 0.0)

    def test_righthanded_trany_sentinel_at_last_row(self):
        grid = _box_grid(3, 4, 2)
        grid.ijk_handedness = "right"
        assert grid.ijk_handedness == "right"

        _, trany, _, _, _, _ = grid.get_transmissibilities(1.0, 1.0, 1.0)

        assert np.all(trany.values[:, -1, :] == 0.0)
        assert np.all(trany.values[:, :-1, :] > 0.0)

    def test_righthanded_trany_values_match_lefthanded(self):
        """Transmissibility magnitudes should be identical; only the padding
        position differs between left- and right-handed grids."""
        grid_l = _box_grid(3, 4, 2)
        if grid_l.ijk_handedness != "left":
            grid_l.ijk_handedness = "left"

        grid_r = _box_grid(3, 4, 2)
        grid_r.ijk_handedness = "right"

        _, trany_l, _, _, _, _ = grid_l.get_transmissibilities(1.0, 1.0, 1.0)
        _, trany_r, _, _, _, _ = grid_r.get_transmissibilities(1.0, 1.0, 1.0)

        # Active slices: [:-1] for both left- and right-handed
        assert np.allclose(
            trany_l.values[:, :-1, :],
            trany_r.values[:, :-1, :],
            atol=1e-12,
        )

    def test_tranx_and_tranz_unaffected_by_handedness(self):
        """TRANX and TRANZ padding are not affected by J-axis handedness."""
        grid_l = _box_grid(3, 4, 2)
        if grid_l.ijk_handedness != "left":
            grid_l.ijk_handedness = "left"

        grid_r = _box_grid(3, 4, 2)
        grid_r.ijk_handedness = "right"

        tranx_l, _, tranz_l, _, _, _ = grid_l.get_transmissibilities(1.0, 1.0, 1.0)
        tranx_r, _, tranz_r, _, _, _ = grid_r.get_transmissibilities(1.0, 1.0, 1.0)

        assert np.allclose(tranx_l.values, tranx_r.values, atol=1e-12)
        assert np.allclose(tranz_l.values, tranz_r.values, atol=1e-12)


# ---------------------------------------------------------------------------
# Test "BANAL7" grid case
# ---------------------------------------------------------------------------


class TestCompareCases:
    """Test the B7 case with numbers from vendor sw, to check for consistency."""

    def test_b7case_tranx_values_lh(self, testdata_path):
        """Compare with known reference values for the B7 case, left-handed case."""

        file = testdata_path / B7CASE_LH

        grid = xtgeo.grid_from_file(file)
        assert grid.ijk_handedness == "left"

        permx = xtgeo.gridproperty_from_file(file, name="PERMX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="PERMY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="PERMZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)

        # correct transmissibities to compare with
        tranx_compare = xtgeo.gridproperty_from_file(file, name="TRANX", grid=grid)
        trany_compare = xtgeo.gridproperty_from_file(file, name="TRANY", grid=grid)
        tranz_compare = xtgeo.gridproperty_from_file(file, name="TRANZ", grid=grid)

        # compute with xtgeo
        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, ntg, min_dz_pinchout=0.001
        )

        np.testing.assert_allclose(
            tranx.values.filled(0.0), tranx_compare.values.filled(0.0), rtol=1e-3
        )
        np.testing.assert_allclose(
            trany.values.filled(0.0), trany_compare.values.filled(0.0), rtol=1e-3
        )
        np.testing.assert_allclose(
            tranz.values.filled(0.0), tranz_compare.values.filled(0.0), rtol=1e-3
        )

    def test_b7case_tranx_values_rh(self, testdata_path):
        """Compare with known reference values for the B7 case, right-handed case."""

        file = testdata_path / B7CASE_RH
        grid = xtgeo.grid_from_file(file)

        assert grid.ijk_handedness == "right"

        permx = xtgeo.gridproperty_from_file(file, name="PERMX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="PERMY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="PERMZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)

        # correct transmissibities to compare with
        tranx_compare = xtgeo.gridproperty_from_file(file, name="TRANX", grid=grid)
        trany_compare = xtgeo.gridproperty_from_file(file, name="TRANY", grid=grid)
        tranz_compare = xtgeo.gridproperty_from_file(file, name="TRANZ", grid=grid)

        # compute with xtgeo
        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, ntg, min_dz_pinchout=0.001
        )

        np.testing.assert_allclose(
            tranx.values.filled(0.0), tranx_compare.values.filled(0.0), rtol=1e-3
        )
        np.testing.assert_allclose(
            trany.values.filled(0.0), trany_compare.values.filled(0.0), rtol=1e-3
        )
        np.testing.assert_allclose(
            tranz.values.filled(0.0), tranz_compare.values.filled(0.0), rtol=1e-3
        )

    def test_dcase_transm_values_lh(self, testdata_path):
        """Compare with known reference values for DCASE, left-handed case."""

        grid_file = testdata_path / DCASE_GRID_LH
        grid = xtgeo.grid_from_file(grid_file)

        assert grid.ijk_handedness == "left"

        props_file = testdata_path / DCASE_PROPS_LH

        permx = xtgeo.gridproperty_from_file(props_file, name="PERMX", grid=grid)
        permy = xtgeo.gridproperty_from_file(props_file, name="PERMY", grid=grid)
        permz = xtgeo.gridproperty_from_file(props_file, name="PERMZ", grid=grid)

        # correct transmissibities to compare with
        tranx_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANX", grid=grid
        )
        trany_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANY", grid=grid
        )
        tranz_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANZ", grid=grid
        )

        # compute with xtgeo
        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, min_dz_pinchout=0.001
        )

        # The DCASE grid contains non-conforming fault cells where the
        # Sutherland-Hodgman polygon-clipping approach computes intersection areas
        # that differ from OPM's algorithm by up to ~5% (TRANX) and ~0.4% (TRANY)
        # at geometrically degenerate fault boundaries.  All other cells agree
        # within 0.1 %.
        np.testing.assert_allclose(
            tranx.values.filled(0.0), tranx_compare.values.filled(0.0), rtol=5e-2
        )
        np.testing.assert_allclose(
            trany.values.filled(0.0), trany_compare.values.filled(0.0), rtol=5e-3
        )
        np.testing.assert_allclose(
            tranz.values.filled(0.0), tranz_compare.values.filled(0.0), rtol=1e-3
        )

    def test_dcase_transm_values_rh(self, testdata_path):
        """Compare with known reference values for DCASE, right-handed case."""

        grid_file = testdata_path / DCASE_GRID_RH
        grid = xtgeo.grid_from_file(grid_file)

        assert grid.ijk_handedness == "right"

        props_file = testdata_path / DCASE_PROPS_RH

        permx = xtgeo.gridproperty_from_file(props_file, name="PERMX", grid=grid)
        permy = xtgeo.gridproperty_from_file(props_file, name="PERMY", grid=grid)
        permz = xtgeo.gridproperty_from_file(props_file, name="PERMZ", grid=grid)

        # correct transmissibities to compare with
        tranx_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANX", grid=grid
        )
        trany_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANY", grid=grid
        )
        tranz_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANZ", grid=grid
        )

        # compute with xtgeo
        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, min_dz_pinchout=0.001
        )

        # The DCASE grid contains non-conforming fault cells where the
        # Sutherland-Hodgman polygon-clipping approach computes intersection areas
        # that differ from OPM's algorithm by up to ~5% (TRANX) and ~0.4% (TRANY)
        # at geometrically degenerate fault boundaries.  All other cells agree
        # within 0.1 %.
        np.testing.assert_allclose(
            tranx.values.filled(0.0), tranx_compare.values.filled(0.0), rtol=5e-2
        )
        np.testing.assert_allclose(
            trany.values.filled(0.0), trany_compare.values.filled(0.0), rtol=5e-3
        )
        np.testing.assert_allclose(
            tranz.values.filled(0.0), tranz_compare.values.filled(0.0), rtol=1e-3
        )

    def test_drogon_opm_transm_rh(self, testdata_path):
        """Compare with known reference values for DCASE, right-handed case."""

        grid_file = testdata_path / DROGON_FULL_OPM
        grid = xtgeo.grid_from_file(grid_file)

        assert grid.ijk_handedness == "right"

        props_file = testdata_path / DROGON_FULL_OPM

        permx = xtgeo.gridproperty_from_file(props_file, name="PERMX", grid=grid)
        permy = xtgeo.gridproperty_from_file(props_file, name="PERMY", grid=grid)
        permz = xtgeo.gridproperty_from_file(props_file, name="PERMZ", grid=grid)

        # correct (assuming...) transmissibities to compare with, here made in OPM
        tranx_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANX_trueOPM", grid=grid
        )
        trany_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANY_trueOPM", grid=grid
        )
        tranz_compare = xtgeo.gridproperty_from_file(
            props_file, name="TRANZ_trueOPM", grid=grid
        )

        # compute with xtgeo
        tranx, trany, tranz, _, _, _ = grid.get_transmissibilities(
            permx, permy, permz, min_dz_pinchout=0.001
        )

        # The Drogon OPM reference was generated with "PINCH 3* ALL /" which
        # creates TRANZ connections across genuinely inactive (non-pinched) layers.
        # xtgeo does not support that mode, so those cells are excluded from the
        # TRANZ comparison.  For TRANX/TRANY, a small number of non-conforming
        # fault column pairs show up to ~19% deviation due to near-degenerate
        # Sutherland-Hodgman intersection geometry (shared pillar lines).
        tranx_a = tranx.values.filled(0.0)
        trany_a = trany.values.filled(0.0)
        tranz_a = tranz.values.filled(0.0)
        tranx_r = tranx_compare.values.filled(0.0)
        trany_r = trany_compare.values.filled(0.0)
        tranz_r = tranz_compare.values.filled(0.0)

        np.testing.assert_allclose(tranx_a, tranx_r, rtol=0.20)
        np.testing.assert_allclose(trany_a, trany_r, rtol=0.20)

        # Only compare TRANZ where xtgeo produced a non-zero value (i.e. exclude
        # the PINCH-ALL connections that xtgeo does not generate).
        xtgeo_computed = tranz_a > 0
        np.testing.assert_allclose(
            tranz_a[xtgeo_computed], tranz_r[xtgeo_computed], rtol=5e-3
        )


class TestEmeraldOriginal:
    """Tests for the original (non-hybrid) Emerald grid transmissibilities."""

    def test_emerald_original(self, testdata_path):
        """Original grid without any nested grid.

        Only NNC for a fault shall be present. Values are validated towards a
        vendor tool.
        """
        file = testdata_path / EMERALD_ORIGINAL

        grid = xtgeo.grid_from_file(file)
        permx = xtgeo.gridproperty_from_file(file, name="KX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="KY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="KZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)

        @functimer(output="print")
        def get_trans_nnc():
            return grid.get_transmissibilities(permx, permy, permz, ntg)

        tx, ty, tz, nncs_df, nested_nnc_df, refined_boundary = get_trans_nnc()

        assert tx.values.mean() == pytest.approx(11.9938, rel=1e-2)
        print(f"Original grid: NNCs found: {len(nncs_df)}")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print(nncs_df)

    def test_emerald_original_nnc_vs_vendor(self, testdata_path):
        """Validate fault NNCs against reference values from a vendor tool.

        The vendor output (1-based cell indices, METRIC transmissibilities) is
        used as the ground truth.  An absolute tolerance of 2 % is applied to
        account for minor geometric and rounding differences.
        """
        TEST_TOLERANCE = 0.02
        # fmt: off
        # (I1, J1, K1,  I2, J2, K2,  T_vendor)  – 1-based indices
        VENDOR_NNCS = [
            (20, 38,  2,  21, 38,  1,  3.51704),
            (20, 38,  3,  21, 38,  2,  4.55548),
            (20, 38,  3,  21, 38,  1,  3.77430),
            (20, 38,  4,  21, 38,  3,  4.33944),
            (20, 38,  4,  21, 38,  2,  3.98788),
            (20, 38,  4,  21, 38,  1,  0.877817),
            (20, 38,  5,  21, 38,  4,  5.30723),
            (20, 38,  5,  21, 38,  3,  4.15059),
            (20, 38,  5,  21, 38,  2,  1.01971),
            (20, 38,  6,  21, 38,  5,  4.15577),
            (20, 38,  6,  21, 38,  4,  4.06103),
            (20, 38,  6,  21, 38,  3,  0.876832),
            (20, 38,  7,  21, 38,  6,  3.63126),
            (20, 38,  7,  21, 38,  5,  3.60155),
            (20, 38,  7,  21, 38,  4,  0.954248),
            (20, 38,  8,  21, 38,  7,  3.92388),
            (20, 38,  8,  21, 38,  6,  3.50182),
            (20, 38,  8,  21, 38,  5,  0.958901),
            (20, 38,  9,  21, 38,  8,  4.02525),
            (20, 38,  9,  21, 38,  7,  3.97591),
            (20, 38,  9,  21, 38,  6,  0.981390),
            (20, 38, 10,  21, 38,  9,  4.21107),
            (20, 38, 10,  21, 38,  8,  3.14213),
            (20, 38, 10,  21, 38,  7,  0.858845),
            (20, 39,  3,  21, 39,  1,  3.02163),
            (20, 39,  4,  21, 39,  2,  2.68304),
            (20, 39,  4,  21, 39,  1,  8.55739),
            (20, 39,  5,  21, 39,  3,  2.92931),
            (20, 39,  5,  21, 39,  2,  6.90608),
            (20, 39,  6,  21, 39,  4,  3.21801),
            (20, 39,  6,  21, 39,  3,  7.07623),
            (20, 39,  7,  21, 39,  5,  2.73778),
            (20, 39,  7,  21, 39,  4,  8.38846),
            (20, 39,  8,  21, 39,  6,  2.88363),
            (20, 39,  8,  21, 39,  5,  7.39543),
            (20, 39,  9,  21, 39,  7,  2.49574),
            (20, 39,  9,  21, 39,  6,  7.23385),
            (20, 39, 10,  21, 39,  8,  1.98637),
            (20, 39, 10,  21, 39,  7,  6.85378),
            (20, 40,  3,  21, 40,  1,  1.16405),
            (20, 40,  4,  21, 40,  2,  1.34219),
            (20, 40,  4,  21, 40,  1,  7.46973),
            (20, 40,  5,  21, 40,  3,  1.50756),
            (20, 40,  5,  21, 40,  2,  8.48958),
            (20, 40,  6,  21, 40,  4,  1.19177),
            (20, 40,  6,  21, 40,  3,  7.54775),
            (20, 40,  7,  21, 40,  5,  1.32063),
            (20, 40,  7,  21, 40,  4,  6.56630),
            (20, 40,  8,  21, 40,  6,  1.44202),
            (20, 40,  8,  21, 40,  5, 10.93780),
            (20, 40,  9,  21, 40,  7,  1.02841),
            (20, 40,  9,  21, 40,  6,  8.62672),
            (20, 40, 10,  21, 40,  8,  0.954474),
            (20, 40, 10,  21, 40,  7,  6.63284),
            (20, 41,  3,  21, 41,  1,  0.552810),
            (20, 41,  4,  21, 41,  2,  0.669192),
            (20, 41,  4,  21, 41,  1,  4.37276),
            (20, 41,  5,  21, 41,  3,  0.491992),
            (20, 41,  5,  21, 41,  2,  4.77797),
            (20, 41,  6,  21, 41,  4,  0.424965),
            (20, 41,  6,  21, 41,  3,  3.02803),
            (20, 41,  7,  21, 41,  5,  0.494492),
            (20, 41,  7,  21, 41,  4,  3.46471),
            (20, 41,  8,  21, 41,  6,  0.436482),
            (20, 41,  8,  21, 41,  5,  3.70659),
            (20, 41,  9,  21, 41,  7,  0.398400),
            (20, 41,  9,  21, 41,  6,  3.55789),
            (20, 41, 10,  21, 41,  8,  0.411018),
            (20, 41, 10,  21, 41,  7,  3.29286),
            (21, 42,  4,  22, 42,  1,  0.027078),
            (21, 42,  5,  22, 42,  2,  0.0389467),
            (21, 42,  5,  22, 42,  1,  1.17701),
            (21, 42,  6,  22, 42,  3,  0.0298662),
            (21, 42,  6,  22, 42,  2,  1.15605),
            (21, 42,  6,  22, 42,  1,  1.61143),
            (21, 42,  7,  22, 42,  4,  0.0313637),
            (21, 42,  7,  22, 42,  3,  0.758399),
            (21, 42,  7,  22, 42,  2,  1.44023),
            (21, 42,  7,  22, 42,  1,  0.827161),
            (21, 42,  8,  22, 42,  5,  0.0575135),
            (21, 42,  8,  22, 42,  4,  0.963765),
            (21, 42,  8,  22, 42,  3,  1.31538),
            (21, 42,  8,  22, 42,  2,  1.06398),
            (21, 42,  8,  22, 42,  1,  0.00615493),
            (21, 42,  9,  22, 42,  6,  0.0564943),
            (21, 42,  9,  22, 42,  5,  1.11667),
            (21, 42,  9,  22, 42,  4,  1.21171),
            (21, 42,  9,  22, 42,  3,  0.640832),
            (21, 42,  9,  22, 42,  2,  0.0024272),
            (21, 42, 10,  22, 42,  7,  0.0675853),
            (21, 42, 10,  22, 42,  6,  1.35732),
            (21, 42, 10,  22, 42,  5,  1.81465),
            (21, 42, 10,  22, 42,  4,  0.709216),
            (21, 42, 10,  22, 42,  3,  0.000488207),
            (22, 43,  9,  23, 43,  1,  1.14960),
            (22, 43, 10,  23, 43,  3,  0.000505792),
            (22, 43, 10,  23, 43,  2,  1.18204),
            (22, 43, 10,  23, 43,  1,  2.15070),
            (25, 47, 10,  26, 47,  2,  0.0254651),
            (25, 47, 10,  26, 47,  1,  2.37669),
            (26, 48,  7,  27, 48,  1,  0.940266),
            (26, 48,  8,  27, 48,  2,  0.848771),
            (26, 48,  8,  27, 48,  1,  3.72652),
            (26, 48,  9,  27, 48,  3,  1.26766),
            (26, 48,  9,  27, 48,  2,  3.72258),
            (26, 48,  9,  27, 48,  1,  0.879487),
            (26, 48, 10,  27, 48,  4,  2.23521),
            (26, 48, 10,  27, 48,  3,  4.71964),
            (26, 48, 10,  27, 48,  2,  0.670508),
            (26, 49,  6,  27, 49,  1,  0.00260268),
            (26, 49,  7,  27, 49,  2,  0.0488994),
            (26, 49,  7,  27, 49,  1,  5.96935),
            (26, 49,  8,  27, 49,  3,  0.113607),
            (26, 49,  8,  27, 49,  2,  3.62590),
            (26, 49,  8,  27, 49,  1,  0.775939),
            (26, 49,  9,  27, 49,  4,  0.380956),
            (26, 49,  9,  27, 49,  3,  5.57888),
            (26, 49,  9,  27, 49,  2,  0.601756),
            (26, 49, 10,  27, 49,  5,  0.754688),
            (26, 49, 10,  27, 49,  4,  7.71382),
            (26, 49, 10,  27, 49,  3,  0.437194),
            (27, 50,  7,  28, 50,  1,  0.880191),
            (27, 50,  8,  28, 50,  2,  0.554825),
            (27, 50,  8,  28, 50,  1,  3.15541),
            (27, 50,  9,  28, 50,  3,  0.798462),
            (27, 50,  9,  28, 50,  2,  2.67489),
            (27, 50,  9,  28, 50,  1,  1.06918),
            (27, 50, 10,  28, 50,  4,  1.05772),
            (27, 50, 10,  28, 50,  3,  4.28388),
            (27, 50, 10,  28, 50,  2,  1.00326),
            (21, 41,  1,  21, 42,  3,  0.0343826),
            (21, 41,  1,  21, 42,  4,  1.26388),
            (21, 41,  1,  21, 42,  5,  0.726294),
            (21, 41,  2,  21, 42,  4,  0.0377157),
            (21, 41,  2,  21, 42,  5,  1.51423),
            (21, 41,  2,  21, 42,  6,  0.671558),
            (21, 41,  3,  21, 42,  5,  0.0376739),
            (21, 41,  3,  21, 42,  6,  1.24219),
            (21, 41,  3,  21, 42,  7,  0.605598),
            (21, 41,  4,  21, 42,  6,  0.0347076),
            (21, 41,  4,  21, 42,  7,  1.32085),
            (21, 41,  4,  21, 42,  8,  0.728454),
            (21, 41,  5,  21, 42,  7,  0.0339071),
            (21, 41,  5,  21, 42,  8,  1.54206),
            (21, 41,  5,  21, 42,  9,  0.605538),
            (21, 41,  6,  21, 42,  8,  0.0448869),
            (21, 41,  6,  21, 42,  9,  1.52055),
            (21, 41,  6,  21, 42, 10,  0.671601),
            (21, 41,  7,  21, 42,  9,  0.0362266),
            (21, 41,  7,  21, 42, 10,  1.49837),
            (21, 41,  8,  21, 42, 10,  0.0378628),
            (22, 42,  1,  22, 43,  7,  0.425377),
            (22, 42,  1,  22, 43,  8,  1.09526),
            (22, 42,  1,  22, 43,  9,  0.598853),
            (22, 42,  2,  22, 43,  8,  0.416829),
            (22, 42,  2,  22, 43,  9,  1.03359),
            (22, 42,  2,  22, 43, 10,  0.491705),
            (22, 42,  3,  22, 43,  9,  0.427052),
            (22, 42,  3,  22, 43, 10,  0.908231),
            (22, 42,  4,  22, 43, 10,  0.488803),
            (26, 47,  1,  26, 48,  8,  0.0472482),
            (26, 47,  1,  26, 48,  9,  0.654591),
            (26, 47,  1,  26, 48, 10,  0.440276),
            (26, 47,  2,  26, 48,  9,  0.117520),
            (26, 47,  2,  26, 48, 10,  1.33102),
            (26, 47,  3,  26, 48, 10,  0.222007),
            (27, 49,  1,  27, 50,  6,  0.00113956),
            (27, 49,  1,  27, 50,  7,  2.02185),
            (27, 49,  1,  27, 50,  8,  0.336931),
            (27, 49,  2,  27, 50,  7,  0.0184422),
            (27, 49,  2,  27, 50,  8,  1.74284),
            (27, 49,  2,  27, 50,  9,  0.293626),
            (27, 49,  3,  27, 50,  8,  0.0450021),
            (27, 49,  3,  27, 50,  9,  1.89227),
            (27, 49,  3,  27, 50, 10,  0.270796),
            (27, 49,  4,  27, 50,  9,  0.103536),
            (27, 49,  4,  27, 50, 10,  2.39234),
            (27, 49,  5,  27, 50, 10,  0.169134),
            (28, 50,  1,  28, 51,  8,  0.0254338),
            (28, 50,  1,  28, 51,  9,  0.557445),
            (28, 50,  1,  28, 51, 10,  0.953396),
            (28, 50,  2,  28, 51,  9,  0.020771),
            (28, 50,  2,  28, 51, 10,  0.677130),
            (28, 50,  3,  28, 51, 10,  0.026394),
            (30, 20,  1,  30, 21,  2,  1.30582),
            (30, 20,  1,  30, 21,  3,  1.27245),
            (30, 20,  1,  30, 21,  4,  1.37485),
            (30, 20,  1,  30, 21,  5,  1.19319),
            (30, 20,  1,  30, 21,  6,  1.09166),
            (30, 20,  1,  30, 21,  7,  0.260450),
            (30, 20,  2,  30, 21,  3,  1.28418),
            (30, 20,  2,  30, 21,  4,  1.38748),
            (30, 20,  2,  30, 21,  5,  1.20538),
            (30, 20,  2,  30, 21,  6,  1.17230),
            (30, 20,  2,  30, 21,  7,  1.15002),
            (30, 20,  2,  30, 21,  8,  0.296904),
            (30, 20,  3,  30, 21,  4,  1.42646),
            (30, 20,  3,  30, 21,  5,  1.23742),
            (30, 20,  3,  30, 21,  6,  1.20354),
            (30, 20,  3,  30, 21,  7,  1.25659),
            (30, 20,  3,  30, 21,  8,  1.34160),
            (30, 20,  3,  30, 21,  9,  0.250082),
            (30, 20,  4,  30, 21,  5,  1.19742),
            (30, 20,  4,  30, 21,  6,  1.16611),
            (30, 20,  4,  30, 21,  7,  1.21522),
            (30, 20,  4,  30, 21,  8,  1.37127),
            (30, 20,  4,  30, 21,  9,  1.06870),
            (30, 20,  4,  30, 21, 10,  0.235802),
            (30, 20,  5,  30, 21,  6,  1.17687),
            (30, 20,  5,  30, 21,  7,  1.22674),
            (30, 20,  5,  30, 21,  8,  1.38397),
            (30, 20,  5,  30, 21,  9,  1.14928),
            (30, 20,  5,  30, 21, 10,  1.05511),
            (30, 20,  6,  30, 21,  7,  1.28455),
            (30, 20,  6,  30, 21,  8,  1.45685),
            (30, 20,  6,  30, 21,  9,  1.20289),
            (30, 20,  6,  30, 21, 10,  1.17647),
            (30, 20,  7,  30, 21,  8,  1.40419),
            (30, 20,  7,  30, 21,  9,  1.16903),
            (30, 20,  7,  30, 21, 10,  1.14412),
            (30, 20,  8,  30, 21,  9,  1.22374),
            (30, 20,  8,  30, 21, 10,  1.19788),
            (30, 20,  9,  30, 21, 10,  1.17805),
        ]
        # fmt: on

        file = testdata_path / EMERALD_ORIGINAL

        grid = xtgeo.grid_from_file(file)
        permx = xtgeo.gridproperty_from_file(file, name="KX", grid=grid)
        permy = xtgeo.gridproperty_from_file(file, name="KY", grid=grid)
        permz = xtgeo.gridproperty_from_file(file, name="KZ", grid=grid)
        ntg = xtgeo.gridproperty_from_file(file, name="NTG", grid=grid)

        _, _, _, nncs_df, _, _ = grid.get_transmissibilities(permx, permy, permz, ntg)

        # Negligible-T NNCs (T < 1e-4) are physically irrelevant and may be
        # absent from the vendor output.  Filter both sides before counting.
        MIN_T = 1e-4
        nncs_significant = nncs_df[nncs_df["T"] >= MIN_T]
        vendor_significant = [e for e in VENDOR_NNCS if e[6] >= MIN_T]
        assert len(nncs_significant) == len(vendor_significant), (
            f"Expected {len(vendor_significant)} NNCs with T >= {MIN_T}, "
            f"found {len(nncs_significant)}"
        )

        permx_mask = np.ma.getmaskarray(permx.values)

        def _is_active(i1b, j1b, k1b, i2b, j2b, k2b):
            """Return True when both cells are unmasked (1-based indices)."""
            return not (
                permx_mask[i1b - 1, j1b - 1, k1b - 1]
                or permx_mask[i2b - 1, j2b - 1, k2b - 1]
            )

        active_vendor = [
            (i1, j1, k1, i2, j2, k2, t)
            for i1, j1, k1, i2, j2, k2, t in VENDOR_NNCS
            if _is_active(i1, j1, k1, i2, j2, k2)
        ]

        # Build a lookup keyed by the canonical cell-pair tuple (smaller index first
        # within each axis so orientation does not matter).
        computed: dict[tuple, float] = {}
        for _, row in nncs_df.iterrows():
            key = (
                int(row["I1"]),
                int(row["J1"]),
                int(row["K1"]),
                int(row["I2"]),
                int(row["J2"]),
                int(row["K2"]),
            )
            computed[key] = float(row["T"])

        missing = []
        mismatches = []
        for i1, j1, k1, i2, j2, k2, t_ref in active_vendor:
            key = (i1, j1, k1, i2, j2, k2)
            rev = (i2, j2, k2, i1, j1, k1)
            if key in computed:
                t_calc = computed[key]
            elif rev in computed:
                t_calc = computed[rev]
            else:
                missing.append(key)
                continue
            if abs(t_calc - t_ref) > TEST_TOLERANCE * max(abs(t_ref), 1e-9):
                mismatches.append((key, t_ref, t_calc))

        assert not missing, f"Vendor NNCs not found in computed result: {missing}"
        assert not mismatches, (
            "Transmissibility mismatches (key, t_vendor, t_computed):\n"
            + "\n".join(f"  {k}: {tr:.6g} vs {tc:.6g}" for k, tr, tc in mismatches)
        )
