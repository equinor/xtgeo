"""Tests for compute_transmissibilities.

Covers:
- Output array shapes for a box grid.
- Unit transmissibility values (perm=1, ntg=1 unit-cube grid → T=1 everywhere).
- Permeability scaling: harmonic mean of two different perms.
- NTG scaling: effective perm = perm * ntg.
- Inactive-cell pairs produce NaN in the TRAN arrays.
- No fault NNCs for a conforming box grid.
- Pinch-out NNCs across a single inactive cell layer.
- Fault NNCs for a column with a vertical Z-offset fault.
  (Only in I-direction; J-direction follows by symmetry.)
"""

import math

import numpy as np
import pytest
from xtgeo._internal.grid3d import (  # type: ignore
    NNCType,
    compute_transmissibilities,
)

import xtgeo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_grid_cpp(nc: int, nr: int, nl: int):
    """Return (xtgeo.Grid, C++ Grid) for a unit-increment box grid."""
    grid = xtgeo.create_box_grid((nc, nr, nl))
    return grid, grid._get_grid_cpp()


def _uniform_perm(nc, nr, nl, value=1.0):
    """Return a float64 numpy array of shape (nc, nr, nl) filled with value."""
    return np.full((nc, nr, nl), value, dtype=np.float64)


def _compute(nc, nr, nl, permx=1.0, permy=1.0, permz=1.0, ntg=1.0):
    """Convenience: create a uniform box grid and compute transmissibilities."""
    _, gcpp = _box_grid_cpp(nc, nr, nl)
    px = _uniform_perm(nc, nr, nl, permx)
    py_ = _uniform_perm(nc, nr, nl, permy)
    pz = _uniform_perm(nc, nr, nl, permz)
    nt = _uniform_perm(nc, nr, nl, ntg)
    return compute_transmissibilities(gcpp, px, py_, pz, nt)


# ---------------------------------------------------------------------------
# Shape assertions
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify TRAN array shapes for various grid sizes."""

    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_tranx_shape(self, nc, nr, nl):
        r = _compute(nc, nr, nl)
        assert r.tranx.shape == (max(nc - 1, 0), nr, nl)

    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_trany_shape(self, nc, nr, nl):
        r = _compute(nc, nr, nl)
        assert r.trany.shape == (nc, max(nr - 1, 0), nl)

    @pytest.mark.parametrize("nc,nr,nl", [(3, 4, 5), (1, 1, 1), (2, 2, 2)])
    def test_tranz_shape(self, nc, nr, nl):
        r = _compute(nc, nr, nl)
        assert r.tranz.shape == (nc, nr, max(nl - 1, 0))


# ---------------------------------------------------------------------------
# Unit transmissibility values
# ---------------------------------------------------------------------------


class TestUnitTran:
    """For a unit cube box grid with perm=ntg=1 the TPFA T should be 1.0
    everywhere.

    HT = k * A / d = 1.0 * 1.0 / 0.5 = 2.0
    T  = HT^2 / (2 * HT)  = 1.0
    """

    def test_tranx_all_one(self):
        r = _compute(3, 4, 5)
        assert np.all(r.tranx == pytest.approx(1.0))

    def test_trany_all_one(self):
        r = _compute(3, 4, 5)
        assert np.all(r.trany == pytest.approx(1.0))

    def test_tranz_all_one(self):
        r = _compute(3, 4, 5)
        assert np.all(r.tranz == pytest.approx(1.0))


# ---------------------------------------------------------------------------
# Permeability scaling
# ---------------------------------------------------------------------------


class TestPermScaling:
    """Verify the TPFA formula with non-unit permeabilities."""

    def test_harmonic_mean_tranx(self):
        """k1=2, k2=0.5, ntg=1, area=1, d1=d2=0.5 → T = 4/5."""
        nc, nr, nl = 2, 1, 1
        _, gcpp = _box_grid_cpp(nc, nr, nl)
        # Grid column 0 has permx=2, column 1 has permx=0.5
        px = np.array([[[2.0]], [[0.5]]], dtype=np.float64)  # (nc,nr,nl)
        py_ = _uniform_perm(nc, nr, nl)
        pz = _uniform_perm(nc, nr, nl)
        nt = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, py_, pz, nt)
        # HT1 = 2.0*1.0/0.5 = 4.0, HT2 = 0.5*1.0/0.5 = 1.0 → T = 4/5
        assert r.tranx[0, 0, 0] == pytest.approx(4.0 * 1.0 / (4.0 + 1.0))

    def test_ntg_scales_horizontal_perm(self):
        """perm=2, ntg=0.5 → k_eff=1 → same T as perm=ntg=1."""
        nc, nr, nl = 3, 4, 5
        r_eff = _compute(nc, nr, nl, permx=2.0, ntg=0.5)
        assert np.all(r_eff.tranx == pytest.approx(1.0))

    def test_ntg_does_not_scale_vertical_perm(self):
        """permz=1, ntg=0.5 → tranz unchanged by ntg (K-direction ignores ntg)."""
        nc, nr, nl = 3, 3, 4
        r_ntg = _compute(nc, nr, nl, permz=1.0, ntg=0.5)
        r_ref = _compute(nc, nr, nl, permz=1.0, ntg=1.0)
        np.testing.assert_array_almost_equal(r_ntg.tranz, r_ref.tranz)


# ---------------------------------------------------------------------------
# Inactive cells
# ---------------------------------------------------------------------------


class TestInactiveCells:
    """Pairs involving at least one inactive cell produce NaN in TRAN arrays."""

    def _grid_with_inactive(self, nc, nr, nl, ii, ij, ik):
        """Box grid with one cell deactivated at (ii, ij, ik)."""
        grid = xtgeo.create_box_grid((nc, nr, nl))
        grid._actnumsv[ii, ij, ik] = 0
        return grid._get_grid_cpp()

    def test_inactive_cell_tranx_nan(self):
        """tranx[i, j, k] should be NaN if cell (i,j,k) or (i+1,j,k) is inactive."""
        nc, nr, nl = 3, 2, 2
        gcpp = self._grid_with_inactive(nc, nr, nl, 1, 0, 0)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # Both TRANX[0,0,0] (pair (0,0,0)→(1,0,0)) and TRANX[1,0,0] should be NaN
        assert math.isnan(r.tranx[0, 0, 0])
        assert math.isnan(r.tranx[1, 0, 0])
        # Pair (0,0,1) → (1,0,1) is unaffected
        assert not math.isnan(r.tranx[0, 0, 1])

    def test_inactive_cell_tranz_nan(self):
        """tranz[i, j, k] is NaN if cell (i,j,k) or (i,j,k+1) is inactive."""
        nc, nr, nl = 2, 2, 3
        gcpp = self._grid_with_inactive(nc, nr, nl, 0, 0, 1)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        assert math.isnan(r.tranz[0, 0, 0])  # pair (0,0,0)→(0,0,1)
        assert math.isnan(r.tranz[0, 0, 1])  # pair (0,0,1)→(0,0,2)
        assert not math.isnan(r.tranz[1, 0, 0])  # different column unaffected


# ---------------------------------------------------------------------------
# NNC — no NNCs for conforming box grid
# ---------------------------------------------------------------------------


class TestNoNNCForConformingGrid:
    """A plain box grid has no fault or pinch-out NNCs."""

    def test_no_nncs(self):
        r = _compute(5, 5, 5)
        assert len(r.nnc_T) == 0

    def test_nnc_type_correct_when_present(self):
        """The NNCType enum values are accessible from Python."""
        assert int(NNCType.Fault) == 0
        assert int(NNCType.Pinchout) == 1


# ---------------------------------------------------------------------------
# Pinch-out NNCs
# ---------------------------------------------------------------------------


class TestPinchoutNNC:
    """Inactive layer between two active layers → Pinchout NNC."""

    def _grid_with_inactive_layer(self, nc, nr, inactive_k):
        """Box grid nl=3 with all cells at k=inactive_k deactivated."""
        nl = 3
        grid = xtgeo.create_box_grid((nc, nr, nl))
        grid._actnumsv[:, :, inactive_k] = 0
        return grid._get_grid_cpp(), nc, nr, nl

    def test_pinchout_nncs_created(self):
        """Every column should produce one Pinchout NNC across the inactive layer."""
        nc, nr = 2, 2
        gcpp, nc_, nr_, nl = self._grid_with_inactive_layer(nc, nr, inactive_k=1)
        px = _uniform_perm(nc_, nr_, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # One NNC per column (nc*nr = 4)
        assert len(r.nnc_T) == nc * nr

    def test_pinchout_type_flag(self):
        nc, nr = 2, 2
        gcpp, nc_, nr_, nl = self._grid_with_inactive_layer(nc, nr, inactive_k=1)
        px = _uniform_perm(nc_, nr_, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        assert np.all(r.nnc_type == int(NNCType.Pinchout))

    def test_pinchout_cells_connect_k0_to_k2(self):
        """The NNC should connect k=0 (above inactive) to k=2 (below inactive)."""
        nc, nr = 1, 1
        gcpp, nc_, nr_, nl = self._grid_with_inactive_layer(nc, nr, inactive_k=1)
        px = _uniform_perm(nc_, nr_, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        assert len(r.nnc_T) == 1
        assert r.nnc_k1[0] == 0
        assert r.nnc_k2[0] == 2

    def test_pinchout_tran_value(self):
        """Unit cube grid, perm=1, ntg=1: pinchout T same as standard K-T (1.0)."""
        nc, nr = 1, 1
        gcpp, nc_, nr_, nl = self._grid_with_inactive_layer(nc, nr, inactive_k=1)
        px = _uniform_perm(nc_, nr_, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # For a unit cube the faces coincide → area=1, d1=d2=0.5 → T=1.0
        assert r.nnc_T[0] == pytest.approx(1.0)

    def test_tranz_nan_adjacent_to_inactive(self):
        """Standard K connections involving the inactive layer are NaN."""
        nc, nr = 1, 1
        gcpp, nc_, nr_, nl = self._grid_with_inactive_layer(nc, nr, inactive_k=1)
        px = _uniform_perm(nc_, nr_, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # tranz[0, 0, 0] is (k=0)→(k=1, inactive) → NaN
        # tranz[0, 0, 1] is (k=1, inactive)→(k=2) → NaN
        assert math.isnan(r.tranz[0, 0, 0])
        assert math.isnan(r.tranz[0, 0, 1])


# ---------------------------------------------------------------------------
# Fault NNCs (I-direction only; J is symmetric)
# ---------------------------------------------------------------------------


def _make_faulted_grid_cpp(nc, nr, nl, fault_column_i, z_offset):
    """Box grid where column i=fault_column_i is shifted by z_offset in Z.

    In xtgeo's corner-point format, zcornsv[ic, jc, kc, sub] stores the Z
    value at pillar (ic, jc) at layer boundary kc for each of the up-to-4
    adjacent cells (sub-index 0-3).

    For cell (i, j, k):
      upper_sw Z = zcornsv[i,     j,   k,   3]
      upper_se Z = zcornsv[i+1,   j,   k,   2]
      upper_nw Z = zcornsv[i,     j+1, k,   1]
      upper_ne Z = zcornsv[i+1,   j+1, k,   0]
      lower_* uses kc = k+1 with the same sub-indices

    Faulting column i=fault_column_i by adding z_offset to all 4 corners of
    each cell in that column.
    """
    grid = xtgeo.create_box_grid((nc, nr, nl))
    z = grid._zcornsv.copy()
    ic = fault_column_i
    # Shift SW corners: zcornsv[ic, :, :, 3]
    z[ic, :, :, 3] += z_offset
    # Shift NW corners: zcornsv[ic, :, :, 1]
    z[ic, :, :, 1] += z_offset
    # Shift SE corners: zcornsv[ic+1, :, :, 2]
    z[ic + 1, :, :, 2] += z_offset
    # Shift NE corners: zcornsv[ic+1, :, :, 0]
    z[ic + 1, :, :, 0] += z_offset
    grid._zcornsv = z
    return grid._get_grid_cpp()


class TestFaultNNC:
    """Shifted column produces fault NNCs with correct types and T values."""

    def test_fault_nncs_created(self):
        """A 0.7-unit Z offset should create cross-K fault NNCs in I-direction."""
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # There must be at least one Fault NNC
        fault_mask = r.nnc_type == int(NNCType.Fault)
        assert np.any(fault_mask), "Expected at least one Fault NNC"

    def test_fault_nnc_type(self):
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # All NNCs should be Fault type (no inactive layers → no Pinchout)
        assert np.all(r.nnc_type == int(NNCType.Fault))

    def test_fault_nnc_direction(self):
        """NNCs should connect i=0 column to i=1 column (I-direction fault)."""
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # All NNCs should have i1=0 and i2=1
        assert np.all(r.nnc_i1 == 0)
        assert np.all(r.nnc_i2 == 1)

    def test_fault_nnc_different_k(self):
        """Fault NNCs must have k1 != k2 (cross-layer connection)."""
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        fault_mask = r.nnc_type == int(NNCType.Fault)
        assert np.all(r.nnc_k1[fault_mask] != r.nnc_k2[fault_mask])

    def test_fault_nnc_positive_T(self):
        """All fault NNC transmissibilities should be strictly positive."""
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        fault_mask = r.nnc_type == int(NNCType.Fault)
        assert np.all(r.nnc_T[fault_mask] > 0.0)

    def test_fault_nnc_reduced_tranx(self):
        """Due to partial face overlap, TRANX values should be < 1.0 (< unit T)."""
        nc, nr, nl = 2, 1, 2
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=0.7)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # TRANX holds same-K connections; 0.7 offset → 0.3 unit overlap area
        # T = k_eff * overlap_area / (d1 + d2) ... with both halves = 0.30
        active_tranx = r.tranx[~np.isnan(r.tranx)]
        assert np.all(active_tranx < 1.0)

    def test_no_nonexistent_z_overlap_nncs(self):
        """A very large Z offset so no same-K or cross-K faces overlap should
        produce no NNCs and NaN for all TRANX values."""
        nc, nr, nl = 2, 1, 2
        # Offset by 100 units — no Z overlap possible
        gcpp = _make_faulted_grid_cpp(nc, nr, nl, fault_column_i=1, z_offset=100.0)
        px = _uniform_perm(nc, nr, nl)
        r = compute_transmissibilities(gcpp, px, px, px, px)
        # No TRANX (NaN = no valid same-K overlap) and no fault NNCs
        assert len(r.nnc_T) == 0 or np.all(r.nnc_type != int(NNCType.Fault))
        # All tranx entries for this pair should be NaN (or 0 if area=0)
        assert not np.any(r.tranx > 0)
