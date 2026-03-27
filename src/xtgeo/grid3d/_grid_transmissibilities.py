"""Private module, Grid ETC 1 methods, info/modify/report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import null_logger

from ._ecl_grid import Units as _GridUnits
from .grid_property import GridProperty

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid

logger = null_logger(__name__)

# ---------------------------------------------------------------------------
# Darcy unit-conversion factors for transmissibility
# T_ecl = _DARCY_FACTOR * k [mD] * A [length²] / d [length]
#   METRES: T in m³·cP/(d·bar)     C = 9.869233e-16 [m²/mD] * 1e3 [1/(Pa·s) per 1/cP]
#                                       * 86400 [s/d] * 1e5 [Pa/bar] ≈ 8.527e-3
#   FEET  : T in rb·cP/(d·psi)     C ≈ 1.127e-3  (standard «darcy» field factor)
# ---------------------------------------------------------------------------
_DARCY_METRES: float = 9.869233e-16 * 1e3 * 86400 * 1e5  # ≈ 8.527e-3
_DARCY_FEET: float = 1.127e-3


def _transmissibility_factor(grid: Grid) -> float:
    """Return the Darcy unit-conversion factor for transmissibility.

    Result depends on the coordinate units stored on the grid.
    Grids without units (e.g. synthetic box grids) default to METRES.
    """
    units = grid.units
    if units == _GridUnits.FEET:
        return _DARCY_FEET
    if units == _GridUnits.CM:
        # cm² / cm = cm; 1 cm = 0.01 m → T_raw_cm = T_raw_m * 100
        return _DARCY_METRES * 1e-2
    # METRES or None → Eclipse METRIC convention
    return _DARCY_METRES


def _to_property_array(
    prop: GridProperty | float | None, nc: int, nr: int, nl: int
) -> np.ndarray:
    """Return a (nc, nr, nl) float64 array from a GridProperty or scalar float."""
    if prop is None or isinstance(prop, (int, float)):
        return np.full(
            (nc, nr, nl), 1.0 if prop is None else float(prop), dtype=np.float64
        )
    return np.asarray(prop.values.filled(0.0), dtype=np.float64)


def _get_cell_mask(
    permx: GridProperty | float,
    nc: int,
    nr: int,
    nl: int,
) -> np.ndarray:
    """Return boolean mask (True = inactive/masked) derived from permx."""
    if isinstance(permx, GridProperty):
        return np.ma.getmaskarray(permx.values).reshape(nc, nr, nl)
    return np.zeros((nc, nr, nl), dtype=bool)


def get_transmissibilities(
    grid: Grid,
    permx: GridProperty | float,
    permy: GridProperty | float,
    permz: GridProperty | float,
    ntg: GridProperty | float = 1.0,
    min_dz_pinchout: float = 1e-4,
    min_fault_throw: float = 0.0,
    nested_id_property: GridProperty | None = None,
    search_radius: float = 600.0,
    min_area: float = 1e-3,
) -> tuple[
    GridProperty,
    GridProperty,
    GridProperty,
    pd.DataFrame,
    pd.DataFrame | None,
    GridProperty | None,
]:
    """Compute TPFA transmissibilities for a corner-point grid.

    Returns three GridProperty objects (tranx, trany, tranz), a NNC DataFrame,
    and optionally nested-hybrid NNC results.
    See Grid.get_transmissibilities() for full documentation.
    """
    grid._set_xtgformat2()
    gcpp = grid._get_grid_cpp()

    nc, nr, nl = grid.ncol, grid.nrow, grid.nlay

    px = _to_property_array(permx, nc, nr, nl)
    py = _to_property_array(permy, nc, nr, nl)
    pz = _to_property_array(permz, nc, nr, nl)
    nt = _to_property_array(ntg, nc, nr, nl)
    cell_mask = _get_cell_mask(permx, nc, nr, nl)

    r = _internal.grid3d.compute_transmissibilities(
        gcpp, px, py, pz, nt, min_dz_pinchout
    )

    # Convert raw TPFA values from mD·length to Eclipse'ish transmissibility units
    # (m³·cP/(d·bar) for METRIC; rb·cP/(d·psi) for FIELD).
    factor = _transmissibility_factor(grid)

    # r.trany[i,j,k] = T between cell (i,j,k) and (i,j+1,k), computed from
    # actual XY geometry regardless of handedness.  The sentinel (no j+1
    # neighbour beyond the boundary) therefore always belongs at j=nrow-1.
    tranx_raw = np.pad(
        np.asarray(r.tranx, dtype=np.float64) * factor,
        ((0, 1), (0, 0), (0, 0)),
        constant_values=0.0,
    )
    trany_raw = np.pad(
        np.asarray(r.trany, dtype=np.float64) * factor,
        ((0, 0), (0, 1), (0, 0)),
        constant_values=0.0,
    )
    tranz_raw = np.pad(
        np.asarray(r.tranz, dtype=np.float64) * factor,
        ((0, 0), (0, 0), (0, 1)),
        constant_values=0.0,
    )

    tranx = GridProperty(
        ncol=nc,
        nrow=nr,
        nlay=nl,
        name="TRANX",
        values=np.ma.MaskedArray(np.nan_to_num(tranx_raw, nan=0.0), mask=cell_mask),
        discrete=False,
    )
    trany = GridProperty(
        ncol=nc,
        nrow=nr,
        nlay=nl,
        name="TRANY",
        values=np.ma.MaskedArray(np.nan_to_num(trany_raw, nan=0.0), mask=cell_mask),
        discrete=False,
    )
    tranz = GridProperty(
        ncol=nc,
        nrow=nr,
        nlay=nl,
        name="TRANZ",
        values=np.ma.MaskedArray(np.nan_to_num(tranz_raw, nan=0.0), mask=cell_mask),
        discrete=False,
    )

    # Convert 0-based C++ indices to 1-based public API convention
    nnc_df = pd.DataFrame(
        {
            "I1": np.asarray(r.nnc_i1, dtype=np.int32) + 1,
            "J1": np.asarray(r.nnc_j1, dtype=np.int32) + 1,
            "K1": np.asarray(r.nnc_k1, dtype=np.int32) + 1,
            "I2": np.asarray(r.nnc_i2, dtype=np.int32) + 1,
            "J2": np.asarray(r.nnc_j2, dtype=np.int32) + 1,
            "K2": np.asarray(r.nnc_k2, dtype=np.int32) + 1,
            "T": np.asarray(r.nnc_T, dtype=np.float64) * factor,
            "TYPE": ["Fault" if int(t) == 0 else "Pinchout" for t in r.nnc_type],
        }
    )

    # Drop NNC connections that involve at least one masked (inactive) cell
    if cell_mask.any() and len(nnc_df) > 0:
        c1_masked = cell_mask[
            nnc_df["I1"].values - 1,
            nnc_df["J1"].values - 1,
            nnc_df["K1"].values - 1,
        ]
        c2_masked = cell_mask[
            nnc_df["I2"].values - 1,
            nnc_df["J2"].values - 1,
            nnc_df["K2"].values - 1,
        ]
        nnc_df = nnc_df[~(c1_masked | c2_masked)].reset_index(drop=True)

    # Drop Fault NNCs whose vertical throw is below the user-specified threshold.
    # Faults with a near-zero throw are geometric artefacts ("numerical faults")
    # and should not produce NNC connections.
    #
    # Throw is measured at the shared fault face by comparing the Z-coordinates
    # of the facing corners of cell(i1,j1,k1) and cell(i2,j2,k1) — same K layer,
    # neighbour column.  In a corner-point grid those two cells sit on IDENTICAL
    # shared pillar nodes when the interface is unfaulted, so the Z difference is
    # exactly zero.  A real fault shifts the pillar system on one side, giving a
    # non-zero Z displacement at the shared interface.
    #
    # We deliberately use k1 (not k2) for cell2 because k2 ≠ k1 for an NNC —
    # using k2 would measure a layer-depth difference, not a fault throw.
    if min_fault_throw > 0.0 and len(nnc_df) > 0:
        fault_sel = nnc_df["TYPE"] == "Fault"
        if fault_sel.any():
            fault_rows = nnc_df[fault_sel]
            i1a = fault_rows["I1"].values - 1
            j1a = fault_rows["J1"].values - 1
            k1a = fault_rows["K1"].values - 1
            i2a = fault_rows["I2"].values - 1
            j2a = fault_rows["J2"].values - 1

            # Cache cell corners to avoid repeated C++ calls for shared cells.
            corner_cache: dict[tuple[int, int, int], object] = {}

            def _get_corners(ci: int, cj: int, ck: int) -> object:
                key = (ci, cj, ck)
                if key not in corner_cache:
                    corner_cache[key] = gcpp.get_cell_corners_from_ijk(ci, cj, ck)
                return corner_cache[key]

            significant = np.zeros(len(fault_rows), dtype=bool)
            for idx in range(len(fault_rows)):
                i1_, j1_, k1_ = int(i1a[idx]), int(j1a[idx]), int(k1a[idx])
                i2_, j2_ = int(i2a[idx]), int(j2a[idx])
                di, dj = i2_ - i1_, j2_ - j1_
                # attrs1[n] and attrs2[n] are on the same corner-point pillar when
                # the interface is unfaulted → Z difference = 0.
                attrs1 = _FACE_CORNER_ATTRS[(di, dj, 0)]
                attrs2 = _FACE_CORNER_ATTRS[(-di, -dj, 0)]
                cc1 = _get_corners(i1_, j1_, k1_)
                cc2 = _get_corners(i2_, j2_, k1_)  # same K layer as cell1
                max_throw = max(
                    abs(getattr(cc1, a1).z - getattr(cc2, a2).z)
                    for a1, a2 in zip(attrs1, attrs2)
                )
                significant[idx] = max_throw >= min_fault_throw

            keep = ~fault_sel.values  # always keep non-Fault rows
            keep[fault_sel.values] = significant
            nnc_df = nnc_df[keep].reset_index(drop=True)

    if nested_id_property is not None:
        nnc_nested_df, refined_boundary_prop = get_nnc_nested_hybrid(
            grid,
            permx,
            permy,
            permz,
            ntg,
            nested_id_property,
            search_radius=search_radius,
            min_area=min_area,
        )
    else:
        nnc_nested_df = None
        refined_boundary_prop = None

    return tranx, trany, tranz, nnc_df, nnc_nested_df, refined_boundary_prop


# ---------------------------------------------------------------------------
# Nested hybrid NNC helpers
# ---------------------------------------------------------------------------

# (di, dj, dk) → CellFaceLabel (initialised once at import time)
_DIR_TO_FACE_LABEL: dict[tuple[int, int, int], "_internal.grid3d.CellFaceLabel"] = {}


def _init_face_labels() -> None:
    F = _internal.grid3d.CellFaceLabel
    _DIR_TO_FACE_LABEL.update(
        {
            (1, 0, 0): F.East,
            (-1, 0, 0): F.West,
            (0, 1, 0): F.North,
            (0, -1, 0): F.South,
            (0, 0, -1): F.Top,
            (0, 0, 1): F.Bottom,
        }
    )


_init_face_labels()

# Attribute names for the 4 face-corner points of each face direction.
_FACE_CORNER_ATTRS: dict[tuple[int, int, int], tuple[str, str, str, str]] = {
    (1, 0, 0): ("upper_se", "upper_ne", "lower_ne", "lower_se"),
    (-1, 0, 0): ("upper_sw", "upper_nw", "lower_nw", "lower_sw"),
    (0, 1, 0): ("upper_nw", "upper_ne", "lower_ne", "lower_nw"),
    (0, -1, 0): ("upper_sw", "upper_se", "lower_se", "lower_sw"),
    (0, 0, -1): ("upper_sw", "upper_se", "upper_ne", "upper_nw"),
    (0, 0, 1): ("lower_sw", "lower_se", "lower_ne", "lower_nw"),
}


def _collect_nh_boundary_faces(
    gcpp: "_internal.grid3d.Grid",
    nv: np.ndarray,
    nest_id: int,
    ncol: int,
    nrow: int,
    nlay: int,
    cell_mask: np.ndarray | None = None,
) -> list[
    tuple[
        np.ndarray,
        tuple[int, int, int],
        object,
        "_internal.grid3d.CellFaceLabel",
        tuple[int, int, int],
    ]
]:
    """Return hole-facing boundary faces for one NEST_ID region.

    A face qualifies when its immediate IJK neighbour has NEST_ID == 0 (the hole).
    Source cells that are masked (inactive) are skipped.  The hole cells themselves
    are intentionally inactive/masked and must NOT be filtered out — their NEST_ID==0
    status is exactly what identifies the boundary.

    Returns a list of ``(centroid_xyz, ijk, cell_corners, face_label, dir_tuple)``
    where *dir_tuple* is the (di, dj, dk) offset into the hole.
    """
    faces = []
    for i in range(ncol):
        for j in range(nrow):
            for k in range(nlay):
                if nv[i, j, k] != nest_id:
                    continue
                if cell_mask is not None and cell_mask[i, j, k]:
                    continue
                cc = gcpp.get_cell_corners_from_ijk(i, j, k)
                for (di, dj, dk), attrs in _FACE_CORNER_ATTRS.items():
                    ni, nj, nk = i + di, j + dj, k + dk
                    if not (0 <= ni < ncol and 0 <= nj < nrow and 0 <= nk < nlay):
                        continue
                    if nv[ni, nj, nk] != 0:
                        continue
                    # Hole cells (NEST_ID==0) are intentionally inactive — do NOT
                    # filter them by cell_mask; their inactive status is expected.
                    pts = [getattr(cc, a) for a in attrs]
                    centroid = np.array(
                        [
                            sum(p.x for p in pts) / 4,
                            sum(p.y for p in pts) / 4,
                            sum(p.z for p in pts) / 4,
                        ]
                    )
                    faces.append(
                        (
                            centroid,
                            (i, j, k),
                            cc,
                            _DIR_TO_FACE_LABEL[(di, dj, dk)],
                            (di, dj, dk),
                        )
                    )
    return faces


def _nh_k_eff(
    face_dir: tuple[int, int, int],
    ijk: tuple[int, int, int],
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    nt: np.ndarray,
) -> float:
    """Effective TPFA permeability for one face: perm*ntg (I/J) or permz (K)."""
    i, j, k = ijk
    if face_dir[0] != 0:  # I-direction
        return float(px[i, j, k]) * float(nt[i, j, k])

    if face_dir[1] != 0:  # J-direction
        return float(py[i, j, k]) * float(nt[i, j, k])

    return float(pz[i, j, k])


def _nh_tpfa(
    k1: float, k2: float, area: float, d1: float, d2: float, factor: float
) -> float:
    """TPFA transmissibility in Darcy-converted units."""
    if area <= 0.0 or d1 <= 0.0 or d2 <= 0.0:
        return 0.0
    ht1 = k1 * area / d1
    ht2 = k2 * area / d2
    denom = ht1 + ht2
    return factor * ht1 * ht2 / denom if denom > 0.0 else 0.0


def get_nnc_nested_hybrid(
    grid: "Grid",
    permx: "GridProperty | float",
    permy: "GridProperty | float",
    permz: "GridProperty | float",
    ntg: "GridProperty | float",
    nested_id_property: "GridProperty",
    search_radius: float = 600.0,
    min_area: float = 1e-3,
) -> tuple[pd.DataFrame, "GridProperty"]:
    """Compute NNC transmissibilities across the boundary of a nested hybrid grid.

    See :meth:`xtgeo.grid3d.Grid.get_nnc_nested_hybrid` for full documentation.
    """
    from scipy.spatial import KDTree

    grid._set_xtgformat2()
    gcpp = grid._get_grid_cpp()

    nc, nr, nl = grid.ncol, grid.nrow, grid.nlay
    factor = _transmissibility_factor(grid)

    px = _to_property_array(permx, nc, nr, nl)
    py = _to_property_array(permy, nc, nr, nl)
    pz = _to_property_array(permz, nc, nr, nl)
    nt = _to_property_array(ntg, nc, nr, nl)
    nv = np.asarray(nested_id_property.values.filled(0), dtype=np.int32)
    cell_mask = _get_cell_mask(permx, nc, nr, nl)

    # --- Collect hole-facing boundary faces for each region ---
    mother_faces = _collect_nh_boundary_faces(gcpp, nv, 1, nc, nr, nl, cell_mask)
    refined_faces = _collect_nh_boundary_faces(gcpp, nv, 2, nc, nr, nl, cell_mask)

    if not mother_faces or not refined_faces:
        empty_df = pd.DataFrame(
            columns=["I1", "J1", "K1", "I2", "J2", "K2", "T", "TYPE", "DIRECTION"]
        )
        empty_prop = GridProperty(
            ncol=nc,
            nrow=nr,
            nlay=nl,
            values=np.zeros((nc, nr, nl), dtype=np.int32),
            name="NNC_REFINED_BOUNDARY",
            discrete=True,
            codes={0: "none", 1: "refined_boundary"},
        )
        return empty_df, empty_prop

    # --- KDTree on mother face centroids for fast candidate proximity search ---
    # Face centroids across the shared hole are typically 20–500 m apart in 3-D.
    # The polygon-overlap test below rejects any spurious proximity matches.
    mother_ctrs = np.array([f[0] for f in mother_faces])
    tree = KDTree(mother_ctrs)

    # --- Match refined faces to mother faces and compute T --------------------
    # Convention: I1/J1/K1 is always the refined (NestReg=2) cell;
    #             I2/J2/K2 is always the mother (NestReg=1) cell.
    _DIR_LABEL: dict[tuple[int, int, int], str] = {
        (1, 0, 0): "I+",
        (-1, 0, 0): "I-",
        (0, 1, 0): "J+",
        (0, -1, 0): "J-",
        (0, 0, 1): "K+",
        (0, 0, -1): "K-",
    }
    rows: list[dict] = []
    refined_boundary_ijk: set[tuple[int, int, int]] = set()

    for ctr_r, ijk_r, cc_r, face_r, dir_r in refined_faces:
        for idx in tree.query_ball_point(ctr_r, r=search_radius):
            _, ijk_m, cc_m, face_m, dir_m = mother_faces[idx]
            # Only attempt overlap for faces that point in opposite directions
            # (one faces the hole from the refined side, the other from the mother side).
            # Perpendicular face pairs would produce a spurious non-zero area from the
            # projection-based overlap algorithm.
            if (dir_r[0] + dir_m[0], dir_r[1] + dir_m[1], dir_r[2] + dir_m[2]) != (
                0,
                0,
                0,
            ):
                continue
            fr = _internal.grid3d.face_overlap_result(cc_r, face_r, cc_m, face_m)
            if fr.area <= min_area:
                continue
            ke_r = _nh_k_eff(dir_r, ijk_r, px, py, pz, nt)
            ke_m = _nh_k_eff(dir_m, ijk_m, px, py, pz, nt)
            T = _nh_tpfa(ke_r, ke_m, fr.area, fr.d1, fr.d2, factor)
            rows.append(
                {
                    # 1-based cell indices (public API convention)
                    # I1/J1/K1 = refined cell, I2/J2/K2 = mother cell
                    "I1": ijk_r[0] + 1,
                    "J1": ijk_r[1] + 1,
                    "K1": ijk_r[2] + 1,
                    "I2": ijk_m[0] + 1,
                    "J2": ijk_m[1] + 1,
                    "K2": ijk_m[2] + 1,
                    "T": T,
                    "TYPE": "NestedHybrid",
                    "DIRECTION": _DIR_LABEL.get(dir_r, "?"),
                }
            )
            refined_boundary_ijk.add(ijk_r)

    nnc_df = pd.DataFrame(rows)

    # --- Mark refined cells that are the first cell in at least one NNC -------
    flag = np.zeros((nc, nr, nl), dtype=np.int32)
    for ijk in refined_boundary_ijk:
        flag[ijk] = 1

    refined_boundary_prop = GridProperty(
        ncol=nc,
        nrow=nr,
        nlay=nl,
        values=flag,
        name="NNC_REFINED_BOUNDARY",
        discrete=True,
        codes={0: "none", 1: "refined_boundary"},
    )

    return nnc_df, refined_boundary_prop
