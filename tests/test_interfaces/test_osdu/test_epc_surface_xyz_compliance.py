"""Surface, Points, and Polygons compliance tests for EPC roundtrip.

Exercises scenarios from the surface/xyz test suites that are missing from
the OSDU path: large surfaces, surface operations post-roundtrip, many
points, complex polygons, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
from xtgeo.interfaces.osdu._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
from xtgeo.interfaces.osdu._polyline import (
    polylineset_to_xtgeo,
    xtgeo_polygons_to_resqml,
)


@pytest.fixture
def epc_path(tmp_path):
    return str(tmp_path / "test.epc")


def _roundtrip_surface(epc_path, surf, title="Surf", **kw):
    """Write surface to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_surface_to_resqml(p, surf, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    s2 = grid2d_to_xtgeo(p2, uuids[title])
    p2.close()
    return s2


def _roundtrip_points(epc_path, pts, title="Pts", **kw):
    """Write points to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_points_to_resqml(p, pts, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    pts2 = pointset_to_xtgeo(p2, uuids[title])
    p2.close()
    return pts2


def _roundtrip_polygons(epc_path, polys, title="Poly", **kw):
    """Write polygons to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_polygons_to_resqml(p, polys, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    polys2 = polylineset_to_xtgeo(p2, uuids[title])
    p2.close()
    return polys2


# ---------------------------------------------------------------------------
# Surface compliance
# ---------------------------------------------------------------------------


class TestSurfaceCompliance:
    """Extended surface roundtrip and post-roundtrip operation tests."""

    def test_large_surface(self, epc_path):
        """100x120 surface (12000 nodes) roundtrip."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=100,
            nrow=120,
            xinc=25.0,
            yinc=25.0,
            xori=460000,
            yori=5930000,
            values=rng.rand(100, 120) * 500 + 1000,
        )

        s2 = _roundtrip_surface(epc_path, s, title="LargeSurf", crs_epsg=23031)

        assert s2.ncol == 100
        assert s2.nrow == 120
        assert np.allclose(s2.values.filled(np.nan), s.values.filled(np.nan), atol=1e-4)

    def test_surface_with_many_nans(self, epc_path):
        """Surface where 50% of values are NaN."""
        rng = np.random.RandomState(42)
        vals = rng.rand(20, 25) * 100
        mask = rng.rand(20, 25) > 0.5
        vals[mask] = np.nan

        s = xtgeo.RegularSurface(
            ncol=20, nrow=25, xinc=10, yinc=10, xori=0, yori=0, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="NanHeavy")

        orig = s.values.filled(np.nan)
        read = s2.values.filled(np.nan)
        assert np.array_equal(np.isnan(orig), np.isnan(read))
        valid = ~np.isnan(orig)
        assert np.allclose(orig[valid], read[valid], atol=1e-6)

    def test_surface_negative_values(self, epc_path):
        """Surface with negative Z values (above sea level)."""
        vals = np.linspace(-500, 500, 30).reshape(5, 6)
        s = xtgeo.RegularSurface(
            ncol=5, nrow=6, xinc=100, yinc=100, xori=0, yori=0, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="NegSurf")

        assert np.allclose(s2.values.filled(np.nan), vals, atol=1e-6)

    def test_surface_constant_value(self, epc_path):
        """Surface where all values are identical (flat horizon)."""
        vals = np.full((8, 10), 2500.0)
        s = xtgeo.RegularSurface(
            ncol=8, nrow=10, xinc=50, yinc=50, xori=460000, yori=5930000, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="FlatSurf")

        assert np.allclose(s2.values.filled(np.nan), vals, atol=1e-6)

    @pytest.mark.parametrize("rotation", [0.0, 15.5, 45.0, 89.9])
    def test_surface_rotation_parametrized(self, epc_path, rotation):
        """Multiple rotation angles roundtrip correctly."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=5,
            nrow=7,
            xinc=25.0,
            yinc=30.0,
            xori=460000,
            yori=5930000,
            rotation=rotation,
            values=rng.rand(5, 7) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="RotSurf")

        assert abs(s2.rotation - rotation) < 0.1

    def test_surface_asymmetric_increments(self, epc_path):
        """xinc != yinc should be preserved."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=6,
            nrow=10,
            xinc=12.5,
            yinc=37.5,
            xori=100,
            yori=200,
            values=rng.rand(6, 10) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="AsymSurf")

        assert abs(s2.xinc - 12.5) < 0.01
        assert abs(s2.yinc - 37.5) < 0.01

    def test_surface_operations_post_roundtrip(self, epc_path):
        """Surface operations (statistics, resample) work after roundtrip."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=10,
            nrow=12,
            xinc=25.0,
            yinc=25.0,
            xori=460000,
            yori=5930000,
            values=rng.rand(10, 12) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="OpsSurf", crs_epsg=23031)

        # Basic statistics should match
        assert abs(s2.values.mean() - s.values.mean()) < 0.01
        assert abs(s2.values.std() - s.values.std()) < 0.01
        assert abs(s2.values.min() - s.values.min()) < 0.01
        assert abs(s2.values.max() - s.values.max()) < 0.01


# ---------------------------------------------------------------------------
# Points compliance
# ---------------------------------------------------------------------------


class TestPointsCompliance:
    """Extended PointSet roundtrip tests."""

    def test_many_points(self, epc_path):
        """1000 randomly distributed points."""
        rng = np.random.RandomState(42)
        data = np.column_stack(
            [
                rng.uniform(460000, 461000, 1000),
                rng.uniform(5930000, 5931000, 1000),
                rng.uniform(1000, 3000, 1000),
            ]
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="ManyPts", crs_epsg=23031)

        df2 = pts2.get_dataframe()
        assert len(df2) == 1000
        assert np.allclose(df2.values[:, :3], data, atol=1e-4)

    def test_single_point(self, epc_path):
        """Single point roundtrip."""
        data = np.array([[460000, 5930000, 1500]], dtype=np.float64)
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="SinglePt")

        df2 = pts2.get_dataframe()
        assert len(df2) == 1
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)

    def test_points_negative_z(self, epc_path):
        """Points with negative Z values."""
        data = np.array(
            [
                [0, 0, -100],
                [100, 100, -50],
                [200, 200, 0],
                [300, 300, 50],
            ],
            dtype=np.float64,
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="NegZPts")

        df2 = pts2.get_dataframe()
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)

    def test_points_collocated(self, epc_path):
        """Multiple points at the same location."""
        data = np.array(
            [
                [100, 200, 300],
                [100, 200, 300],
                [100, 200, 400],
            ],
            dtype=np.float64,
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="CollocPts")

        df2 = pts2.get_dataframe()
        assert len(df2) == 3
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)


# ---------------------------------------------------------------------------
# Polygons compliance
# ---------------------------------------------------------------------------


class TestPolygonsCompliance:
    """Extended PolylineSet (Polygons) roundtrip tests."""

    def test_single_polygon(self, epc_path):
        """Single closed polygon."""
        df = pd.DataFrame(
            {
                "X_UTME": [0, 100, 100, 0, 0],
                "Y_UTMN": [0, 0, 100, 100, 0],
                "Z_TVDSS": [1000, 1000, 1000, 1000, 1000],
                "POLY_ID": [0, 0, 0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="SinglePoly")

        df2 = polys2.get_dataframe()
        assert len(df2) == 5
        assert np.allclose(
            df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            df2[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            atol=1e-6,
        )

    def test_many_polygons(self, epc_path):
        """5 separate polygons with different sizes."""
        frames = []
        for pid in range(5):
            n = 4 + pid  # varying number of vertices
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x = 100 * pid + 50 * np.cos(angles)
            y = 50 * np.sin(angles)
            z = np.full(n, 1000 + pid * 100, dtype=np.float64)
            # Close the polygon
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            frames.append(
                pd.DataFrame(
                    {
                        "X_UTME": x,
                        "Y_UTMN": y,
                        "Z_TVDSS": z,
                        "POLY_ID": pid,
                    }
                )
            )

        df = pd.concat(frames, ignore_index=True)
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="ManyPolys")

        df2 = polys2.get_dataframe()
        assert len(df2) == len(df)
        assert set(df2["POLY_ID"].unique()) == set(range(5))
        assert np.allclose(
            df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            df2[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            atol=1e-6,
        )

    def test_polygon_varying_z(self, epc_path):
        """Polygon with varying Z values (3D fault trace)."""
        df = pd.DataFrame(
            {
                "X_UTME": [0, 100, 200, 300, 200, 100, 0],
                "Y_UTMN": [0, 50, 0, 50, 100, 50, 0],
                "Z_TVDSS": [1000, 1100, 1200, 1300, 1200, 1100, 1000],
                "POLY_ID": [0, 0, 0, 0, 0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="3DPoly")

        df2 = polys2.get_dataframe()
        assert np.allclose(df["Z_TVDSS"].values, df2["Z_TVDSS"].values, atol=1e-6)
