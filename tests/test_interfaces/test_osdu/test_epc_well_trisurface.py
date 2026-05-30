"""EPC round-trip tests for Well, BlockedWell, and TriangulatedSurface."""

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._triangulated_surface import (
    triangulated_surface_to_xtgeo,
    xtgeo_triangulated_surface_to_resqml,
)
from xtgeo.interfaces.osdu._well import well_to_xtgeo, xtgeo_well_to_resqml
from xtgeo.interfaces.osdu._blocked_well import (
    blocked_well_to_xtgeo,
    xtgeo_blocked_well_to_resqml,
)


@pytest.fixture
def epc_path(tmp_path):
    return str(tmp_path / "test.epc")


class TestTriangulatedSurfaceRoundTrip:
    """Tests for TriangulatedSurface round-trips via EPC."""

    def test_simple_triangle(self, epc_path):
        """Single triangle roundtrip."""
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_triangulated_surface_to_resqml(
            p, trisurf, title="Triangle", crs_epsg=23031
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        trisurf2 = triangulated_surface_to_xtgeo(p2, uuids["Triangle"])
        p2.close()

        assert np.allclose(trisurf.vertices, trisurf2.vertices, atol=1e-10)
        assert np.array_equal(trisurf.triangles, trisurf2.triangles)

    def test_multi_triangle_mesh(self, epc_path):
        """Multiple triangles forming a mesh."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_triangulated_surface_to_resqml(
            p, trisurf, title="Mesh", crs_epsg=23031
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        trisurf2 = triangulated_surface_to_xtgeo(p2, uuids["Mesh"])
        p2.close()

        assert trisurf2.vertices.shape == (4, 3)
        assert trisurf2.triangles.shape == (2, 3)
        assert np.allclose(trisurf.vertices, trisurf2.vertices)
        assert np.array_equal(trisurf.triangles, trisurf2.triangles)

    def test_high_level_api(self, epc_path):
        """Test via the high-level triangulated_surface_to/from_osdu."""
        vertices = np.array([[0, 0, 100], [10, 0, 100], [5, 10, 110]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        uuids = xtgeo.triangulated_surface_to_osdu(
            epc_path, trisurf, title="FaultPlane", crs_epsg=23031
        )
        assert "FaultPlane" in uuids

        trisurf2 = xtgeo.triangulated_surface_from_osdu(
            epc_path, uuid=uuids["FaultPlane"]
        )
        assert np.allclose(trisurf.vertices, trisurf2.vertices)


class TestWellRoundTrip:
    """Tests for Well (trajectory + logs) round-trips via EPC."""

    def _make_well(self):
        """Create a simple well with trajectory and logs."""
        npts = 10
        md = np.linspace(0, 1000, npts)
        x = np.full(npts, 460000.0)
        y = np.full(npts, 5930000.0)
        z = np.linspace(0, 1000, npts)
        df = pd.DataFrame(
            {
                "X_UTME": x,
                "Y_UTMN": y,
                "Z_TVDSS": z,
                "M_DEPTH": md,
                "GR": np.linspace(30, 120, npts),
                "PORO": np.linspace(0.1, 0.3, npts),
            }
        )
        return xtgeo.Well(
            xpos=460000.0,
            ypos=5930000.0,
            wname="WELL-A",
            df=df,
            mdlogname="M_DEPTH",
            zonelogname=None,
        )

    def test_trajectory_roundtrip(self, epc_path):
        """Test well trajectory geometry roundtrip."""
        well = self._make_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_well_to_resqml(
            p, well, title="WELL-A", crs_epsg=23031, export_logs=False
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        well2 = well_to_xtgeo(p2, uuids["WELL-A"], load_logs=False)
        p2.close()

        assert np.allclose(
            well.get_dataframe()["M_DEPTH"].values,
            well2.get_dataframe()[well2.mdlogname].values,
            atol=1e-6,
        )
        assert np.allclose(
            well.get_dataframe()["X_UTME"].values,
            well2.get_dataframe()["X_UTME"].values,
            atol=1e-6,
        )

    def test_trajectory_with_logs_roundtrip(self, epc_path):
        """Test well trajectory + logs roundtrip."""
        well = self._make_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_well_to_resqml(
            p, well, title="WELL-A", crs_epsg=23031, export_logs=True
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        well2 = well_to_xtgeo(p2, uuids["WELL-A"], load_logs=True)
        p2.close()

        df1 = well.get_dataframe()
        df2 = well2.get_dataframe()
        assert "GR" in df2.columns
        assert "PORO" in df2.columns
        assert np.allclose(df1["GR"].values, df2["GR"].values, atol=1e-6)
        assert np.allclose(df1["PORO"].values, df2["PORO"].values, atol=1e-6)
        # XYZ preserved
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)

    def test_high_level_api(self, epc_path):
        """Test via well_from_osdu / well_to_osdu."""
        well = self._make_well()

        uuids = xtgeo.well_to_osdu(epc_path, well, title="WELL-A", crs_epsg=23031)
        assert "WELL-A" in uuids

        well2 = xtgeo.well_from_osdu(epc_path, uuid=uuids["WELL-A"])
        df1 = well.get_dataframe()
        df2 = well2.get_dataframe()
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)
        assert well2.mdlogname is not None


class TestBlockedWellRoundTrip:
    """Tests for BlockedWell round-trips via EPC."""

    def _make_blocked_well(self):
        """Create a simple blocked well."""
        npts = 5
        md = np.linspace(100, 500, npts)
        x = np.full(npts, 460050.0)
        y = np.full(npts, 5930050.0)
        z = np.linspace(100, 500, npts)
        df = pd.DataFrame(
            {
                "X_UTME": x,
                "Y_UTMN": y,
                "Z_TVDSS": z,
                "M_DEPTH": md,
                "I_INDEX": np.array([1, 1, 2, 2, 3], dtype=np.int32),
                "J_INDEX": np.array([1, 2, 2, 3, 3], dtype=np.int32),
                "K_INDEX": np.array([1, 1, 2, 2, 3], dtype=np.int32),
                "Facies": np.array([1, 2, 1, 3, 2], dtype=np.int32),
            }
        )
        return xtgeo.BlockedWell(
            xpos=460050.0,
            ypos=5930050.0,
            wname="BW-1",
            df=df,
            mdlogname="M_DEPTH",
            zonelogname=None,
        )

    def test_blocked_well_roundtrip(self, epc_path):
        """Test blocked well geometry + properties roundtrip."""
        bwell = self._make_blocked_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_blocked_well_to_resqml(p, bwell, title="BW-1", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        bwell2 = blocked_well_to_xtgeo(p2, uuids["BW-1"])
        p2.close()

        df1 = bwell.get_dataframe()
        df2 = bwell2.get_dataframe()
        # MD values preserved (column name may differ after roundtrip)
        md1 = df1[bwell.mdlogname].values
        md2 = df2[bwell2.mdlogname].values
        assert np.allclose(md1, md2, atol=1e-6)
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)
        assert np.array_equal(
            df1["I_INDEX"].values.astype(int), df2["I_INDEX"].values.astype(int)
        )
        assert np.array_equal(
            df1["J_INDEX"].values.astype(int), df2["J_INDEX"].values.astype(int)
        )
        assert np.array_equal(
            df1["K_INDEX"].values.astype(int), df2["K_INDEX"].values.astype(int)
        )

    def test_high_level_api(self, epc_path):
        """Test via blocked_well_from_osdu / blocked_well_to_osdu."""
        bwell = self._make_blocked_well()

        uuids = xtgeo.blocked_well_to_osdu(
            epc_path, bwell, title="BW-1", crs_epsg=23031
        )
        assert "BW-1" in uuids

        bwell2 = xtgeo.blocked_well_from_osdu(epc_path, uuid=uuids["BW-1"])
        df1 = bwell.get_dataframe()
        df2 = bwell2.get_dataframe()
        md1 = df1[bwell.mdlogname].values
        md2 = df2[bwell2.mdlogname].values
        assert np.allclose(md1, md2, atol=1e-6)
