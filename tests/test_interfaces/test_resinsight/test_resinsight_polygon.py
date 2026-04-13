"""Tests for the ResInsight polygon interface.

Covers:
- :class:`PolygonDataResInsight` data container (validation, eq, roundtrip)
- :meth:`PolygonDataResInsight.to_xtgeo_polygons`
- :meth:`PolygonDataResInsight.from_xtgeo_polygons`
- :meth:`PolygonDataResInsight.from_xtgeo_polygons_all`
- :class:`PolygonReader` and :class:`PolygonWriter` (live ResInsight tests)
"""

from __future__ import annotations

import pandas as pd
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.interfaces.resinsight._polygon import (
    PolygonDataResInsight,
    PolygonReader,
    PolygonWriter,
)

# ---------------------------------------------------------------------------
# PolygonDataResInsight – construction and validation
# ---------------------------------------------------------------------------


def test_empty_coordinates_raises():
    with pytest.raises(ValueError, match="coordinates cannot be empty"):
        PolygonDataResInsight(name="p", coordinates=[])


def test_bad_coordinate_length_raises():
    with pytest.raises(ValueError, match="point 1 has 2 elements"):
        PolygonDataResInsight(name="p", coordinates=[[0.0, 0.0, 0.0], [1.0, 2.0]])


def test_eq_same():
    coords = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    a = PolygonDataResInsight(name="p", coordinates=coords)
    b = PolygonDataResInsight(name="p", coordinates=coords)
    assert a == b


def test_eq_different_name():
    coords = [[0.0, 1.0, 2.0]]
    a = PolygonDataResInsight(name="p1", coordinates=coords)
    b = PolygonDataResInsight(name="p2", coordinates=coords)
    assert a != b


def test_eq_different_coords():
    a = PolygonDataResInsight(name="p", coordinates=[[0.0, 0.0, 0.0]])
    b = PolygonDataResInsight(name="p", coordinates=[[1.0, 0.0, 0.0]])
    assert a != b


def test_not_equal_to_other_type():
    a = PolygonDataResInsight(name="p", coordinates=[[0.0, 0.0, 0.0]])
    assert a.__eq__("string") is NotImplemented


def test_unhashable():
    a = PolygonDataResInsight(name="p", coordinates=[[0.0, 0.0, 0.0]])
    with pytest.raises(TypeError):
        hash(a)


# ---------------------------------------------------------------------------
# to_xtgeo_polygons
# ---------------------------------------------------------------------------


def test_to_xtgeo_polygons_basic():
    coords = [[100.0, 200.0, -1000.0], [110.0, 210.0, -1010.0], [120.0, 220.0, -1020.0]]
    data = PolygonDataResInsight(name="fault", coordinates=coords)
    poly = data.to_xtgeo_polygons()

    assert isinstance(poly, xtgeo.Polygons)
    assert poly.name == "fault"
    df = poly.get_dataframe()
    assert len(df) == 3
    assert list(df["POLY_ID"].unique()) == [0]
    assert_allclose(df["X_UTME"].values, [100.0, 110.0, 120.0])
    assert_allclose(df["Y_UTMN"].values, [200.0, 210.0, 220.0])
    assert_allclose(df["Z_TVDSS"].values, [-1000.0, -1010.0, -1020.0])


def test_to_xtgeo_polygons_single_point():
    data = PolygonDataResInsight(name="pt", coordinates=[[1.0, 2.0, 3.0]])
    poly = data.to_xtgeo_polygons()
    assert len(poly.get_dataframe()) == 1


# ---------------------------------------------------------------------------
# from_xtgeo_polygons
# ---------------------------------------------------------------------------


def _make_polygons(n_segments: int = 1, pts_per_seg: int = 3) -> xtgeo.Polygons:
    """Helper: build a Polygons object with n_segments segments."""
    rows = []
    for seg in range(n_segments):
        for i in range(pts_per_seg):
            rows.append(
                {
                    "X_UTME": float(seg * 100 + i),
                    "Y_UTMN": float(seg * 100 + i + 10),
                    "Z_TVDSS": float(-seg * 10 - i),
                    "POLY_ID": seg,
                }
            )
    df = pd.DataFrame(rows)
    return xtgeo.Polygons(values=df, name="test_poly")


def test_from_xtgeo_polygons_default_poly_id():
    """None default selects the first (lowest) POLY_ID, even when it is not 0."""
    poly = _make_polygons(n_segments=2)
    # Default (poly_id=None) should pick the first existing segment (POLY_ID=0)
    data = PolygonDataResInsight.from_xtgeo_polygons(poly)
    assert data.name == "test_poly"
    assert len(data.coordinates) == 3
    assert_allclose(data.coordinates[0], [0.0, 10.0, 0.0])


def test_from_xtgeo_polygons_default_selects_first_when_not_zero():
    """When POLY_ID numbering starts at a value other than 0 the default still works."""
    rows = [
        {"X_UTME": 5.0, "Y_UTMN": 15.0, "Z_TVDSS": -5.0, "POLY_ID": 1},
        {"X_UTME": 6.0, "Y_UTMN": 16.0, "Z_TVDSS": -6.0, "POLY_ID": 2},
    ]
    df = pd.DataFrame(rows)
    poly = xtgeo.Polygons(values=df, name="well_poly")
    # Default should select POLY_ID=1 (the minimum present), not raise
    data = PolygonDataResInsight.from_xtgeo_polygons(poly)
    assert len(data.coordinates) == 1
    assert_allclose(data.coordinates[0], [5.0, 15.0, -5.0])


def test_from_xtgeo_polygons_second_segment():
    poly = _make_polygons(n_segments=2)
    data = PolygonDataResInsight.from_xtgeo_polygons(poly, poly_id=1)
    assert len(data.coordinates) == 3
    assert_allclose(data.coordinates[0], [100.0, 110.0, -10.0])


def test_from_xtgeo_polygons_missing_poly_id_raises():
    poly = _make_polygons(n_segments=1)
    with pytest.raises(ValueError, match="POLY_ID=99"):
        PolygonDataResInsight.from_xtgeo_polygons(poly, poly_id=99)


# ---------------------------------------------------------------------------
# from_xtgeo_polygons_all
# ---------------------------------------------------------------------------


def test_from_xtgeo_polygons_all_single_segment():
    """Single-segment polygon gets a suffixed name for stable round-trips."""
    poly = _make_polygons(n_segments=1)
    result = PolygonDataResInsight.from_xtgeo_polygons_all(poly)
    assert len(result) == 1
    # Always suffixed — stable regardless of segment count
    assert result[0].name == "test_poly_0"


def test_from_xtgeo_polygons_all_multi_segment():
    poly = _make_polygons(n_segments=3)
    result = PolygonDataResInsight.from_xtgeo_polygons_all(poly)
    assert len(result) == 3
    assert result[0].name == "test_poly_0"
    assert result[1].name == "test_poly_1"
    assert result[2].name == "test_poly_2"


def test_from_xtgeo_polygons_all_preserves_coordinates():
    poly = _make_polygons(n_segments=2, pts_per_seg=4)
    result = PolygonDataResInsight.from_xtgeo_polygons_all(poly)
    for item in result:
        assert len(item.coordinates) == 4
        for pt in item.coordinates:
            assert len(pt) == 3


# ---------------------------------------------------------------------------
# Roundtrip: PolygonDataResInsight → XTGeo Polygons → PolygonDataResInsight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coords",
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[100.0, 200.0, -500.0], [110.0, 210.0, -510.0]],
        [[0.0, 0.0, 1000.0]],
    ],
    ids=["triangle", "two-points", "single-point"],
)
def test_roundtrip_via_xtgeo(coords: list[list[float]]):
    original = PolygonDataResInsight(name="rt", coordinates=coords)
    poly = original.to_xtgeo_polygons()
    recovered = PolygonDataResInsight.from_xtgeo_polygons(poly)
    assert recovered.name == original.name
    assert_allclose(recovered.coordinates, original.coordinates)


def test_from_xtgeo_polygons_all_stable_naming_single_vs_multi():
    """Names are always suffixed so segment count changes never orphan polygons."""
    single = _make_polygons(n_segments=1)
    result_single = PolygonDataResInsight.from_xtgeo_polygons_all(single)
    assert result_single[0].name == "test_poly_0"

    multi = _make_polygons(n_segments=2)
    result_multi = PolygonDataResInsight.from_xtgeo_polygons_all(multi)
    # The first segment has the same name as when there was only one segment
    assert result_multi[0].name == "test_poly_0"
    assert result_multi[1].name == "test_poly_1"


# ---------------------------------------------------------------------------
# Live ResInsight tests (require @pytest.mark.requires_resinsight)
# ---------------------------------------------------------------------------


@pytest.mark.requires_resinsight
def test_polygon_writer_creates_new(resinsight_instance):
    """PolygonWriter.save creates a new polygon when none exists."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    data = PolygonDataResInsight(name="NEW_POLY", coordinates=coords)

    writer = PolygonWriter(resinsight_instance)
    writer.save(data)

    reader = PolygonReader(resinsight_instance)
    loaded = reader.load("NEW_POLY")
    assert loaded.name == "NEW_POLY"
    assert_allclose(loaded.coordinates, coords)


@pytest.mark.requires_resinsight
def test_polygon_writer_updates_existing(resinsight_instance):
    """PolygonWriter.save updates coordinates of an existing polygon."""
    initial = PolygonDataResInsight(
        name="UPDATE_POLY", coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    PolygonWriter(resinsight_instance).save(initial)

    updated_coords = [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [6.0, 6.0, 5.0]]
    updated = PolygonDataResInsight(name="UPDATE_POLY", coordinates=updated_coords)
    PolygonWriter(resinsight_instance).save(updated)

    loaded = PolygonReader(resinsight_instance).load("UPDATE_POLY")
    assert_allclose(loaded.coordinates, updated_coords)


@pytest.mark.requires_resinsight
def test_polygon_reader_missing_raises(resinsight_instance):
    reader = PolygonReader(resinsight_instance)
    with pytest.raises(RuntimeError, match="Cannot find any polygon with name"):
        reader.load("DOES_NOT_EXIST_XYZ")


@pytest.mark.requires_resinsight
def test_polygon_reader_load_all(resinsight_instance):
    result = PolygonReader(resinsight_instance).load_all()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, PolygonDataResInsight)
        assert len(item.coordinates) > 0


@pytest.mark.requires_resinsight
def test_roundtrip_resinsight(resinsight_instance):
    """Full ResInsight roundtrip: write and read back preserves data exactly."""
    coords = [
        [100.0, 200.0, -1000.0],
        [200.0, 200.0, -1000.0],
        [200.0, 300.0, -1000.0],
        [100.0, 200.0, -1000.0],
    ]
    data = PolygonDataResInsight(name="ROUNDTRIP_POLY", coordinates=coords)
    PolygonWriter(resinsight_instance).save(data)
    loaded = PolygonReader(resinsight_instance).load("ROUNDTRIP_POLY")
    assert loaded == data


# ---------------------------------------------------------------------------
# Public API: polygons_from_resinsight / Polygons.to_resinsight
# ---------------------------------------------------------------------------


@pytest.mark.requires_resinsight
def test_public_api_polygons_from_resinsight(resinsight_instance):
    coords = [[10.0, 20.0, -100.0], [20.0, 20.0, -100.0], [20.0, 30.0, -100.0]]
    PolygonWriter(resinsight_instance).save(
        PolygonDataResInsight(name="PUBLIC_API_POLY", coordinates=coords)
    )
    poly = xtgeo.polygons_from_resinsight(resinsight_instance, "PUBLIC_API_POLY")
    assert isinstance(poly, xtgeo.Polygons)
    df = poly.get_dataframe()
    assert len(df) == 3
    assert_allclose(df["X_UTME"].values, [10.0, 20.0, 20.0])


@pytest.mark.requires_resinsight
def test_public_api_to_resinsight(resinsight_instance):
    df = pd.DataFrame(
        {
            "X_UTME": [50.0, 60.0, 60.0],
            "Y_UTMN": [50.0, 50.0, 60.0],
            "Z_TVDSS": [-500.0, -500.0, -500.0],
            "POLY_ID": [0, 0, 0],
        }
    )
    poly = xtgeo.Polygons(values=df, name="TO_RESINSIGHT")
    poly.to_resinsight(resinsight_instance)

    loaded = xtgeo.polygons_from_resinsight(resinsight_instance, "TO_RESINSIGHT")
    df_loaded = loaded.get_dataframe()
    assert len(df_loaded) == 3
    assert_allclose(df_loaded["X_UTME"].values, [50.0, 60.0, 60.0])
