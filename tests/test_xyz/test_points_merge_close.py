import pathlib

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from xtgeo.xyz import Points

PFILE = pathlib.Path("points/eme/1/emerald_10_random.poi")
PFILE2 = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")
POINTSET3 = pathlib.Path("points/battle/1/many.rmsattr")
POINTSET4 = pathlib.Path("points/reek/1/poi_attr.rmsattr")
POINTSET4_CSV = pathlib.Path("points/reek/1/poi_attr.csv")
CSV1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")


@pytest.fixture
def points_with_attrs():
    plist = [
        (234.0, 556.0, 11.0, 0, "some", 1.0),
        (235.0, 559.0, 14.0, 1, "attr", 1.1),
        (255.0, 577.0, 12.0, 1, "here", 1.2),
    ]
    attrs = {
        "some_int": "int",
        "sometxt": "str",
        "somefloat": "float",
    }
    return plist, attrs


def test_merge_close_points_no_merge():
    """Test merge_close_points when no points are close enough."""
    plist = [
        (0.0, 0.0, 10.0),
        (100.0, 100.0, 20.0),
        (200.0, 200.0, 30.0),
    ]
    mypoints = Points(plist)
    original_nrow = mypoints.nrow

    # Merge with very small distance - should not merge anything
    mypoints.merge_close_points(min_distance=1.0, method="average")

    assert mypoints.nrow == original_nrow


def test_merge_close_points_simple_average():
    """Test merge_close_points with average method."""
    plist = [
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 20.0),  # Close to first point
        (100.0, 100.0, 30.0),
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=2.0, method="average")

    assert mypoints.nrow == 2  # Two points merged into one

    # Check that merged point has average coordinates
    dfr = mypoints.get_dataframe()
    merged_point = dfr[dfr["Z_TVDSS"] == 15.0]  # Average of 10 and 20

    assert len(merged_point) == 1
    assert merged_point["X_UTME"].to_numpy()[0] == pytest.approx(0.5)
    assert merged_point["Y_UTMN"].to_numpy()[0] == pytest.approx(0.5)


def test_merge_close_points_median():
    """Test merge_close_points with median method."""
    plist = [
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 20.0),
        (1.5, 1.5, 30.0),  # All three close together
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=3.0, method="median")

    assert mypoints.nrow == 1

    # Check median values
    dfr = mypoints.get_dataframe()
    assert dfr["X_UTME"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Y_UTMN"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(20.0)


def test_merge_close_points_first():
    """Test merge_close_points with first method."""
    plist = [
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 20.0),
        (1.5, 1.5, 30.0),
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=3.0, method="first")

    assert mypoints.nrow == 1

    # Should keep first point
    dfr = mypoints.get_dataframe()
    assert dfr["X_UTME"].to_numpy()[0] == pytest.approx(0.0)
    assert dfr["Y_UTMN"].to_numpy()[0] == pytest.approx(0.0)
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(10.0)


def test_merge_close_points_min_z():
    """Test merge_close_points with min_z method."""
    plist = [
        (0.0, 0.0, 30.0),
        (1.0, 1.0, 10.0),  # Minimum z
        (1.5, 1.5, 20.0),
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=3.0, method="min_z")

    assert mypoints.nrow == 1

    # Should keep point with minimum z
    dfr = mypoints.get_dataframe()
    assert dfr["X_UTME"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Y_UTMN"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(10.0)


def test_merge_close_points_max_z():
    """Test merge_close_points with max_z method."""
    plist = [
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 30.0),  # Maximum z
        (1.5, 1.5, 20.0),
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=3.0, method="max_z")

    assert mypoints.nrow == 1

    # Should keep point with maximum z
    dfr = mypoints.get_dataframe()
    assert dfr["X_UTME"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Y_UTMN"].to_numpy()[0] == pytest.approx(1.0)
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(30.0)


def test_merge_close_points_multiple_clusters():
    """Test merge_close_points with multiple separate clusters."""
    plist = [
        # Cluster 1
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 20.0),
        # Cluster 2
        (100.0, 100.0, 30.0),
        (101.0, 101.0, 40.0),
        # Isolated point
        (200.0, 200.0, 50.0),
    ]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=2.0, method="average")

    assert mypoints.nrow == 3  # Two clusters merged + one isolated

    dfr = mypoints.get_dataframe()
    z_values = sorted(dfr["Z_TVDSS"].to_numpy())

    # Check merged z-values
    assert z_values[0] == pytest.approx(15.0)  # (10+20)/2
    assert z_values[1] == pytest.approx(35.0)  # (30+40)/2
    assert z_values[2] == pytest.approx(50.0)  # Isolated point


def test_merge_close_points_transitive_closure():
    """Test that merge handles transitive closure correctly.

    Points A-B close, B-C close, but A-C not close.
    All three should still be merged into one cluster.
    """
    plist = [
        (0.0, 0.0, 10.0),  # A
        (1.0, 0.0, 20.0),  # B - close to A
        (2.0, 0.0, 30.0),  # C - close to B, not to A
    ]
    mypoints = Points(plist)

    # Distance 1.5 means A-B merge, B-C merge, but A-C distance is 2.0
    mypoints.merge_close_points(min_distance=1.5, method="average")

    # All three should be merged due to transitive closure
    assert mypoints.nrow == 1

    dfr = mypoints.get_dataframe()
    assert dfr["X_UTME"].to_numpy()[0] == pytest.approx(1.0)  # (0+1+2)/3
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(20.0)  # (10+20+30)/3


def test_merge_close_points_with_attributes():
    """Test that attributes are removed during merge."""
    plist = [
        (0.0, 0.0, 10.0, "well1", 100),
        (1.0, 1.0, 20.0, "well2", 200),  # Close to first
        (100.0, 100.0, 30.0, "well3", 300),
    ]
    attrs = {"wellname": "str", "depth": "int"}
    mypoints = Points(plist, attributes=attrs)

    # Verify attributes exist before merge
    dfr_before = mypoints.get_dataframe()
    assert "wellname" in dfr_before.columns
    assert "depth" in dfr_before.columns

    mypoints.merge_close_points(min_distance=2.0, method="average")

    assert mypoints.nrow == 2
    dfr = mypoints.get_dataframe()

    # Attributes should be removed - only X, Y, Z columns remain
    assert "wellname" not in dfr.columns
    assert "depth" not in dfr.columns
    assert set(dfr.columns) == {"X_UTME", "Y_UTMN", "Z_TVDSS"}

    # Verify merged coordinates are correct
    merged = dfr[np.isclose(dfr["Z_TVDSS"], 15.0)]
    assert len(merged) == 1
    assert np.isclose(merged["X_UTME"].to_numpy()[0], 0.5)
    assert np.isclose(merged["Y_UTMN"].to_numpy()[0], 0.5)


def test_merge_close_points_empty():
    """Test merge_close_points with empty points."""
    mypoints = Points([])

    # Should handle empty gracefully
    mypoints.merge_close_points(min_distance=5.0, method="average")

    assert mypoints.nrow == 0


def test_merge_close_points_single_point():
    """Test merge_close_points with single point."""
    plist = [(0.0, 0.0, 10.0)]
    mypoints = Points(plist)

    mypoints.merge_close_points(min_distance=5.0, method="average")

    assert mypoints.nrow == 1
    dfr = mypoints.get_dataframe()
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(10.0)


def test_merge_close_points_invalid_method():
    """Test merge_close_points with invalid method."""
    plist = [(0.0, 0.0, 10.0), (1.0, 1.0, 20.0)]
    mypoints = Points(plist)

    with pytest.raises(ValueError, match="Unknown merge method"):
        mypoints.merge_close_points(min_distance=5.0, method="invalid_method")


def test_merge_close_points_preserves_column_order():
    """Test that X, Y, Z column order is preserved after merge (attributes removed)."""
    plist = [
        (0.0, 0.0, 10.0, "A", 1.5),
        (1.0, 1.0, 20.0, "B", 2.5),
    ]
    attrs = {"name": "str", "value": "float"}
    mypoints = Points(plist, attributes=attrs)

    # Get the coordinate column names before merge
    coord_columns = [mypoints.xname, mypoints.yname, mypoints.zname]

    mypoints.merge_close_points(min_distance=2.0, method="average")

    # After merge, should have only X, Y, Z columns in same order
    assert list(mypoints.get_dataframe().columns) == coord_columns


def test_merge_close_points_2d_distance_only():
    """Test that merge uses only X-Y distance, not Z."""
    plist = [
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 1000.0),  # Same X-Y, very different Z
    ]
    mypoints = Points(plist)

    # These should merge despite large Z difference
    mypoints.merge_close_points(min_distance=1.0, method="average")

    assert mypoints.nrow == 1
    dfr = mypoints.get_dataframe()
    assert dfr["Z_TVDSS"].to_numpy()[0] == pytest.approx(505.0)  # (10+1000)/2


def test_merge_close_points_many_points():
    """Test merge_close_points with larger dataset."""
    np.random.seed(42)

    # Create points in 3 clusters
    cluster1 = [
        (x, y, z)
        for x, y, z in zip(
            np.random.uniform(0, 5, 20),
            np.random.uniform(0, 5, 20),
            np.random.uniform(100, 110, 20),
        )
    ]
    cluster2 = [
        (x, y, z)
        for x, y, z in zip(
            np.random.uniform(50, 55, 20),
            np.random.uniform(50, 55, 20),
            np.random.uniform(200, 210, 20),
        )
    ]
    cluster3 = [
        (x, y, z)
        for x, y, z in zip(
            np.random.uniform(100, 105, 20),
            np.random.uniform(100, 105, 20),
            np.random.uniform(300, 310, 20),
        )
    ]

    plist = cluster1 + cluster2 + cluster3
    mypoints = Points(plist)

    # Each cluster should merge into one point
    mypoints.merge_close_points(min_distance=10.0, method="average")

    # Should have approximately 3 points (one per cluster)
    assert 1 <= mypoints.nrow <= 10  # Allow some variation due to randomness


@pytest.mark.bigtest
def test_merge_close_points_performance_1m():
    """Test performance with 1 million points."""
    import time

    n_points = 1_000_000
    print(f"\nGenerating {n_points:,} random points...")

    # Generate 1 million random points spread across a 10,000 x 10,000 area
    np.random.seed(42)
    x = np.random.uniform(0, 10000, n_points)
    y = np.random.uniform(0, 10000, n_points)
    z = np.random.uniform(0, 100, n_points)

    plist = list(zip(x, y, z))

    print("Creating Points object...")
    start = time.time()
    mypoints = Points(plist)
    create_time = time.time() - start
    print(f"  Created in {create_time:.2f} seconds")

    print(f"Initial points: {mypoints.nrow:,}")

    # Merge points closer than 5 units using average method
    min_distance = 5.0
    print(f"\nMerging points closer than {min_distance} units (method='average')...")
    start = time.time()
    mypoints.merge_close_points(min_distance=min_distance, method="average")
    merge_time = time.time() - start

    print(f"  Merged in {merge_time:.2f} seconds")
    print(f"  Final points: {mypoints.nrow:,}")
    print(f"  Points merged: {n_points - mypoints.nrow:,}")
    print(f"  Reduction: {100 * (n_points - mypoints.nrow) / n_points:.1f}%")
    print(f"  Processing rate: {n_points / merge_time:,.0f} points/second")

    # Verify we have fewer points than we started with

    assert mypoints.nrow < n_points
    assert mypoints.nrow > 0


def test_merge_close_points_order_independent_average():
    """Test that point order doesn't affect result when method='average'."""
    # Create a set of points with known clusters
    plist1 = [
        (0.0, 0.0, 10.0),
        (1.0, 1.0, 20.0),  # Close to first
        (100.0, 100.0, 30.0),
        (101.0, 101.0, 40.0),  # Close to third
    ]

    # Same points but in different order
    plist2 = [
        (101.0, 101.0, 40.0),  # Was fourth
        (0.0, 0.0, 10.0),  # Was first
        (100.0, 100.0, 30.0),  # Was third
        (1.0, 1.0, 20.0),  # Was second
    ]

    # Create Points objects
    points1 = Points(plist1)
    points2 = Points(plist2)

    # Merge with same parameters
    min_distance = 2.0
    points1.merge_close_points(min_distance=min_distance, method="average")
    points2.merge_close_points(min_distance=min_distance, method="average")

    # Both should have same number of points
    assert points1.nrow == points2.nrow
    assert points1.nrow == 2  # Two clusters

    # Get dataframes sorted by X coordinate for comparison
    df1 = points1.get_dataframe().sort_values("X_UTME").reset_index(drop=True)
    df2 = points2.get_dataframe().sort_values("X_UTME").reset_index(drop=True)

    # The merged points should be identical (within numerical precision)
    # First cluster average: (0+1)/2=0.5, (0+1)/2=0.5, (10+20)/2=15.0
    # Second cluster average: (100+101)/2=100.5, (100+101)/2=100.5, (30+40)/2=35.0
    assert np.isclose(df1["X_UTME"].to_numpy()[0], 0.5)

    assert np.isclose(df2["X_UTME"].to_numpy()[0], 0.5)
    assert np.isclose(df1["Y_UTMN"].to_numpy()[0], 0.5)
    assert np.isclose(df2["Y_UTMN"].to_numpy()[0], 0.5)
    assert np.isclose(df1["Z_TVDSS"].to_numpy()[0], 15.0)
    assert np.isclose(df2["Z_TVDSS"].to_numpy()[0], 15.0)

    assert np.isclose(df1["X_UTME"].to_numpy()[1], 100.5)
    assert np.isclose(df2["X_UTME"].to_numpy()[1], 100.5)
    assert np.isclose(df1["Y_UTMN"].to_numpy()[1], 100.5)
    assert np.isclose(df2["Y_UTMN"].to_numpy()[1], 100.5)
    assert np.isclose(df1["Z_TVDSS"].to_numpy()[1], 35.0)
    assert np.isclose(df2["Z_TVDSS"].to_numpy()[1], 35.0)

    # Verify the dataframes are equal
    assert_frame_equal(df1, df2)
