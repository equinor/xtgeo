"""Tests for WellData.to_dataframe() method."""

import numpy as np
import pandas as pd
import pytest

from xtgeo.io.welldata._well_io import WellData, WellLog


def test_welldata_to_dataframe_all_logs():
    """Test converting WellData to DataFrame with all logs."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(name="GR", values=np.array([50.0, 60.0, 70.0]), is_discrete=False)
    phit_log = WellLog(
        name="PHIT", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log),
    )

    df = well.to_dataframe()

    # Check DataFrame shape
    assert len(df) == 3
    assert len(df.columns) == 5  # X, Y, Z + 2 logs

    # Check column names
    assert "X_UTME" in df.columns
    assert "Y_UTMN" in df.columns
    assert "Z_TVDSS" in df.columns
    assert "GR" in df.columns
    assert "PHIT" in df.columns

    # Check values
    np.testing.assert_array_equal(df["X_UTME"].values, survey_x)
    np.testing.assert_array_equal(df["Y_UTMN"].values, survey_y)
    np.testing.assert_array_equal(df["Z_TVDSS"].values, survey_z)
    np.testing.assert_array_equal(df["GR"].values, gr_log.values)
    np.testing.assert_array_equal(df["PHIT"].values, phit_log.values)


def test_welldata_to_dataframe_subset_logs():
    """Test converting WellData to DataFrame with subset of logs."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(name="GR", values=np.array([50.0, 60.0, 70.0]), is_discrete=False)
    phit_log = WellLog(
        name="PHIT", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )
    perm_log = WellLog(
        name="PERM", values=np.array([150.0, 200.0, 175.0]), is_discrete=False
    )

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log, perm_log),
    )

    # Request only specific logs
    df = well.to_dataframe(lognames=["GR", "PERM"])

    # Check DataFrame shape
    assert len(df) == 3
    assert len(df.columns) == 5  # X, Y, Z + 2 selected logs

    # Check column names
    assert "X_UTME" in df.columns
    assert "Y_UTMN" in df.columns
    assert "Z_TVDSS" in df.columns
    assert "GR" in df.columns
    assert "PERM" in df.columns
    assert "PHIT" not in df.columns  # Should not be included

    # Check values
    np.testing.assert_array_equal(df["GR"].values, gr_log.values)
    np.testing.assert_array_equal(df["PERM"].values, perm_log.values)


def test_welldata_to_dataframe_custom_column_names():
    """Test converting WellData to DataFrame with custom column names."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(name="GR", values=np.array([50.0, 60.0, 70.0]), is_discrete=False)

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log,),
    )

    df = well.to_dataframe(xname="EASTING", yname="NORTHING", zname="DEPTH")

    # Check custom column names
    assert "EASTING" in df.columns
    assert "NORTHING" in df.columns
    assert "DEPTH" in df.columns
    assert "X_UTME" not in df.columns
    assert "Y_UTMN" not in df.columns
    assert "Z_TVDSS" not in df.columns

    # Check values
    np.testing.assert_array_equal(df["EASTING"].values, survey_x)
    np.testing.assert_array_equal(df["NORTHING"].values, survey_y)
    np.testing.assert_array_equal(df["DEPTH"].values, survey_z)


def test_welldata_to_dataframe_no_logs():
    """Test converting WellData to DataFrame with no logs."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(),
    )

    df = well.to_dataframe()

    # Check DataFrame shape - only survey columns
    assert len(df) == 3
    assert len(df.columns) == 3  # Only X, Y, Z

    # Check column names
    assert "X_UTME" in df.columns
    assert "Y_UTMN" in df.columns
    assert "Z_TVDSS" in df.columns


def test_welldata_to_dataframe_invalid_logname():
    """Test that requesting non-existent log raises ValueError."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(name="GR", values=np.array([50.0, 60.0, 70.0]), is_discrete=False)

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log,),
    )

    with pytest.raises(ValueError, match="Log 'NONEXISTENT' not found"):
        well.to_dataframe(lognames=["GR", "NONEXISTENT"])


def test_welldata_to_dataframe_with_nan_values():
    """Test converting WellData to DataFrame with NaN values in logs."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(
        name="GR", values=np.array([50.0, np.nan, 70.0]), is_discrete=False
    )

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log,),
    )

    df = well.to_dataframe()

    # Check NaN values are preserved
    assert df["GR"].iloc[0] == pytest.approx(50.0)
    assert pd.isna(df["GR"].iloc[1])
    assert df["GR"].iloc[2] == pytest.approx(70.0)


def test_welldata_to_dataframe_discrete_logs():
    """Test converting WellData to DataFrame with discrete logs."""
    survey_x = np.array([100.0, 101.0, 102.0, 103.0])
    survey_y = np.array([200.0, 201.0, 202.0, 203.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0, 1003.0])

    facies_log = WellLog(
        name="FACIES",
        values=np.array([0.0, 1.0, 1.0, 2.0]),
        is_discrete=True,
        code_names={0: "Shale", 1: "Sand", 2: "Limestone"},
    )

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(facies_log,),
    )

    df = well.to_dataframe()

    # Check discrete log values
    assert len(df) == 4
    np.testing.assert_array_equal(df["FACIES"].values, [0.0, 1.0, 1.0, 2.0])


def test_welldata_to_dataframe_empty_lognames_list():
    """Test converting WellData to DataFrame with empty lognames list."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    gr_log = WellLog(name="GR", values=np.array([50.0, 60.0, 70.0]), is_discrete=False)

    well = WellData(
        name="TEST_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log,),
    )

    # Empty list should result in only survey columns
    df = well.to_dataframe(lognames=[])

    assert len(df) == 3
    assert len(df.columns) == 3  # Only X, Y, Z
    assert "GR" not in df.columns
