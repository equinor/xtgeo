"""Tests for WellData and WellLog dataclasses (i/o in separate tests)."""

from __future__ import annotations

import numpy as np
import pytest

from xtgeo.io.welldata._well_io import WellData, WellLog


def test_welllog_continuous_creation():
    """Test creating a continuous well log."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    log = WellLog(name="GR", values=values, is_discrete=False)

    assert log.name == "GR"
    assert len(log.values) == 5
    assert not log.is_discrete
    assert log.code_names is None
    np.testing.assert_array_equal(log.values, values)


def test_welllog_discrete_creation():
    """Test creating a discrete well log with code names."""
    values = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    code_names = {1: "SHALE", 2: "SAND", 3: "LIMESTONE"}
    log = WellLog(name="FACIES", values=values, is_discrete=True, code_names=code_names)

    assert log.name == "FACIES"
    assert log.is_discrete
    assert log.code_names == code_names
    np.testing.assert_array_equal(log.values, values)


def test_welllog_with_nan_values():
    """Test well log with undefined (NaN) values."""
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    log = WellLog(name="PHIT", values=values)

    assert log.name == "PHIT"
    assert np.isnan(log.values[1])
    assert np.isnan(log.values[3])
    assert log.values[0] == 1.0


def test_welllog_code_names_validation():
    """Test that code_names validation works."""
    values = np.array([1.0, 2.0, 3.0])

    # This should raise an error - code names with string keys
    with pytest.raises(ValueError, match="code_names keys must be integers"):
        WellLog(
            name="FACIES",
            values=values,
            is_discrete=True,
            code_names={"1": "SHALE", "2": "SAND"},  # String keys - should fail
        )


def test_welllog_immutability():
    """Test that WellLog is immutable."""
    log = WellLog(name="GR", values=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(Exception):  # FrozenInstanceError or similar
        log.name = "NEW_NAME"


def test_welldata_creation():
    """Test creating WellData with survey and logs."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    gr_log = WellLog(name="GR", values=np.random.rand(n))
    phit_log = WellLog(name="PHIT", values=np.random.rand(n))

    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log),
    )

    assert well.name == "TestWell"
    assert well.xpos == 100.0
    assert well.ypos == 200.0
    assert well.zpos == 25.0
    assert well.n_records == n
    assert len(well.logs) == 2
    assert well.log_names == ("GR", "PHIT")


def test_welldata_validation_survey_length_mismatch():
    """Test that WellData validates survey array lengths."""
    with pytest.raises(ValueError, match="Survey arrays must have the same length"):
        WellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=np.array([1.0, 2.0, 3.0]),
            survey_y=np.array([1.0, 2.0]),  # Wrong length
            survey_z=np.array([1.0, 2.0, 3.0]),
        )


def test_welldata_validation_log_length_mismatch():
    """Test that WellData validates log lengths match survey."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    # Create log with wrong length
    wrong_log = WellLog(name="GR", values=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Log 'GR' has 3 values, but survey has 5"):
        WellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            logs=(wrong_log,),
        )


def test_welldata_get_log():
    """Test getting a log by name."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    gr_log = WellLog(name="GR", values=np.random.rand(n))
    phit_log = WellLog(name="PHIT", values=np.random.rand(n))

    well = WellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log),
    )

    # Get existing log
    retrieved_log = well.get_log("GR")
    assert retrieved_log is not None
    assert retrieved_log.name == "GR"

    # Get non-existing log
    missing_log = well.get_log("NONEXISTENT")
    assert missing_log is None


def test_welldata_get_continuous_and_discrete_logs():
    """Test filtering logs by type."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    gr_log = WellLog(name="GR", values=np.random.rand(n), is_discrete=False)
    phit_log = WellLog(name="PHIT", values=np.random.rand(n), is_discrete=False)
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0, 3.0, 2.0]),
        is_discrete=True,
        code_names={1: "SHALE", 2: "SAND", 3: "LIMESTONE"},
    )

    well = WellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log, facies_log),
    )

    continuous = well.get_continuous_logs()
    assert len(continuous) == 2
    assert all(log.name in ["GR", "PHIT"] for log in continuous)

    discrete = well.get_discrete_logs()
    assert len(discrete) == 1
    assert discrete[0].name == "FACIES"


def test_welldata_empty_logs():
    """Test WellData with no logs."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    well = WellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(),  # Empty tuple
    )

    assert well.n_records == n
    assert len(well.logs) == 0
    assert well.log_names == ()
    assert well.get_continuous_logs() == ()
    assert well.get_discrete_logs() == ()


def test_welldata_immutability():
    """Test that WellData is immutable."""
    well = WellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=np.array([1.0, 2.0, 3.0]),
        survey_y=np.array([1.0, 2.0, 3.0]),
        survey_z=np.array([1.0, 2.0, 3.0]),
    )

    # Try to modify - should fail
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        well.name = "NewName"

    with pytest.raises(Exception):
        well.xpos = 999.0
