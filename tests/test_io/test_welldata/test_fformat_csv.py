"""Tests for CSV format I/O for WellData and BlockedWellData."""

from __future__ import annotations

import io

import numpy as np
import pytest

from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellFileFormat, WellLog


@pytest.fixture
def sample_csv_well_file(tmp_path):
    """Create a sample CSV file for testing WellData I/O."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,PHIT,PERM
464789.0625,6553551.625,1620.5,0.326,125.5
464789.0625,6553551.625,1621.5,0.316,98.2
464789.0625,6553551.625,1622.5,0.318,105.7
464789.0625,6553551.625,1623.5,0.315,92.3
464789.0625,6553551.625,1624.5,0.307,85.1
"""
    csv_file = tmp_path / "test_well.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_csv_blockedwell_file(tmp_path):
    """Create a sample CSV file for testing BlockedWellData I/O."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,PHIT,PERM
464789.0625,6553551.625,1620.5,109,115,0,0.326,125.5
464789.0625,6553551.625,1621.5,109,115,1,0.316,98.2
464789.0625,6553551.625,1622.5,109,115,2,0.318,105.7
464789.0625,6553551.625,1623.5,109,115,3,0.315,92.3
464789.0625,6553551.625,1624.5,109,115,4,0.307,85.1
"""
    csv_file = tmp_path / "test_blockedwell.csv"
    csv_file.write_text(csv_content)
    return csv_file


def test_welldata_from_file_csv_format(sample_csv_well_file):
    """Test reading WellData using from_file method with fformat='csv'."""
    well = WellData.from_file(
        filepath=sample_csv_well_file,
        fformat=WellFileFormat.CSV,
    )

    assert well.n_records == 5

    assert well.xpos == pytest.approx(464789.0625)
    assert well.ypos == pytest.approx(6553551.625)
    assert well.zpos == pytest.approx(1620.5)

    assert len(well.survey_x) == 5
    assert len(well.survey_y) == 5
    assert len(well.survey_z) == 5

    assert np.all(well.survey_x == 464789.0625)
    assert np.all(well.survey_y == 6553551.625)

    expected_z = np.array([1620.5, 1621.5, 1622.5, 1623.5, 1624.5])
    np.testing.assert_array_almost_equal(well.survey_z, expected_z)

    assert len(well.logs) == 2
    assert "PHIT" in well.log_names
    assert "PERM" in well.log_names

    phit = well.get_log("PHIT")
    assert phit is not None
    assert phit.name == "PHIT"
    assert not phit.is_discrete
    assert len(phit.values) == 5
    expected_phit = np.array([0.326, 0.316, 0.318, 0.315, 0.307])
    np.testing.assert_array_almost_equal(phit.values, expected_phit)

    perm = well.get_log("PERM")
    assert perm is not None
    assert perm.name == "PERM"
    assert not perm.is_discrete
    assert len(perm.values) == 5
    expected_perm = np.array([125.5, 98.2, 105.7, 92.3, 85.1])
    np.testing.assert_array_almost_equal(perm.values, expected_perm)


def test_blockedwell_from_file_csv_format(sample_csv_blockedwell_file):
    """Test reading BlockedWellData using from_file method with fformat='csv'."""
    well = BlockedWellData.from_file(
        filepath=sample_csv_blockedwell_file,
        fformat=WellFileFormat.CSV,
    )

    assert well.n_records == 5
    assert well.n_blocked_cells == 5

    assert well.xpos == pytest.approx(464789.0625)
    assert well.ypos == pytest.approx(6553551.625)
    assert well.zpos == pytest.approx(1620.5)

    assert len(well.survey_x) == 5
    assert len(well.survey_y) == 5
    assert len(well.survey_z) == 5

    assert len(well.i_index) == 5
    assert len(well.j_index) == 5
    assert len(well.k_index) == 5

    assert np.all(well.i_index == 109)
    assert np.all(well.j_index == 115)

    expected_k = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_array_equal(well.k_index, expected_k)

    assert len(well.logs) == 2
    assert "PHIT" in well.log_names
    assert "PERM" in well.log_names

    phit = well.get_log("PHIT")
    assert phit is not None
    expected_phit = np.array([0.326, 0.316, 0.318, 0.315, 0.307])
    np.testing.assert_array_almost_equal(phit.values, expected_phit)


def test_welldata_roundtrip_csv(sample_csv_well_file, tmp_path):
    """Test that reading and writing WellData preserves data."""
    well1 = WellData.from_file(
        filepath=sample_csv_well_file,
        fformat=WellFileFormat.CSV,
    )

    output_file = tmp_path / "roundtrip.csv"
    well1.to_file(filepath=output_file, fformat=WellFileFormat.CSV)

    well2 = WellData.from_file(
        filepath=output_file,
        fformat=WellFileFormat.CSV,
    )

    assert well1.n_records == well2.n_records
    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z)

    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        np.testing.assert_array_almost_equal(log1.values, log2.values)


def test_blockedwell_roundtrip_csv(sample_csv_blockedwell_file, tmp_path):
    """Test that reading and writing BlockedWellData preserves data."""
    well1 = BlockedWellData.from_file(
        filepath=sample_csv_blockedwell_file,
        fformat=WellFileFormat.CSV,
    )

    output_file = tmp_path / "roundtrip_blocked.csv"
    well1.to_file(filepath=output_file, fformat=WellFileFormat.CSV)

    well2 = BlockedWellData.from_file(
        filepath=output_file,
        fformat=WellFileFormat.CSV,
    )

    assert well1.n_records == well2.n_records
    assert well1.n_blocked_cells == well2.n_blocked_cells

    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z)

    np.testing.assert_array_almost_equal(well1.i_index, well2.i_index)
    np.testing.assert_array_almost_equal(well1.j_index, well2.j_index)
    np.testing.assert_array_almost_equal(well1.k_index, well2.k_index)

    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        np.testing.assert_array_almost_equal(log1.values, log2.values)


def test_welldata_from_csv_custom_columns(tmp_path):
    """Test reading WellData with custom column names."""
    csv_content = """EASTING,NORTHING,DEPTH,POR,K
100.0,200.0,1000.0,0.25,150.0
100.0,200.0,1001.0,0.30,200.0
100.0,200.0,1002.0,0.28,175.0
"""
    csv_file = tmp_path / "custom_columns.csv"
    csv_file.write_text(csv_content)

    well = WellData.from_csv(
        filepath=csv_file,
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
    )

    assert well.n_records == 3
    assert well.xpos == 100.0
    assert well.ypos == 200.0
    assert well.zpos == 1000.0

    assert len(well.logs) == 2
    assert "POR" in well.log_names
    assert "K" in well.log_names


def test_blockedwell_from_csv_custom_columns(tmp_path):
    """Test reading BlockedWellData with custom column names."""
    csv_content = """EASTING,NORTHING,DEPTH,I,J,K,POR
100.0,200.0,1000.0,10,20,1,0.25
100.0,200.0,1001.0,10,20,2,0.30
100.0,200.0,1002.0,10,20,3,0.28
"""
    csv_file = tmp_path / "custom_blocked.csv"
    csv_file.write_text(csv_content)

    well = BlockedWellData.from_csv(
        filepath=csv_file,
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
        i_indexname="I",
        j_indexname="J",
        k_indexname="K",
    )

    assert well.n_records == 3
    assert well.n_blocked_cells == 3

    assert np.all(well.i_index == 10)
    assert np.all(well.j_index == 20)
    expected_k = np.array([1, 2, 3], dtype=np.float64)
    np.testing.assert_array_equal(well.k_index, expected_k)


def test_welldata_to_file_csv_format(tmp_path):
    """Test writing WellData with to_file method."""
    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(),
    )

    output_file = tmp_path / "output.csv"
    well.to_file(filepath=output_file, fformat=WellFileFormat.CSV)

    assert output_file.exists()

    well2 = WellData.from_file(
        filepath=output_file,
        fformat=WellFileFormat.CSV,
    )

    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.survey_z, well.survey_z)


def test_blockedwell_to_file_csv_format(tmp_path):
    """Test writing BlockedWellData with to_file method."""
    well = BlockedWellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        i_index=np.array([10.0, 10.0, 11.0]),
        j_index=np.array([20.0, 20.0, 21.0]),
        k_index=np.array([1.0, 2.0, 3.0]),
        logs=(),
    )

    output_file = tmp_path / "output_blocked.csv"
    well.to_file(filepath=output_file, fformat=WellFileFormat.CSV)

    assert output_file.exists()

    well2 = BlockedWellData.from_file(
        filepath=output_file,
        fformat=WellFileFormat.CSV,
    )

    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.i_index, well.i_index)
    np.testing.assert_array_almost_equal(well2.k_index, well.k_index)


def test_welldata_csv_from_stringio():
    """Test reading CSV WellData from StringIO stream."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,PHIT,PERM
100.0,200.0,1000.0,0.25,150.0
101.0,201.0,1001.0,0.30,200.0
102.0,202.0,1002.0,0.28,175.0
"""
    stream = io.StringIO(csv_content)
    well = WellData.from_csv(filepath=stream)

    assert well.n_records == 3
    assert well.xpos == pytest.approx(100.0)

    phit = well.get_log("PHIT")
    np.testing.assert_array_almost_equal(phit.values, [0.25, 0.30, 0.28])


def test_welldata_csv_to_stringio():
    """Test writing CSV WellData to StringIO stream."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    poro_log = WellLog(
        name="Poro", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = WellData(
        name="STREAM_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(poro_log,),
    )

    stream = io.StringIO()
    well.to_csv(filepath=stream)

    stream.seek(0)
    well2 = WellData.from_csv(filepath=stream)

    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.survey_x, survey_x)


def test_blockedwell_csv_from_stringio():
    """Test reading CSV BlockedWellData from StringIO stream."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,PHIT
100.0,200.0,1000.0,10,20,1,0.25
101.0,201.0,1001.0,10,20,2,0.30
102.0,202.0,1002.0,10,20,3,0.28
"""
    stream = io.StringIO(csv_content)
    well = BlockedWellData.from_csv(filepath=stream)

    assert well.n_records == 3
    np.testing.assert_array_equal(well.i_index, [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(well.k_index, [1.0, 2.0, 3.0])


def test_blockedwell_csv_to_stringio():
    """Test writing CSV BlockedWellData to StringIO stream."""
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])
    i_index = np.array([10.0, 10.0, 10.0])
    j_index = np.array([20.0, 20.0, 20.0])
    k_index = np.array([1.0, 2.0, 3.0])

    poro_log = WellLog(
        name="Poro", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = BlockedWellData(
        name="BW_STREAM",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
        logs=(poro_log,),
    )

    stream = io.StringIO()
    well.to_csv(filepath=stream)

    stream.seek(0)
    well2 = BlockedWellData.from_csv(filepath=stream)

    assert well2.n_records == 3
    np.testing.assert_array_equal(well2.i_index, i_index)
    np.testing.assert_array_equal(well2.k_index, k_index)


def test_welldata_csv_missing_required_columns(tmp_path):
    """Test that reading CSV with missing required columns raises ValueError."""
    csv_content = """X_UTME,PHIT
100.0,0.25
101.0,0.26
"""
    csv_file = tmp_path / "missing_cols.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="Missing required columns"):
        WellData.from_csv(csv_file)


def test_blockedwell_csv_missing_required_columns(tmp_path):
    """Test that reading blocked well CSV with missing columns raises ValueError."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,PHIT
100.0,200.0,1000.0,0.25
101.0,201.0,1001.0,0.26
"""
    csv_file = tmp_path / "missing_blocked_cols.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="Missing required columns"):
        BlockedWellData.from_csv(csv_file)


def test_welldata_with_discrete_log(tmp_path):
    """Test that integer-valued logs are detected as discrete."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,FACIES,PHIT
100.0,200.0,1000.0,1,0.25
100.0,200.0,1001.0,2,0.30
100.0,200.0,1002.0,1,0.28
"""
    csv_file = tmp_path / "with_facies.csv"
    csv_file.write_text(csv_content)

    well = WellData.from_csv(csv_file)

    facies = well.get_log("FACIES")
    assert facies is not None
    assert facies.is_discrete

    phit = well.get_log("PHIT")
    assert phit is not None
    assert not phit.is_discrete


def test_welldata_csv_write_with_logs(tmp_path):
    """Test writing WellData with multiple logs."""
    survey_x = np.array([100.0, 100.0, 100.0])
    survey_y = np.array([200.0, 200.0, 200.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    phit_log = WellLog(
        name="PHIT", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )
    perm_log = WellLog(
        name="PERM", values=np.array([100.0, 150.0, 125.0]), is_discrete=False
    )

    well = WellData(
        name="MULTI_LOG_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(phit_log, perm_log),
    )

    output_file = tmp_path / "multi_log.csv"
    well.to_csv(filepath=output_file)

    well2 = WellData.from_csv(output_file)
    assert len(well2.logs) == 2
    assert "PHIT" in well2.log_names
    assert "PERM" in well2.log_names

    phit2 = well2.get_log("PHIT")
    np.testing.assert_array_almost_equal(phit2.values, phit_log.values)


def test_blockedwell_csv_write_with_logs(tmp_path):
    """Test writing BlockedWellData with multiple logs."""
    survey_x = np.array([100.0, 100.0, 100.0])
    survey_y = np.array([200.0, 200.0, 200.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])
    i_index = np.array([10.0, 10.0, 10.0])
    j_index = np.array([20.0, 20.0, 20.0])
    k_index = np.array([1.0, 2.0, 3.0])

    phit_log = WellLog(
        name="PHIT", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = BlockedWellData(
        name="BLOCKED_MULTI_LOG",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
        logs=(phit_log,),
    )

    output_file = tmp_path / "blocked_multi_log.csv"
    well.to_csv(filepath=output_file)

    well2 = BlockedWellData.from_csv(output_file)
    assert len(well2.logs) == 1
    assert "PHIT" in well2.log_names
    np.testing.assert_array_equal(well2.i_index, i_index)


def test_welldata_multiwell_csv_filter_by_wellname(tmp_path):
    """Test reading specific well from multi-well CSV file."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
WELL-1,100.0,200.0,1000.0,0.25
WELL-1,100.0,200.0,1001.0,0.26
WELL-2,150.0,250.0,1500.0,0.30
WELL-2,150.0,250.0,1501.0,0.32
WELL-3,200.0,300.0,2000.0,0.28
"""
    csv_file = tmp_path / "multiwell.csv"
    csv_file.write_text(csv_content)

    well1 = WellData.from_csv(csv_file, wellname="WELL-1")
    assert well1.name == "WELL-1"
    assert well1.n_records == 2
    assert well1.xpos == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(well1.survey_z, [1000.0, 1001.0])

    well2 = WellData.from_csv(csv_file, wellname="WELL-2")
    assert well2.name == "WELL-2"
    assert well2.n_records == 2
    assert well2.xpos == pytest.approx(150.0)
    np.testing.assert_array_almost_equal(well2.survey_z, [1500.0, 1501.0])

    well3 = WellData.from_csv(csv_file, wellname="WELL-3")
    assert well3.name == "WELL-3"
    assert well3.n_records == 1
    assert well3.xpos == pytest.approx(200.0)


def test_welldata_multiwell_csv_auto_select_first(tmp_path):
    """Test that first well is selected when wellname is None."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
WELL-2,150.0,250.0,1500.0,0.30
WELL-2,150.0,250.0,1501.0,0.32
WELL-1,100.0,200.0,1000.0,0.25
WELL-1,100.0,200.0,1001.0,0.26
"""
    csv_file = tmp_path / "multiwell_order.csv"
    csv_file.write_text(csv_content)

    well = WellData.from_csv(csv_file, wellname=None)
    assert well.name == "WELL-2"
    assert well.n_records == 2


def test_welldata_multiwell_csv_nonexistent_well(tmp_path):
    """Test error when requesting non-existent well."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
WELL-1,100.0,200.0,1000.0,0.25
WELL-2,150.0,250.0,1500.0,0.30
"""
    csv_file = tmp_path / "multiwell.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="Well 'WELL-999' not found"):
        WellData.from_csv(csv_file, wellname="WELL-999")


def test_blockedwell_multiwell_csv_filter(tmp_path):
    """Test reading specific blocked well from multi-well CSV file."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,PHIT
WELL-A,100.0,200.0,1000.0,10,20,1,0.25
WELL-A,100.0,200.0,1001.0,10,20,2,0.26
WELL-B,150.0,250.0,1500.0,15,25,1,0.30
WELL-B,150.0,250.0,1501.0,15,25,2,0.32
"""
    csv_file = tmp_path / "multiwell_blocked.csv"
    csv_file.write_text(csv_content)

    well_a = BlockedWellData.from_csv(csv_file, wellname="WELL-A")
    assert well_a.name == "WELL-A"
    assert well_a.n_records == 2
    np.testing.assert_array_equal(well_a.i_index, [10.0, 10.0])
    np.testing.assert_array_equal(well_a.k_index, [1.0, 2.0])

    well_b = BlockedWellData.from_csv(csv_file, wellname="WELL-B")
    assert well_b.name == "WELL-B"
    assert well_b.n_records == 2
    np.testing.assert_array_equal(well_b.i_index, [15.0, 15.0])
    np.testing.assert_array_equal(well_b.k_index, [1.0, 2.0])


def test_welldata_singlewell_csv_no_wellname_column(tmp_path):
    """Test reading single-well CSV without WELLNAME column (backward compatibility)."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,PHIT
100.0,200.0,1000.0,0.25
100.0,200.0,1001.0,0.26
100.0,200.0,1002.0,0.28
"""
    csv_file = tmp_path / "singlewell_no_name.csv"
    csv_file.write_text(csv_content)

    well = WellData.from_csv(csv_file)
    assert well.n_records == 3
    assert "singlewell_no_name" in well.name or well.name == "UNKNOWN"
