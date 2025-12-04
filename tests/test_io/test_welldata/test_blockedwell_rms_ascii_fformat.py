import numpy as np
import pytest

from xtgeo.io.welldata._blockedwell_io import BlockedWellData


@pytest.fixture
def sample_rms_ascii_blockedwell_file(tmp_path):
    """Create a sample RMS ASCII blocked well file for testing."""
    rms_content = """1.0
Undefined
OP_1 461809.6 5932990.4
5
I_INDEX UNK lin
J_INDEX UNK lin
K_INDEX UNK lin
Facies DISC 0 Background 1 Channel 2 Crevasse
Poro UNK lin
462688.159424 5934241.512695 1595.049561 58 55 1 1 0.232928
462689.460205 5934243.043579 1596.280640 58 55 2 1 0.238611
462690.760986 5934244.574463 1597.511597 58 55 3 1 0.216984
462692.061646 5934246.105347 1598.742432 58 55 4 1 0.205116
462693.362549 5934247.636230 1599.973389 58 55 5 1 0.224485
462694.663330 5934249.167114 1601.204346 58 55 6 1 0.236330
462695.964233 5934250.698120 1602.435303 58 55 7 1 0.250664
462697.264893 5934252.228882 1603.666382 58 55 8 1 0.264704
462698.565674 5934253.759766 1604.897217 58 55 9 1 0.268860
"""
    rms_file = tmp_path / "test_blockedwell.rms_ascii"
    rms_file.write_text(rms_content)
    return rms_file


@pytest.fixture
def sample_rms_ascii_blockedwell_file_no_rkb(tmp_path):
    """Create a sample RMS ASCII blocked well file without RKB."""
    rms_content = """1.0
Test blocked well
BW_1 100000.0 200000.0
3
I_INDEX UNK lin
J_INDEX UNK lin
K_INDEX UNK lin
100000.0 200000.0 1000.0 10 20 1
100000.5 200000.5 1001.0 10 20 2
100001.0 200001.0 1002.0 10 20 3
100001.5 200001.5 1003.0 10 21 1
100002.0 200002.0 1004.0 10 21 2
"""
    rms_file = tmp_path / "test_blockedwell_no_rkb.rms_ascii"
    rms_file.write_text(rms_content)
    return rms_file


def test_blockedwell_from_file_rms_ascii_format(sample_rms_ascii_blockedwell_file):
    """Test reading BlockedWellData using from_file method with fformat='rms_ascii'."""
    well = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file,
        fformat="rms_ascii",
    )

    # Verify basic properties
    assert well.name == "OP_1"
    assert well.n_records == 9

    # Verify header position
    assert well.xpos == pytest.approx(461809.6)
    assert well.ypos == pytest.approx(5932990.4)
    assert well.zpos == pytest.approx(0.0)

    # Verify survey arrays
    assert len(well.survey_x) == 9
    assert len(well.survey_y) == 9
    assert len(well.survey_z) == 9

    # Check first and last survey points
    assert well.survey_x[0] == pytest.approx(462688.159424)
    assert well.survey_z[-1] == pytest.approx(1604.897217)

    # Verify grid indices
    assert len(well.i_index) == 9
    assert len(well.j_index) == 9
    assert len(well.k_index) == 9

    # Check index values
    np.testing.assert_array_equal(well.i_index, np.full(9, 58.0))
    np.testing.assert_array_equal(well.j_index, np.full(9, 55.0))
    np.testing.assert_array_equal(well.k_index, np.arange(1.0, 10.0))

    # Verify logs (should not include I_INDEX, J_INDEX, K_INDEX)
    assert len(well.logs) == 2
    assert "Facies" in well.log_names
    assert "Poro" in well.log_names
    assert "I_INDEX" not in well.log_names
    assert "J_INDEX" not in well.log_names
    assert "K_INDEX" not in well.log_names

    # Check log types
    facies = well.get_log("Facies")
    assert facies.is_discrete
    poro = well.get_log("Poro")
    assert not poro.is_discrete


def test_blockedwell_roundtrip_rms_ascii(sample_rms_ascii_blockedwell_file, tmp_path):
    """Test reading and writing BlockedWellData in RMS ASCII format."""
    # Read original
    well1 = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file,
        fformat="rms_ascii",
    )

    # Write to new file
    output_file = tmp_path / "roundtrip_blocked.rms_ascii"
    well1.to_file(filepath=output_file, fformat="rms_ascii")

    # Read back
    well2 = BlockedWellData.from_file(
        filepath=output_file,
        fformat="rms_ascii",
    )

    # Verify data matches
    assert well1.name == well2.name
    assert well1.n_records == well2.n_records
    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x, decimal=4)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y, decimal=4)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z, decimal=4)

    # Check indices
    np.testing.assert_array_almost_equal(well1.i_index, well2.i_index, decimal=4)
    np.testing.assert_array_almost_equal(well1.j_index, well2.j_index, decimal=4)
    np.testing.assert_array_almost_equal(well1.k_index, well2.k_index, decimal=4)

    # Check logs
    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        assert log1.is_discrete == log2.is_discrete
        np.testing.assert_array_almost_equal(log1.values, log2.values, decimal=4)


def test_blockedwell_rms_ascii_no_rkb(sample_rms_ascii_blockedwell_file_no_rkb):
    """Test reading RMS ASCII blocked well file without RKB in header."""
    well = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file_no_rkb,
        fformat="rms_ascii",
    )

    # Verify basic properties
    assert well.name == "BW_1"
    assert well.n_records == 5

    # Verify header position (zpos should default to 0.0)
    assert well.xpos == pytest.approx(100000.0)
    assert well.ypos == pytest.approx(200000.0)
    assert well.zpos == pytest.approx(0.0)

    # Verify indices
    expected_i = np.full(5, 10.0)
    expected_j = np.array([20.0, 20.0, 20.0, 21.0, 21.0])
    expected_k = np.array([1.0, 2.0, 3.0, 1.0, 2.0])

    np.testing.assert_array_equal(well.i_index, expected_i)
    np.testing.assert_array_equal(well.j_index, expected_j)
    np.testing.assert_array_equal(well.k_index, expected_k)

    # No other logs in this file
    assert len(well.logs) == 0


def test_blockedwell_rms_ascii_log_values(sample_rms_ascii_blockedwell_file):
    """Test that log values are read correctly from RMS ASCII blocked well."""
    well = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file,
        fformat="rms_ascii",
    )

    facies = well.get_log("Facies")
    poro = well.get_log("Poro")

    # Check facies values (all are 1 in the sample)
    np.testing.assert_array_equal(facies.values, np.ones(9))

    # Check poro values
    expected_poro = [
        0.232928,
        0.238611,
        0.216984,
        0.205116,
        0.224485,
        0.236330,
        0.250664,
        0.264704,
        0.268860,
    ]
    np.testing.assert_array_almost_equal(poro.values, expected_poro)


def test_blockedwell_rms_ascii_n_blocked_cells(sample_rms_ascii_blockedwell_file):
    """Test that n_blocked_cells property works correctly."""
    well = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file,
        fformat="rms_ascii",
    )

    # All 9 records should have valid indices
    assert well.n_blocked_cells == 9


def test_blockedwell_to_file_rms_ascii_format(tmp_path):
    """Test writing BlockedWellData to RMS ASCII format."""
    # Create a simple blocked well
    survey_x = np.array([100.0, 101.0, 102.0, 103.0])
    survey_y = np.array([200.0, 201.0, 202.0, 203.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    i_index = np.array([10.0, 10.0, 11.0, 11.0])
    j_index = np.array([20.0, 20.0, 20.0, 20.0])
    k_index = np.array([1.0, 2.0, 1.0, 2.0])

    from xtgeo.io.welldata._well_io import WellLog

    poro_values = np.array([0.25, 0.30, 0.28, 0.26])
    poro_log = WellLog(name="Poro", values=poro_values, is_discrete=False)

    well = BlockedWellData(
        name="TEST_BW",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
        logs=(poro_log,),
    )

    # Write to file
    output_file = tmp_path / "test_write.rms_ascii"
    well.to_file(filepath=output_file, fformat="rms_ascii")

    # Verify file exists
    assert output_file.exists()

    # Read back and verify
    well2 = BlockedWellData.from_file(filepath=output_file, fformat="rms_ascii")

    assert well2.name == "TEST_BW"
    assert well2.n_records == 4
    np.testing.assert_array_almost_equal(well2.i_index, i_index)
    np.testing.assert_array_almost_equal(well2.j_index, j_index)
    np.testing.assert_array_almost_equal(well2.k_index, k_index)

    poro2 = well2.get_log("Poro")
    np.testing.assert_array_almost_equal(poro2.values, poro_values)


def test_blockedwell_rms_ascii_discrete_codes(sample_rms_ascii_blockedwell_file):
    """Test that discrete log codes are properly parsed in blocked wells."""
    well = BlockedWellData.from_file(
        filepath=sample_rms_ascii_blockedwell_file,
        fformat="rms_ascii",
    )

    facies = well.get_log("Facies")
    assert facies.is_discrete

    # Check that code mapping exists
    assert hasattr(facies, "code_names")
    if facies.code_names:
        assert 0 in facies.code_names
        assert 1 in facies.code_names
        assert 2 in facies.code_names
