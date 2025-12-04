import io

import numpy as np
import pytest

from xtgeo.io.welldata._well_io import WellData


@pytest.fixture
def sample_rms_ascii_well_file(tmp_path):
    """Create a sample RMS ASCII file for testing WellData i/o (with RKB)."""
    rms_content = """1.0
OIL - DRILLED
OP_1 461809.590 5932990.360 0.0000
4
Zonelog DISC 0 Above_TUR 1 Below_TUR 2 Below_TMR 3 Below_TMR 4 Below_BLR
Perm UNK lin
Poro UNK lin
Facies DISC 0 Background 1 Channel 2 Crevasse
461809.590 5932990.360 0.0000 0.00000000 -999.00000000 -999.00000000 -999.00000000
461809.722 5932990.544 0.4857 0.00000000 -999.00000000 -999.00000000 -999.00000000
461809.854 5932990.728 0.9714 0.00000000 150.00000000 0.25000000 -999.00000000
461809.986 5932990.912 1.4571 1.00000000 200.00000000 0.30000000 1.00000000
461810.118 5932991.096 1.9428 1.00000000 175.00000000 0.28000000 2.00000000
461810.250 5932991.280 2.4285 2.00000000 160.00000000 0.26000000 1.00000000
461810.382 5932991.464 2.9143 2.00000000 180.00000000 0.29000000 0.00000000
461810.515 5932991.648 3.4000 3.00000000 -999.00000000 -999.00000000 -999.00000000
461810.647 5932991.831 3.8857 3.00000000 -999.00000000 -999.00000000 -999.00000000
461810.779 5932992.015 4.3714 4.00000000 -999.00000000 -999.00000000 -999.00000000
"""
    rms_file = tmp_path / "test_well.rms_ascii"
    rms_file.write_text(rms_content)
    return rms_file


@pytest.fixture
def sample_rms_ascii_well_file_no_rkb(tmp_path):
    """Create a sample RMS ASCII file without RKB (only name, x, y in header)."""
    rms_content = """1.0
GAS - DRILLED
OP_2 461900.000 5933100.000
3
Perm UNK lin
Poro UNK lin
Facies DISC 0 Shale 1 Sand
461900.000 5933100.000 1000.0000 -999.00000000 -999.00000000 -999.00000000
461900.100 5933100.100 1001.5000 150.00000000 0.25000000 0.00000000
461900.200 5933100.200 1003.0000 200.00000000 0.30000000 1.00000000
461900.300 5933100.300 1004.5000 175.00000000 0.28000000 1.00000000
461900.400 5933100.400 1006.0000 -999.00000000 -999.00000000 -999.00000000
"""
    rms_file = tmp_path / "test_well_no_rkb.rms_ascii"
    rms_file.write_text(rms_content)
    return rms_file


def test_welldata_from_file_rms_ascii_format(sample_rms_ascii_well_file):
    """Test reading WellData using from_file method with fformat='rms_ascii'."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )

    # Verify basic properties
    assert well.name == "OP_1"
    assert well.n_records == 10

    # Verify header position
    assert well.xpos == pytest.approx(461809.590)
    assert well.ypos == pytest.approx(5932990.360)
    assert well.zpos == pytest.approx(0.0)

    # Verify survey arrays
    assert len(well.survey_x) == 10
    assert len(well.survey_y) == 10
    assert len(well.survey_z) == 10

    # Check first and last survey points
    assert well.survey_x[0] == pytest.approx(461809.590)
    assert well.survey_z[-1] == pytest.approx(4.3714)

    # Verify logs
    assert len(well.logs) == 4
    assert "Zonelog" in well.log_names
    assert "Perm" in well.log_names
    assert "Poro" in well.log_names
    assert "Facies" in well.log_names

    # Check discrete logs
    zonelog = well.get_log("Zonelog")
    assert zonelog.is_discrete
    facies = well.get_log("Facies")
    assert facies.is_discrete

    # Check continuous logs
    perm = well.get_log("Perm")
    assert not perm.is_discrete
    poro = well.get_log("Poro")
    assert not poro.is_discrete


def test_welldata_roundtrip_rms_ascii(sample_rms_ascii_well_file, tmp_path):
    """Test that reading and writing WellData in RMS ASCII format preserves data."""
    # Read original
    well1 = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )

    # Write to new file
    output_file = tmp_path / "roundtrip.rms_ascii"
    well1.to_file(filepath=output_file, fformat="rms_ascii")

    # Read back
    well2 = WellData.from_file(
        filepath=output_file,
        fformat="rms_ascii",
    )

    # Verify data matches
    assert well1.name == well2.name
    assert well1.n_records == well2.n_records
    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z)

    # Check logs
    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        assert log1.is_discrete == log2.is_discrete
        np.testing.assert_array_almost_equal(log1.values, log2.values)


def test_welldata_from_file_rms_ascii_uppercase(sample_rms_ascii_well_file):
    """Test that fformat='rms_ascii' works (case-insensitive)."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )
    assert well.name == "OP_1"
    assert well.n_records == 10


def test_welldata_rms_ascii_no_rkb(sample_rms_ascii_well_file_no_rkb):
    """Test reading RMS ASCII file without RKB in header (only name, x, y)."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file_no_rkb,
        fformat="rms_ascii",
    )

    # Verify basic properties
    assert well.name == "OP_2"
    assert well.n_records == 5

    # Verify header position (zpos should default to 0.0)
    assert well.xpos == pytest.approx(461900.000)
    assert well.ypos == pytest.approx(5933100.000)
    assert well.zpos == pytest.approx(0.0)

    # Verify logs
    assert len(well.logs) == 3
    assert "Perm" in well.log_names
    assert "Poro" in well.log_names
    assert "Facies" in well.log_names

    # Check some values
    perm = well.get_log("Perm")
    assert np.isnan(perm.values[0])
    assert perm.values[1] == pytest.approx(150.0)
    assert perm.values[2] == pytest.approx(200.0)

    # Check Z coordinates are read correctly
    assert well.survey_z[0] == pytest.approx(1000.0)
    assert well.survey_z[-1] == pytest.approx(1006.0)


def test_welldata_rms_ascii_undef_to_nan(sample_rms_ascii_well_file):
    """Test that -999 values are converted to NaN."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )

    # Check that -999 values are converted to NaN
    perm = well.get_log("Perm")
    poro = well.get_log("Poro")
    facies = well.get_log("Facies")

    # First two records have -999 for all logs
    assert np.isnan(perm.values[0])
    assert np.isnan(perm.values[1])
    assert np.isnan(poro.values[0])
    assert np.isnan(poro.values[1])
    assert np.isnan(facies.values[0])
    assert np.isnan(facies.values[1])

    # Third record has valid Perm and Poro, but -999 for Facies
    assert perm.values[2] == pytest.approx(150.0)
    assert poro.values[2] == pytest.approx(0.25)
    assert np.isnan(facies.values[2])

    # Last three records have -999 for all logs
    assert np.isnan(perm.values[-1])
    assert np.isnan(poro.values[-1])
    assert np.isnan(facies.values[-1])


def test_welldata_rms_ascii_discrete_log_codes(sample_rms_ascii_well_file):
    """Test that discrete log codes are properly parsed."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )

    zonelog = well.get_log("Zonelog")
    assert zonelog.is_discrete

    # Check that code mapping exists
    assert hasattr(zonelog, "codes") or hasattr(zonelog, "code_names")

    # Values should be integers where defined
    assert zonelog.values[3] == pytest.approx(1.0)
    assert zonelog.values[4] == pytest.approx(1.0)
    assert zonelog.values[5] == pytest.approx(2.0)
    assert zonelog.values[6] == pytest.approx(2.0)
    assert zonelog.values[7] == pytest.approx(3.0)


def test_welldata_rms_ascii_log_values(sample_rms_ascii_well_file):
    """Test specific log values are read correctly."""
    well = WellData.from_file(
        filepath=sample_rms_ascii_well_file,
        fformat="rms_ascii",
    )

    perm = well.get_log("Perm")
    poro = well.get_log("Poro")

    # Check known values from the test file
    expected_perm = [
        np.nan,
        np.nan,
        150.0,
        200.0,
        175.0,
        160.0,
        180.0,
        np.nan,
        np.nan,
        np.nan,
    ]
    expected_poro = [
        np.nan,
        np.nan,
        0.25,
        0.30,
        0.28,
        0.26,
        0.29,
        np.nan,
        np.nan,
        np.nan,
    ]

    for i, (exp_perm, exp_poro) in enumerate(zip(expected_perm, expected_poro)):
        if np.isnan(exp_perm):
            assert np.isnan(perm.values[i])
        else:
            assert perm.values[i] == pytest.approx(exp_perm)

        if np.isnan(exp_poro):
            assert np.isnan(poro.values[i])
        else:
            assert poro.values[i] == pytest.approx(exp_poro)


def test_welldata_rms_ascii_empty_well(tmp_path):
    """Test reading RMS ASCII file with no data points."""
    rms_content = """1.0
Test Well
WELL_1 100.0 200.0 0.0
1
TestLog UNK lin
"""
    rms_file = tmp_path / "empty_well.rms_ascii"
    rms_file.write_text(rms_content)

    well = WellData.from_file(filepath=rms_file, fformat="rms_ascii")

    assert well.name == "WELL_1"
    assert well.n_records == 0
    assert len(well.logs) == 1


def test_welldata_rms_ascii_only_coordinates(tmp_path):
    """Test RMS ASCII file with coordinates but no logs."""
    rms_content = """1.0
Coordinates Only
WELL_2 100.0 200.0 1000.0
0
100.0 200.0 1000.0
100.5 200.5 1001.0
101.0 201.0 1002.0
"""
    rms_file = tmp_path / "coords_only.rms_ascii"
    rms_file.write_text(rms_content)

    well = WellData.from_file(filepath=rms_file, fformat="rms_ascii")

    assert well.name == "WELL_2"
    assert well.n_records == 3
    assert len(well.logs) == 0
    assert len(well.survey_x) == 3
    assert well.survey_z[0] == pytest.approx(1000.0)
    assert well.survey_z[-1] == pytest.approx(1002.0)


def test_welldata_to_file_rms_ascii_preserves_undef(tmp_path):
    """Test that NaN values are written as -999 in RMS ASCII format."""
    # Create well with NaN values
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    from xtgeo.io.welldata._well_io import WellLog

    log_values = np.array([10.0, np.nan, 30.0])
    test_log = WellLog(name="TestLog", values=log_values, is_discrete=False)

    well = WellData(
        name="TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(test_log,),
    )

    # Write to file
    output_file = tmp_path / "test_undef.rms_ascii"
    well.to_file(filepath=output_file, fformat="rms_ascii")

    # Read back
    well2 = WellData.from_file(filepath=output_file, fformat="rms_ascii")

    log2 = well2.get_log("TestLog")
    assert log2.values[0] == pytest.approx(10.0)
    assert np.isnan(log2.values[1])
    assert log2.values[2] == pytest.approx(30.0)


def test_welldata_rms_ascii_multiple_discrete_logs(tmp_path):
    """Test RMS ASCII with multiple discrete logs."""
    rms_content = """1.0
Multiple Discrete
WELL_3 100.0 200.0 0.0
3
Zone DISC 1 Zone1 2 Zone2 3 Zone3
Facies DISC 0 Shale 1 Sand
Poro UNK lin
100.0 200.0 1000.0 1.0 0.0 0.15
100.0 200.0 1001.0 1.0 1.0 0.25
100.0 200.0 1002.0 2.0 1.0 0.30
100.0 200.0 1003.0 3.0 0.0 0.10
"""
    rms_file = tmp_path / "multi_disc.rms_ascii"
    rms_file.write_text(rms_content)

    well = WellData.from_file(filepath=rms_file, fformat="rms_ascii")

    assert well.n_records == 4
    assert len(well.logs) == 3

    zone = well.get_log("Zone")
    facies = well.get_log("Facies")
    poro = well.get_log("Poro")

    assert zone.is_discrete
    assert facies.is_discrete
    assert not poro.is_discrete

    np.testing.assert_array_equal(zone.values, [1.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(facies.values, [0.0, 1.0, 1.0, 0.0])


def test_welldata_rms_ascii_from_stringio():
    """Test reading RMS ASCII from StringIO stream."""
    rms_content = """1.0
Test Stream
STREAM_WELL 100.0 200.0 0.0
2
Poro UNK lin
Facies DISC 0 Shale 1 Sand
100.0 200.0 1000.0 0.25 0.0
101.0 201.0 1001.0 0.30 1.0
102.0 202.0 1002.0 0.28 1.0
"""
    stream = io.StringIO(rms_content)
    well = WellData.from_file(filepath=stream, fformat="rms_ascii")

    assert well.name == "STREAM_WELL"
    assert well.n_records == 3
    assert well.xpos == pytest.approx(100.0)
    assert well.ypos == pytest.approx(200.0)

    poro = well.get_log("Poro")
    np.testing.assert_array_almost_equal(poro.values, [0.25, 0.30, 0.28])


def test_welldata_rms_ascii_to_stringio():
    """Test writing RMS ASCII to StringIO stream."""
    # Create a simple well
    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    from xtgeo.io.welldata._well_io import WellLog

    poro_values = np.array([0.25, 0.30, 0.28])
    poro_log = WellLog(name="Poro", values=poro_values, is_discrete=False)

    well = WellData(
        name="TEST_STREAM",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(poro_log,),
    )

    # Write to StringIO
    stream = io.StringIO()
    well.to_file(filepath=stream, fformat="rms_ascii")

    # Read back
    stream.seek(0)
    well2 = WellData.from_file(filepath=stream, fformat="rms_ascii")

    assert well2.name == "TEST_STREAM"
    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.survey_x, survey_x, decimal=4)
    poro2 = well2.get_log("Poro")
    np.testing.assert_array_almost_equal(poro2.values, poro_values, decimal=4)


def test_welldata_rms_ascii_roundtrip_stringio():
    """Test roundtrip through StringIO preserves data."""
    rms_content = """1.0
Roundtrip Test
RT_WELL 500.0 600.0 10.0
3
Zone DISC 1 Upper 2 Middle 3 Lower
Perm UNK lin
Poro UNK lin
500.0 600.0 2000.0 1.0 150.0 0.25
501.0 601.0 2001.0 2.0 200.0 0.30
502.0 602.0 2002.0 3.0 175.0 0.28
"""
    # Read from StringIO
    stream1 = io.StringIO(rms_content)
    well1 = WellData.from_file(filepath=stream1, fformat="rms_ascii")

    # Write to new StringIO
    stream2 = io.StringIO()
    well1.to_file(filepath=stream2, fformat="rms_ascii")

    # Read back
    stream2.seek(0)
    well2 = WellData.from_file(filepath=stream2, fformat="rms_ascii")

    # Verify
    assert well1.name == well2.name
    assert well1.n_records == well2.n_records
    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x, decimal=4)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z, decimal=4)
