import pathlib

import pytest
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

WFILE = pathlib.Path("wells/etc/otest.rmswell")


def test_wellzone_to_points(testdata_path):
    """Import well from file and put zone boundaries to a Pandas object."""

    mywell = xtgeo.well_from_file(
        testdata_path / WFILE, zonelogname="Zone_model2", mdlogname="M_MDEPTH"
    )

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=False)
    assert zpoints.iat[9, 6] == 6

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=True)
    assert zpoints.iat[9, 6] == 7

    with pytest.raises(ValueError):
        zpoints = mywell.get_zonation_points(zonelist=[1, 3, 4, 5])

    zpoints = mywell.get_zonation_points(zonelist=[3, 4, 5])
    assert zpoints.iat[6, 6] == 4

    zpoints = mywell.get_zonation_points(zonelist=(3, 5))
    assert zpoints.iat[6, 6] == 4


def test_wellzone_to_isopoints(testdata_path):
    """Import well from file and find thicknesses"""

    mywell = xtgeo.well_from_file(
        testdata_path / WFILE, zonelogname="Zone_model2", mdlogname="M_MDEPTH"
    )
    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=False, tops=True)
    assert zpoints["Zone"].min() == 3
    assert zpoints["Zone"].max() == 9

    zisos = mywell.get_zonation_points(use_undef=False, tops=False)
    assert zisos.iat[10, 8] == 4


def test_zonepoints_non_existing(string_to_well):
    well = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 2 1 1
3 3 1 1"""
    )
    points = well.get_zonation_points(use_undef=False, tops=True, zonelist=[5, 10])
    assert points is None


@pytest.mark.parametrize(
    "well_spec, zonelist, use_undef, expected_result",
    [
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 1
4 5 6 2
7 8 9 3
""",
            [1, 2],
            False,
            {"X_UTME": [4.0], "Y_UTMN": [5.0]},  # Expected result
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 1
4 5 6 2
7 8 9 3
""",
            [2, 3],
            False,
            {"X_UTME": [4.0, 7.0], "Y_UTMN": [5.0, 8.0]},  # Expected result
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 nan
4 5 6 2
7 8 9 3
""",
            [1, 2],
            False,
            {"X_UTME": [], "Y_UTMN": []},  # Expected result
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 1
4 5 6 2
7 8 9 3
""",
            [1, 2],
            True,
            {"X_UTME": [1.0, 4.0], "Y_UTMN": [2.0, 5.0]},  # Expected result
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 1
4 5 6 2
7 8 9 3
""",
            [2, 3],
            True,
            {"X_UTME": [4.0, 7.0], "Y_UTMN": [5.0, 8.0]},  # Expected result
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone2
1 2 3 nan
4 5 6 2
7 8 9 3
""",
            [1, 2],
            True,
            {"X_UTME": [4.0, 4.0], "Y_UTMN": [5.0, 5.0]},  # Expected result
        ),
    ],
)
def test_simple_points(well_spec, expected_result, string_to_well, zonelist, use_undef):
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(well_spec, **kwargs)
    points = well.get_zonation_points(use_undef=use_undef, tops=True, zonelist=zonelist)
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result


def test_simple_points_two(string_to_well):
    wellstring = """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
0 0 0 nan
1 2 3 1
4 5 6 1
7 8 9 2
10 11 12 2
13 14 15 3
"""
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    points = well.get_zonation_points(use_undef=False, tops=True, zonelist=[1, 2, 3])
    expected_result = {"X_UTME": [7.0, 13.0], "Y_UTMN": [8.0, 14.0]}
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result
    assert len(points) == 2
    points = well.get_zonation_points(use_undef=False, tops=True, zonelist=[2, 3])
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result
    assert len(points) == 2


def test_break_zonation(string_to_well):
    wellstring = """1.01
    Unknown
    name 0 0 0
    1
    Zonelog DISC 1 zone1 2 zone2 3 zone3
    0 0 0 2000000001
    1 2 3 1
    4 5 6 2
    7 8 9 1
    10 11 12 2
    13 14 15 3
    """
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    points = well.get_zonation_points(use_undef=False, tops=True, zonelist=[1, 2, 3])
    expected_result = {
        "X_UTME": [4.0, 4.0, 10.0, 13.0],
        "Y_UTMN": [5.0, 5.0, 11.0, 14.0],
    }
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result


@pytest.mark.parametrize(
    "zonelist,error_type,error_message",
    [
        ([1], ValueError, "list must contain two or"),
        ((1,), ValueError, "tuple must be of length 2, was 1"),
        ({"a": 2}, TypeError, "zonelist must be either list"),
        (1, TypeError, "zonelist must be either list"),
        ("zonelist", TypeError, "zonelist must be either list"),
        ([1, 3], ValueError, "zonelist must be strictly increasing"),
        ([1, 2, 1], ValueError, "zonelist must be strictly increasing"),
    ],
)
def test_invalid_zonelist_type(string_to_well, zonelist, error_type, error_message):
    wellstring = """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2
1 2 3 1
3 4 5 2
5 6 7 2
7 8 9 2
"""
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    with pytest.raises(error_type, match=error_message):
        well.get_zonation_points(use_undef=False, tops=True, zonelist=zonelist)


@pytest.mark.parametrize(
    "tops_flag, expected_result",
    [
        pytest.param(
            False, {"X_UTME": [10.0], "Y_UTMN": [11.0]}, id="Points (thickness)"
        ),
        pytest.param(True, {"X_UTME": [7.0, 13.0], "Y_UTMN": [8.0, 14.0]}, id="Tops"),
    ],
)
def test_tops_value(string_to_well, tops_flag, expected_result):
    wellstring = """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 2 3 1
4 5 6 1
7 8 9 2
10 11 12 2
13 14 15 3
"""
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    points = well.get_zonation_points(use_undef=False, tops=tops_flag, zonelist=[2, 3])
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result


@pytest.mark.parametrize(
    "include_limit, expected_result",
    [
        (12, {"X_UTME": [], "Y_UTMN": []}),
        (1, {"X_UTME": [], "Y_UTMN": []}),
        (45, {"X_UTME": [], "Y_UTMN": []}),
        (80, {"X_UTME": [10.0], "Y_UTMN": [11.0]}),
        (90, {"X_UTME": [10.0], "Y_UTMN": [11.0]}),
        (100, {"X_UTME": [10.0], "Y_UTMN": [11.0]}),
        (360, {"X_UTME": [10.0], "Y_UTMN": [11.0]}),
    ],
)
def test_include_limit(string_to_well, include_limit, expected_result):
    wellstring = """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 2 3 1
4 5 6 1
7 8 9 2
10 11 12 2
13 14 15 3
"""
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    points = well.get_zonation_points(
        use_undef=False, tops=False, zonelist=(1, 5), incl_limit=include_limit
    )
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result


@pytest.mark.parametrize(
    "zone_list, expected_result",
    [
        (
            (1, 2),
            {"X_UTME": [7.0], "Y_UTMN": [8.0]},
        ),
        (
            (1, 3),
            {"X_UTME": [7.0, 13.0], "Y_UTMN": [8.0, 14.0]},
        ),
        (
            (1, 4),
            {"X_UTME": [7.0, 13.0, 16.0], "Y_UTMN": [8.0, 14.0, 17.0]},
        ),
        (
            (2, 4),
            {"X_UTME": [7.0, 13.0, 16.0], "Y_UTMN": [8.0, 14.0, 17.0]},
        ),
        (
            None,
            {"X_UTME": [7.0, 13.0, 16.0], "Y_UTMN": [8.0, 14.0, 17.0]},
        ),
        (
            [1, 2],
            {"X_UTME": [7.0], "Y_UTMN": [8.0]},
        ),
        (
            [1, 2, 3],
            {"X_UTME": [7.0, 13.0], "Y_UTMN": [8.0, 14.0]},
        ),
        (
            [1, 2, 3, 4],
            {"X_UTME": [7.0, 13.0, 16.0], "Y_UTMN": [8.0, 14.0, 17.0]},
        ),
    ],
)
def test_zonelist(string_to_well, zone_list, expected_result):
    wellstring = """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3 4 zone4
1 2 3 1
4 5 6 1
7 8 9 2
10 11 12 2
13 14 15 3
16 17 18 4
"""
    kwargs = {"zonelogname": "Zonelog"}
    well = string_to_well(wellstring, **kwargs)
    points = well.get_zonation_points(
        use_undef=False,
        tops=True,
        zonelist=zone_list,
    )
    assert {
        "X_UTME": points["X_UTME"].to_list(),
        "Y_UTMN": points["Y_UTMN"].to_list(),
    } == expected_result
