import pathlib
from io import StringIO

import pytest

import xtgeo
from xtgeo.surface._zmap_parser import ZMAPFormatException, ZMAPSurface, parse_zmap


@pytest.mark.parametrize(
    "zmap_input",
    [
        StringIO(
            """!     RMS - Reservoir Modeling System GRID TO ZYCOR GRID
!     RMS - Reservoir Modeling System VERSION : 12.1
!     GRID FILE NAME   : zmap_example.zmap
!     CREATION DATE    : 2021.05.05
!     CREATION TIME    : 08:20:10
@zmap_example.zmap HEADER                     ,      GRID, 5
      15,    -99999.0000,       ,      4,      1
       5,      4,    459218.9063,    460718.9063,   5924787.0000,   5925787.0000
         0.0000,         0.0000,         0.0000
@
+ Grid data starts after this line
        41.8596        43.1112        44.5115        45.6861        46.5453
        42.3947        42.8493        44.9472        47.2538        49.0361
        40.1785        51.0874        54.6339        55.6271        55.9614
    -99999.0000    -99999.0000    -99999.0000        68.1617        63.9811"""
        ),
        StringIO(
            """! Accept header values ending with comma which may occur
! Cf: https://github.com/abduhbm/zmapio/blob/main/examples/NSLCU.dat
@zmap_example.zmap HEADER                     ,      GRID, 5
      15,    -99999.0000,       ,      4,      1
       5,      4,    459218.9063,    460718.9063,   5924787.0000,   5925787.0000,
         0.0000,         0.0000,         0.0000
@
+ Grid data starts after this line
        41.8596        43.1112        44.5115        45.6861        46.5453
        42.3947        42.8493        44.9472        47.2538        49.0361
        40.1785        51.0874        54.6339        55.6271        55.9614
    -99999.0000    -99999.0000    -99999.0000        68.1617        63.9811"""
        ),
        pathlib.Path("surfaces/etc/zmap_example.zmap"),
    ],
)
def test_zmap_input_format(zmap_input, testdata_path):
    if isinstance(zmap_input, pathlib.Path):
        zmap_input = testdata_path / zmap_input
    header = parse_zmap(zmap_input)
    expected_header = {
        "nan_value": -99999.0,
        "ncol": 4,
        "node_width": 15,
        "nrow": 5,
        "precision": 4,
        "start_column": 1,
        "xmax": 460718.9063,
        "xmin": 459218.9063,
        "ymax": 5925787.0,
        "ymin": 5924787.0,
        "nr_nodes_per_line": 5,
    }
    header.values = None  # Not asserting on the values in this case
    assert header == ZMAPSurface(**expected_header)


def test_integration(testdata_path):
    result = xtgeo.surface_from_file(
        testdata_path / pathlib.Path("surfaces/etc/zmap_example.zmap")
    )
    assert result.xmax == 460718.9063
    assert result.ymax == 5925787.0


@pytest.mark.parametrize(
    "values_flag, expected_result",
    [(True, [2.0, 1.0, 4.0, 3.0]), (False, [0.0, 0.0, 0.0, 0.0])],
)
def test_integration_values(values_flag, expected_result):
    result = xtgeo.surface_from_file(
        StringIO(
            """!     Example 2x2 grid
@zmap_example.zmap HEADER  ,      GRID, 5
      15,    -99999.0000,       ,      4,      1
       2,      2,    1.0000,    2.0000,   1.0000,   2.0000
         0.0000,         0.0000,         0.0000
@
+ Grid data starts after this line
        1.0000        2.0000        3.0000        4.0000
"""
        ),
        fformat="zmap",
        values=values_flag,
    )
    assert result.xmax == 2.0
    assert result.ymax == 2.0
    assert result.xinc == 1.0
    assert result.yinc == 1.0
    assert list(result.values.data.flatten()) == expected_result


@pytest.mark.parametrize(
    "zmap_input, expected_error",
    [
        (
            """@zmap_example.zmap HEADER  ,      GRID, 5
  15,    -99999.0000,       ,      4,      1
   2,      2,    1.0000,    2.0000,   1.0000,   2.0000
@
+ Grid data starts after this line
    1.0000        2.0000        3.0000        4.0000
""",
            r"Failed to unpack line: \['@'\]",
        ),
        (
            """@zmap_example.zmap HEADER  ,      GRID, 5
  15,    -99999.0000,       ,      4,      1
   2,      2,    1.0000,    2.0000,   1.0000
""",
            r"Failed to unpack line: \['2', '2', '1.0000', '2.0000', '1.0000'\]",
        ),
        (
            """!     Example 2x2 grid
@zmap_example.zmap HEADER  ,      GRID, 5
  15,    -99999.0000,       ,      4,      1
   2,      2,    1.0000,    2.0000,   1.0000,   2.0000
         0.0000,         0.0000,         0.0000
    1.0000        2.0000        3.0000        4.0000
""",
            "Did not reach the values section, expected @, found",
        ),
        (
            """@zmap_example.zmap HEADER  ,      NOT CORRECT, 5
    """,
            "Expected GRID as second entry in line, got: NOT CORRECT",
        ),
        (
            """!     Example 2x2 grid
@zmap_example.zmap HEADER  ,      GRID, 5
!
""",
            "End reached without complete header",
        ),
    ],
    ids=[
        "Miss last hdr line",
        "Miss item in hdr line 3",
        "Miss @ after hdr",
        "GRID key is missing",
        "Incomplete hdr",
    ],
)
def test_not_complete_header(zmap_input, expected_error):
    zmap = StringIO(zmap_input)
    with pytest.raises(ZMAPFormatException, match=expected_error):
        parse_zmap(zmap)
