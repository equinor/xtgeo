from io import StringIO

import pytest

from xtgeo.io.tsurf._tsurf_reader import TSurfData, read_tsurf


@pytest.mark.parametrize(
    "tsurf_file",
    [
        "GOCAD TSurf 1\n"
        "HEADER {\n"
        "name:header_name\n"
        "}\n"
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME NiceCoordSysName\n"
        'AXIS_NAME "X" "Y" "Z"\n'
        'AXIS_UNIT "m" "m" "m"\n'
        "ZPOSITIVE Depth\n"
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
        "TFACE\n"
        "VRTX 1 459621.051270 5934843.011475 1685.590820 CNXYZ\n"
        "VRTX 2 459372.795166 5935276.098389 1758.701050 CNXYZ\n"
        "VRTX 3 459372.795166 5935276.098389 1758.701050 CNXYZ\n"
        "VRTX 4 459621.051270 5934843.011475 1685.590820 CNXYZ\n"
        "TRGL 1 2 3\n"
        "TRGL 1 4 2\n"
        "END\n"
    ],
)
def test_read_tsurf(tsurf_file: str) -> None:
    result: TSurfData = read_tsurf(StringIO(tsurf_file))
    assert result is not None
    assert isinstance(result, TSurfData)
    assert result.header.name == "header_name"
    assert result.coord_sys.name == "NiceCoordSysName"
    assert result.coord_sys.axis_unit == ["m", "m", "m"]
    assert len(result.vertices) == 4
    assert len(result.triangles) == 2
