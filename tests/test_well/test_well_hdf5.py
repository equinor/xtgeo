
import pytest

from xtgeo.well._well_io import import_wlogs


@pytest.mark.parametrize(
    "wlogs, expected_output",
    [
        (dict(), {"wlogtypes": {}, "wlogrecords": {}}),
        (
            dict([("X_UTME", ("CONT", None))]),
            {"wlogtypes": {"X_UTME": "CONT"}, "wlogrecords": {"X_UTME": None}},
        ),
        (
            dict([("ZONELOG", ("DISC", {"0": "ZONE00"}))]),
            {
                "wlogtypes": {"ZONELOG": "DISC"},
                "wlogrecords": {"ZONELOG": {"0": "ZONE00"}},
            },
        ),
    ],
)
def test_import_wlogs(wlogs, expected_output):
    assert import_wlogs(wlogs) == expected_output
