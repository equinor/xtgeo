import logging
import pathlib

import numpy as np
import pandas as pd
import pytest

import xtgeo

logger = logging.getLogger(__name__)

GRIDFILE = pathlib.Path("3dgrids/reek/reek_sim_grid.roff")
ZONEFILE = pathlib.Path("3dgrids/reek/reek_sim_zone.roff")
WELL1 = pathlib.Path("wells/reek/1/OP_1.w")
WELL2 = pathlib.Path("wells/reek/1/OP_2.w")
WELL3 = pathlib.Path("wells/reek/1/OP_3.w")
WELL4 = pathlib.Path("wells/reek/1/OP_4.w")
WELL5 = pathlib.Path("wells/reek/1/OP_5.w")
WELL6 = pathlib.Path("wells/reek/1/WI_1.w")
WELL7 = pathlib.Path("wells/reek/1/WI_3.w")

PWELL1 = pathlib.Path("wells/reek/1/OP1_perf.w")

MATCHD1 = {
    "WI_1": 75,
    "WI_3": 75,
    "OP_4": 78,
    "OP_5": 78,
    "OP_1": 80,
    "OP_2": 77,
    "OP_3": 77,
}

MATCHD2 = {
    "WI_1": 65,
    "WI_3": 40,
    "OP_4": 71,
    "OP_5": 69,
    "OP_1": 71,
    "OP_2": 65,
    "OP_3": 70,
}


def _well(values):
    return xtgeo.Well(df=pd.DataFrame(values, dtype=float))


def test_get_distance_to_wells():
    """Get three-dimensional and lateral distances to a well."""
    grid = xtgeo.create_box_grid((3, 3, 2), origin=(0, 0, 0), increment=(10, 10, 10))
    well = _well(
        {
            "X_UTME": [5, 5],
            "Y_UTMN": [5, 5],
            "Z_TVDSS": [5, 15],
        }
    )

    distance_3d = grid.get_distance_to_wells(well)
    distance_lateral = grid.get_distance_to_wells(well, metric="horizontal")

    assert distance_3d.name == "DISTANCE2WELL"
    assert not distance_3d.isdiscrete
    assert distance_3d.dimensions == grid.dimensions
    assert np.allclose(distance_3d.values[0, 0, :], 0)
    assert distance_3d.values[1, 0, 0] == pytest.approx(10)
    assert np.all(distance_lateral.values <= distance_3d.values)


def test_get_distance_to_wells_multiple_and_blocked_well():
    """Use the nearest of multiple wells and blocked-well indices."""
    grid = xtgeo.create_box_grid((3, 1, 1), increment=(10, 10, 10))
    first_well = _well({"X_UTME": [5], "Y_UTMN": [5], "Z_TVDSS": [5]})
    second_well = _well({"X_UTME": [25], "Y_UTMN": [5], "Z_TVDSS": [5]})
    blocked_well = xtgeo.BlockedWell(
        df=pd.DataFrame(
            {
                "X_UTME": [5.0],
                "Y_UTMN": [5.0],
                "Z_TVDSS": [5.0],
                "I_INDEX": [0.0],
                "J_INDEX": [0.0],
                "K_INDEX": [0.0],
            }
        )
    )

    first_distance = grid.get_distance_to_wells(first_well)
    combined_distance = grid.get_distance_to_wells([first_well, second_well])
    blocked_distance = grid.get_distance_to_wells(blocked_well)

    assert np.allclose(combined_distance.values[:, 0, 0], [0, 10, 0])
    assert np.allclose(first_distance.values, blocked_distance.values)


def test_get_distance_to_wells_handles_invalid_cells():
    """Mask inactive cells and reject input that cannot identify well cells."""
    grid = xtgeo.create_box_grid((2, 2, 1), increment=(10, 10, 10))
    grid._actnumsv[1, 1, 0] = 0
    well = _well({"X_UTME": [5], "Y_UTMN": [5], "Z_TVDSS": [5]})
    outside_well = _well({"X_UTME": [-5], "Y_UTMN": [5], "Z_TVDSS": [5]})

    distance = grid.get_distance_to_wells(well)

    assert distance.values[1, 1, 0] is np.ma.masked
    with pytest.raises(ValueError, match="None of the wells penetrates"):
        grid.get_distance_to_wells(outside_well)
    with pytest.raises(ValueError, match="Unsupported metric"):
        grid.get_distance_to_wells(well, metric="invalid")
    with pytest.raises(TypeError, match="Well or BlockedWell"):
        grid.get_distance_to_wells([well, object()])


# A problem here is that the OP wells has very few samples, which
# makes a assumed match of 100% (since only one point)
# Also, the match percent seems to be a bit unstable, hence
# the rounding to INT...


@pytest.mark.bigtest
def test_report_zlog_mismatch(testdata_path):
    """Report zone log mismatch grid and well."""
    g1 = xtgeo.grid_from_file(testdata_path / GRIDFILE)

    zo = xtgeo.gridproperty_from_file(testdata_path / ZONEFILE, name="Zone")

    w1 = xtgeo.well_from_file(testdata_path / WELL1)
    w2 = xtgeo.well_from_file(testdata_path / WELL2)
    w3 = xtgeo.well_from_file(testdata_path / WELL3)
    w4 = xtgeo.well_from_file(testdata_path / WELL4)
    w5 = xtgeo.well_from_file(testdata_path / WELL5)
    w6 = xtgeo.well_from_file(testdata_path / WELL6)
    w7 = xtgeo.well_from_file(testdata_path / WELL7)

    wells = [w1, w2, w3, w4, w5, w6, w7]

    for wll in wells:
        response = g1.report_zone_mismatch(
            well=wll,
            zonelogname="Zonelog",
            zoneprop=zo,
            zonelogrange=(1, 3),
            depthrange=[1300, 9999],
        )

        match = int(float(f"{response[0]:.4f}"))
        logger.info("Match for %s is %s", wll.wellname, match)
        # assert match == MATCHD1[wll.name]

        # check also with resultformat=2
        res = g1.report_zone_mismatch(
            well=wll,
            zonelogname="Zonelog",
            zoneprop=zo,
            zonelogrange=(1, 3),
            depthrange=[1300, 9999],
            resultformat=2,
        )

        match = int(float(f"{res['MATCH2']:.4f}"))
        logger.info("Match for %s is %s", wll.wellname, match)
        # assert match == MATCHD2[wll.name]


def test_report_zlog_mismatch_resultformat3(tmp_path, testdata_path):
    """Report zone log mismatch grid and well, export updated wellsegment"""
    g1 = xtgeo.grid_from_file(testdata_path / GRIDFILE)

    zo = xtgeo.gridproperty_from_file(testdata_path / ZONEFILE, name="Zone")

    w1 = xtgeo.well_from_file(testdata_path / WELL1)

    res = g1.report_zone_mismatch(
        well=w1,
        zonelogname="Zonelog",
        zoneprop=zo,
        zonelogrange=(1, 3),
        depthrange=[1300, 9999],
        resultformat=3,
    )
    mywell = res["WELLINTV"]
    logger.info("\n%s", mywell.get_dataframe().to_string())
    mywell.to_file(tmp_path / "w1_zlog_report.rmswell")


def test_report_zlog_mismatch_perflog(tmp_path, testdata_path):
    """Report zone log mismatch grid and well filter on PERF"""
    g1 = xtgeo.grid_from_file(testdata_path / GRIDFILE)

    zo = xtgeo.gridproperty_from_file(testdata_path / ZONEFILE, name="Zone")

    w1 = xtgeo.well_from_file(testdata_path / PWELL1)

    w1.get_dataframe().to_csv(tmp_path / "testw1.csv")

    res = g1.report_zone_mismatch(
        well=w1,
        zonelogname="Zonelog",
        zoneprop=zo,
        zonelogrange=(1, 3),
        depthrange=[1580, 9999],
        perflogname="PERF",
        resultformat=2,
    )
    mywell = res["WELLINTV"]
    logger.info("\n%s", mywell.get_dataframe().to_string())
    mywell.to_file(tmp_path / "w1_perf_report.rmswell")

    assert res["MATCH2"] == pytest.approx(81, 1.5)
    assert res["TCOUNT2"] == 56
    assert res["MCOUNT2"] == 46

    w1 = xtgeo.well_from_file(testdata_path / WELL1)

    # well is missing perflog; hence result shall be None
    res = g1.report_zone_mismatch(
        well=w1,
        zonelogname="Zonelog",
        zoneprop=zo,
        zonelogrange=(1, 3),
        depthrange=[1580, 9999],
        perflogname="PERF",
        resultformat=2,
    )

    # ask for perflogname but no such present
    assert res is None
