import logging
import pathlib

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
