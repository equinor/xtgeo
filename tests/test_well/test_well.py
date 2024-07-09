import pathlib

import numpy as np
import pandas as pd
import pytest
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.well import Well
from xtgeo.xyz import Polygons

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

WFILE = pathlib.Path("wells/reek/1/OP_1.w")
WFILE2 = pathlib.Path("wells/reek/2/OP_6a.w")
WFILE_HOLES = pathlib.Path("wells/reek/1/OP_1_zholes.w")
WFILES = pathlib.Path("wells/reek/1/*")

WELL1 = pathlib.Path("wells/battle/1/WELL09.rmswell")
WELL2 = pathlib.Path("wells/battle/1/WELL36.rmswell")
WELL3 = pathlib.Path("wells/battle/1/WELL10.rmswell")

WELL4 = pathlib.Path("wells/drogon/1/55_33-1.rmswell")

SURF1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
SURF2 = pathlib.Path("surfaces/reek/1/basereek_rota.gri")


@pytest.fixture(name="loadwell1")
def fixture_loadwell1(testdata_path):
    """Fixture for loading a well (pytest setup)."""
    return xtgeo.well_from_file(testdata_path / WFILE)


@pytest.fixture(name="loadwell3")
def fixture_loadwell3(testdata_path):
    """Fixture for loading a well (pytest setup)."""
    return xtgeo.well_from_file(testdata_path / WELL3)


@pytest.fixture(name="simple_well")
def fixture_simple_well(string_to_well):
    wellstring = """1.01
Unknown
OP_1 0 0 0
4
Zonelog DISC 1 zone1 2 zone2 3 zone3
Poro UNK lin
Perm UNK lin
Facies DISC 0 Background 1 Channel 2 Crevasse
0 0 0 nan -999 -999 -999 -999
1 1 1 1 0.1 0.01 1
2 2 2 1 0.2 0.02 1
3 3 3 2 0.3 0.03 1
4 4 4 2 0.4 0.04 2
5 5 5 3 0.5 0.05 2"""
    well = string_to_well(wellstring)
    yield well


@pytest.fixture(name="create_well")
def fixture_create_well():
    dfr = pd.DataFrame(
        {
            "X_UTME": [444.00, 444.00],
            "Y_UTMN": [464.00, 464.00],
            "Z_TVDSS": [1000.00, 1001.00],
        }
    )
    return Well(25.0, 444.1, 464.1, "91/99-1", dfr)


def test_import(loadwell1, snapshot, helpers):
    """Import well from file."""

    mywell = loadwell1

    expected_result = {
        "well_name": "OP_1",
        "xpos": 461809.59,
        "wpos": 5932990.36,
        "wname": "OP_1",
        "rkb": 0.0,
        "nrow": 4866,
        "ncol": 7,
        "nlogs": 4,
        "lognames": ["Zonelog", "Perm", "Poro", "Facies"],
        "lognames_all": [
            "X_UTME",
            "Y_UTMN",
            "Z_TVDSS",
            "Zonelog",
            "Perm",
            "Poro",
            "Facies",
        ],
    }

    snapshot.assert_match(
        helpers.df2csv(mywell.get_dataframe().head(10).round()),
        "loadwell1.csv",
    )

    assert expected_result == {
        "well_name": mywell.wellname,
        "xpos": mywell.xpos,
        "wpos": mywell.ypos,
        "wname": mywell.wname,
        "rkb": mywell.rkb,
        "nrow": mywell.nrow,
        "ncol": mywell.ncol,
        "nlogs": mywell.nlogs,
        "lognames": mywell.lognames,
        "lognames_all": mywell.lognames_all,
    }


def test_import_long_well(loadwell3):
    """Import a longer well from file."""

    mywell = loadwell3

    logger.debug("True well name: %s", mywell.truewellname)

    mywell.geometrics()
    dfr = mywell.get_dataframe()

    assert dfr["Q_AZI"][27] == pytest.approx(91.856158, abs=0.0001)


def test_import_well_selected_logs(testdata_path):
    """Import a well but restrict on lognames"""

    mywell = xtgeo.well_from_file(testdata_path / WELL1, lognames="all")
    assert "ZONELOG" in mywell.get_dataframe()

    mywell = xtgeo.well_from_file(testdata_path / WELL1, lognames="GR")
    assert "ZONELOG" not in mywell.get_dataframe()

    mywell = xtgeo.well_from_file(testdata_path / WELL1, lognames=["GR"])
    assert "ZONELOG" not in mywell.get_dataframe()

    mywell = xtgeo.well_from_file(testdata_path / WELL1, lognames=["DUMMY"])
    assert "ZONELOG" not in mywell.get_dataframe()
    assert "GR" not in mywell.get_dataframe()

    with pytest.raises(ValueError) as msg:
        logger.info(msg)
        mywell = xtgeo.well_from_file(
            testdata_path / WELL1, lognames=["DUMMY"], lognames_strict=True
        )

    mywell = xtgeo.well_from_file(testdata_path / WELL1, mdlogname="GR")
    assert mywell.mdlogname == "GR"

    with pytest.raises(ValueError) as msg:
        mywell = xtgeo.well_from_file(
            testdata_path / WELL1, mdlogname="DUMMY", strict=True
        )
        logger.info(msg)

    mywell = xtgeo.well_from_file(
        testdata_path / WELL1, mdlogname="DUMMY", strict=False
    )
    assert mywell.mdlogname is None


@pytest.mark.parametrize(
    "log_name, newdict, expected",
    [
        ("Poro", {0: "null"}, "Cannot set a log record for a continuous log"),
        ("not_in_lognames", {}, "No such attr_name"),
        ("Facies", [], "Input is not a dictionary"),
    ],
)
def test_set_logrecord_invalid(simple_well, log_name, newdict, expected):
    mywell = simple_well

    with pytest.raises(ValueError, match=expected):
        mywell.set_logrecord(log_name, newdict)


def test_set_logrecord(simple_well):
    mywell = simple_well

    mywell.set_logrecord("Facies", {"some_key": "some_val"})
    assert mywell.get_logrecord("Facies") == {"some_key": "some_val"}


@pytest.mark.parametrize(
    "old_name, new_name, expected",
    [
        ("Poro", "Perm", "New log name exists already"),
        ("not_in_log", "irrelevant", "Input log does not exist"),
    ],
)
def test_rename_log_invalid(simple_well, old_name, new_name, expected):
    mywell = simple_well

    with pytest.raises(ValueError, match=expected):
        mywell.rename_log(old_name, new_name)


def test_rename_log(simple_well):
    mywell = simple_well
    old_log = mywell.get_logrecord("Poro")
    mywell.rename_log("Poro", "new_name")
    assert "new_name" in mywell.lognames
    assert mywell.get_logrecord("new_name") == old_log


@pytest.mark.parametrize(
    "log_name, change_from, change_to",
    [("Poro", "CONT", "DISC"), ("Poro", "CONT", "CONT"), ("Facies", "DISC", "CONT")],
)
def test_set_log_type(simple_well, log_name, change_from, change_to):
    mywell = simple_well
    assert mywell.get_logtype(log_name) == change_from
    mywell.set_logtype(log_name, change_to)
    assert mywell.get_logtype(log_name) == change_to


def test_loadwell1_properties(simple_well):
    """Import well from file and try to change lognames etc."""

    mywell = simple_well

    assert mywell.get_logtype("Poro") == "CONT"
    assert mywell.get_logrecord("Poro") == ("UNK", "lin")

    assert mywell.name == "OP_1"
    mywell.name = "OP_1_EDITED"
    assert mywell.name == "OP_1_EDITED"
    assert mywell.safewellname == "OP_1_EDITED"
    assert mywell.xwellname == "OP_1_EDITED"
    assert mywell.truewellname == "OP/1 EDITED"
    assert mywell.isdiscrete("Facies")
    assert not mywell.isdiscrete("Poro")
    assert not mywell.isdiscrete("Perm")
    assert mywell.get_logrecord("Facies") == {
        0: "Background",
        1: "Channel",
        2: "Crevasse",
    }


def test_shortwellname(create_well):
    """Test that shortwellname gives wanted result."""
    mywell = create_well

    mywell._wname = "31/2-A-14 2H"
    short = mywell.shortwellname

    assert short == "A-142H"

    mywell._wname = "6412_2-A-14_2H"
    short = mywell.shortwellname

    assert short == "A-142H"


# @pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No pytables on macOS")
# def test_hdf_io_single(tmp_path):
#     """Test HDF io, single well."""
#     mywell = xtgeo.well_from_file(WELL1)

#     wname = (tmp_path / "hdfwell").with_suffix(".hdf")
#     mywell.to_hdf(wname)
#     mywell2 = xtgeo.well_from_file(wname, fformat="hdf")
#     assert mywell2.nrow == mywell.nrow


# @pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No pytables on macOS")
# def test_import_as_rms_export_as_hdf_many(tmp_path, simple_well):
#     """Import RMS and export as HDF5 and RMS asc, many, and compare timings."""
#     t0 = xtg.timer()
#     wname = (tmp_path / "$random").with_suffix(".hdf")
#     wuse = simple_well.to_hdf(wname, compression=None)
#     print("Time for save HDF: ", xtg.timer(t0))

#     t0 = xtg.timer()
#     result = xtgeo.well_from_file(wuse, fformat="hdf5")
#     assert result.get_dataframe().equals(simple_well.get_dataframe())
#     print("Time for load HDF: ", xtg.timer(t0))


def test_import_export_rmsasc(tmp_path, simple_well):
    t0 = xtg.timer()
    wname = (tmp_path / "$random").with_suffix(".rmsasc")
    wuse = simple_well.to_file(wname)
    print("Time for save RMSASC: ", xtg.timer(t0))

    t0 = xtg.timer()
    result = xtgeo.well_from_file(wuse)
    assert result.get_dataframe().equals(result.get_dataframe())
    print("Time for load RMSASC: ", xtg.timer(t0))


def test_create_and_delete_logs(loadwell3):
    """Test create and delete logs, using explicit create_log() and delete_log()."""
    mywell = loadwell3

    status = mywell.create_log("NEWLOG")
    assert status is True

    status = mywell.create_log("NEWLOG", force=False)
    assert status is False

    status = mywell.create_log("NEWLOG", force=True, value=200)
    assert status is True
    assert mywell.get_dataframe().NEWLOG.mean() == 200.0

    ndeleted = mywell.delete_log("NEWLOG")

    assert ndeleted == 1
    status = mywell.create_log("NEWLOG", force=True, value=200)

    ndeleted = mywell.delete_log(["NEWLOG", "GR"])
    assert ndeleted == 2


def test_create_and_delete_logs_implicit(loadwell3):
    """Test create and delete logs, using implicit get_dataframe() operations."""
    mywell = loadwell3

    dfr = mywell.get_dataframe()
    dfr["NEWLOG"] = 1234.0
    mywell.set_dataframe(dfr)
    assert mywell.get_dataframe().NEWLOG.mean() == 1234.0
    assert "NEWLOG" in mywell.get_lognames()

    # status = mywell.create_log("NEWLOG", force=True, value=200)
    # assert status is True
    # assert mywell.get_dataframe().NEWLOG.mean() == 200.0

    # ndeleted = mywell.delete_log("NEWLOG")

    # assert ndeleted == 1
    # status = mywell.create_log("NEWLOG", force=True, value=200)

    # ndeleted = mywell.delete_log(["NEWLOG", "GR"])
    # assert ndeleted == 2


def test_wlogtypes(loadwell3):
    mywell = loadwell3
    wlogtypes = mywell.wlogtypes
    assert wlogtypes == {
        "X_UTME": "CONT",
        "Y_UTMN": "CONT",
        "Z_TVDSS": "CONT",
        "GR": "CONT",
        "ZONELOG": "DISC",
    }


def test_get_set_wlogs(loadwell3):
    """Test on getting ans setting a dictionary with some log attributes."""
    mywell = loadwell3

    mydict = mywell.get_wlogs()
    print(mydict)

    assert isinstance(mydict, dict)

    assert mydict["X_UTME"][0] == "CONT"
    assert mydict["ZONELOG"][0] == "DISC"
    assert mydict["ZONELOG"][1][24] == "ZONE_24"

    mydict["ZONELOG"][1][24] = "ZONE_24_EDITED"
    mywell.set_wlogs(mydict)

    mydict2 = mywell.get_wlogs()
    assert mydict2["X_UTME"][0] == "CONT"
    assert mydict2["ZONELOG"][0] == "DISC"
    assert mydict2["ZONELOG"][1][24] == "ZONE_24_EDITED"

    mydict2["EXTRA"] = None
    mywell.set_wlogs(mydict2)


def test_make_hlen(loadwell1):
    """Create a hlen log."""

    mywell = loadwell1
    mywell.create_relative_hlen()

    logger.debug(mywell.get_dataframe())


def test_make_zqual_log(loadwell3):
    """Make a zonelog FLAG quality log."""
    mywell = loadwell3
    mywell.zonelogname = "ZONELOG"

    logger.debug("True well name: %s", mywell.truewellname)

    mywell.make_zone_qual_log("ZQUAL")

    recs = mywell.get_logrecord("ZQUAL")
    assert recs == {
        0: "UNDETERMINED",
        1: "INCREASE",
        2: "DECREASE",
        3: "U_TURN",
        4: "INV_U_TURN",
        9: "INCOMPLETE",
    }
    unique, counts = np.unique(
        mywell.get_dataframe()["ZQUAL"].values, return_counts=True
    )
    assert unique.tolist() == [0.0, 1.0, 2.0, 3.0, 9.0]
    assert counts.tolist() == [714, 123, 612, 782, 90]


@pytest.mark.parametrize(
    "logseries, nsamples, expected",
    [
        ([], 0, []),
        ([1], 0, [False]),
        ([1], 1, [False]),  # intentional
        ([1, 2], 0, [False, False]),
        ([0, 0], 1, [False, False]),
        ([1, 2], 1, [True, True]),
        ([1, 1, 1, 2, 2, 2], 1, [False, False, True, True, False, False]),
        ([1, 1, 1, 2, 2, 2], 2, [False, True, True, True, True, False]),
        ([10, 10, 10, 2, 2, 2], 1, [False, False, True, True, False, False]),
        ([1, 1, 1, 2, 2, 2], 10, [True, True, True, True, True, True]),
        ([np.nan, 1, 1, np.nan, 2, 2], 1, [False, False, False, False, False, False]),
        ([np.nan, 1, 1, 2, np.nan, 2], 1, [False, False, True, True, False, False]),
    ],
)
def test_mask_shoulderbeds_get_bseries(logseries, nsamples, expected):
    """Test for corner cases in _mask_shoulderbeds_logs."""
    from xtgeo.well._well_oper import _get_bseries

    logseries = pd.Series(logseries, dtype="float64")
    expected = pd.Series(expected, dtype="bool")

    results = _get_bseries(logseries, nsamples)
    if logseries.empty:
        assert expected.empty is True
    else:
        assert results.equals(expected)


@pytest.mark.parametrize(
    "dep, lseries, distance, expected",
    [
        (
            [1.0],
            [1],
            1.2,
            [False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            0.7,
            [False, False, True, True, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            1.2,
            [False, False, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            0.49999,
            [False, False, False, False, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            0.50,
            [False, False, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, np.nan, np.nan],
            1.2,
            [False, False, True, True, False, False],
        ),
    ],
)
def test_well_mask_shoulder_get_bseries_by_distance(dep, lseries, distance, expected):
    """Test for corner cases in get bseries by distance."""
    # note the cxtgeo function is also tested in test_cxtgeo_lowlevel!
    from xtgeo.well._well_oper import _get_bseries_by_distance

    dep = pd.Series(dep, dtype="float64")
    lseries = pd.Series(lseries, dtype="float64")
    expected = np.array(expected, dtype="bool")

    result = _get_bseries_by_distance(dep, lseries, distance)
    assert (result == expected).all()


def test_mask_shoulderbeds(loadwell3, loadwell1):
    """Test masking shoulderbeds effects."""
    mywell = loadwell3

    usewell = mywell.copy()
    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples=3)

    assert not np.isnan(mywell.get_dataframe().at[1595, "GR"])
    assert np.isnan(usewell.get_dataframe().at[1595, "GR"])

    # another well set with more discrete logs and several logs to modify
    mywell = loadwell1
    usewell = mywell.copy()
    usewell.mask_shoulderbeds(["Zonelog", "Facies"], ["Perm", "Poro"], nsamples=2)
    assert np.isnan(usewell.get_dataframe().at[4763, "Perm"])
    assert np.isnan(usewell.get_dataframe().at[4763, "Poro"])

    # corner cases
    with pytest.raises(ValueError):
        usewell.mask_shoulderbeds(
            ["Zonelog", "Facies"], ["NOPerm", "Poro"], nsamples=2, strict=True
        )

    with pytest.raises(ValueError):
        usewell.mask_shoulderbeds(["Perm", "Facies"], ["NOPerm", "Poro"], nsamples=2)

    assert usewell.mask_shoulderbeds(["Zonelog"], ["Perm", "Poro"]) is True
    assert usewell.mask_shoulderbeds(["Zonelog"], ["Dummy"]) is False
    assert usewell.mask_shoulderbeds(["Dummy"], ["Perm", "Poro"]) is False


def test_mask_shoulderbeds_use_tvd_md(loadwell3):
    """Test masking shoulderbeds effects using tvd or md distance."""
    mywell = loadwell3

    usewell = mywell.copy()
    # small distance for this test since almost horizontal well
    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"tvd": 0.01})

    assert not np.isnan(mywell.get_dataframe().at[1595, "GR"])
    assert np.isnan(usewell.get_dataframe().at[1595, "GR"])

    usewell = mywell.copy()

    with pytest.raises(ValueError) as verr:
        # since mdlogname does not exist
        usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"md": 1.6})
    assert "no mdlogname attribute present" in str(verr)

    usewell.geometrics()  # to create Q_MDEPTH as mdlogname
    usewell.mask_shoulderbeds(["ZONELOG"], ["GR"], nsamples={"md": 1.6})
    assert not np.isnan(mywell.get_dataframe().at[1595, "GR"])
    assert np.isnan(usewell.get_dataframe().at[1595, "GR"])


def test_geometrics_exception(string_to_well):
    well = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 1 1 1
"""
    )
    with pytest.raises(ValueError, match="Not enough trajectory points"):
        well.geometrics()


def test_geometrics_simple(string_to_well):
    well = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 1 1 1
3 1 1 1
4 1 1 1"""
    )
    expected_result = {
        "Q_MDEPTH": [0, 1, 2, 3],
        "Q_INCL": [90, 90, 90, 90],
        "Q_AZI": [90, 90, 90, 90],
    }
    well.geometrics()
    result = {
        "Q_MDEPTH": well.get_dataframe()["Q_MDEPTH"].values.tolist(),
        "Q_INCL": well.get_dataframe()["Q_INCL"].values.tolist(),
        "Q_AZI": well.get_dataframe()["Q_AZI"].values.tolist(),
    }
    assert result == expected_result


@pytest.mark.parametrize(
    "well_input, expected",
    [
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
0 0 1 nan
3 4 1 1
6 8 1 1
11 20 1 1
18 44 1 1""",
            [0, 5, 10, 23, 48],
        ),
        (
            """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
0 1 0 nan
3 1 4 1
6 1 8 1
11 1 20 1
18 1 44 1""",
            [0, 5, 10, 23, 48],
        ),
    ],
)
def test_geometrics(string_to_well, well_input, expected):
    well = string_to_well(well_input)
    well.geometrics()
    assert well.get_dataframe()["Q_MDEPTH"].values.tolist() == expected


def test_rescale_well(loadwell1):
    """Rescale (resample) a well to a finer increment"""

    mywell = loadwell1

    df1 = mywell.get_dataframe().copy()
    df1 = df1[(df1["Zonelog"] == 1)]

    mywell.rescale(delta=0.2)

    df2 = mywell.get_dataframe().copy()
    df2 = df2[(df2["Zonelog"] == 1)]

    assert df1["Perm"].mean() == pytest.approx(df2["Perm"].mean(), abs=20.0)
    assert df1["Poro"].mean() == pytest.approx(df2["Poro"].mean(), abs=0.001)


def test_rescale_well_tvdrange(tmp_path, testdata_path):
    """Rescale (resample) a well to a finer increment within a TVD range"""

    mywell = xtgeo.well_from_file(testdata_path / WELL1)
    mywell.to_file(tmp_path / "wll1_pre_rescale.w")
    gr_avg1 = mywell.get_dataframe()["GR"].mean()
    mywell.rescale(delta=2, tvdrange=(1286, 1333))
    mywell.to_file(tmp_path / "wll1_post_rescale.w")
    gr_avg2 = mywell.get_dataframe()["GR"].mean()
    assert gr_avg1 == pytest.approx(gr_avg2, abs=0.9)

    mywell1 = xtgeo.well_from_file(testdata_path / WELL1)
    mywell1.rescale(delta=2)

    mywell2 = xtgeo.well_from_file(testdata_path / WELL1)
    mywell2.rescale(delta=2, tvdrange=(0, 9999))
    assert mywell1.get_dataframe()["GR"].mean() == mywell2.get_dataframe()["GR"].mean()
    assert mywell1.get_dataframe()["GR"].all() == mywell2.get_dataframe()["GR"].all()


def test_rescale_well_tvdrange_coarsen_upper(tmp_path, testdata_path):
    """Rescale (resample) a well to a coarser increment in top part"""

    mywell = xtgeo.well_from_file(testdata_path / WELL1)
    mywell.rescale(delta=20, tvdrange=(0, 1200))
    mywell.to_file(tmp_path / "wll1_rescale_coarsen.w")
    assert mywell.get_dataframe().iat[10, 3] == pytest.approx(365.8254, abs=0.1)


def test_fence(testdata_path):
    """Return a resampled fence."""

    mywell = xtgeo.well_from_file(testdata_path / WFILE)
    pline = mywell.get_fence_polyline(nextend=10, tvdmin=1000)
    assert pline.shape == (31, 5)


def test_fence2(testdata_path):
    """Return a resampled fence, another dataset that troubled in scipy interpd."""
    mywell = xtgeo.well_from_file(testdata_path / WFILE2)
    pline = mywell.get_fence_polyline(tvdmin=0, asnumpy=False)
    assert pline.get_dataframe().shape == (137, 6)
    assert pline.get_dataframe().at[136, "H_CUMLEN"] == pytest.approx(2713.7405)


def test_fence_as_polygons(testdata_path):
    """Return a resampled fence as Polygons."""

    mywell = xtgeo.well_from_file(testdata_path / WFILE)
    pline = mywell.get_fence_polyline(nextend=3, tvdmin=1000, asnumpy=False)

    assert isinstance(pline, Polygons)
    dfr = pline.get_dataframe()
    assert dfr["X_UTME"][5] == pytest.approx(462569.00, abs=2.0)


def test_fence_as_polygons_drogon(testdata_path):
    """Return a resampled fence as Polygons for a 100% vertical."""

    mywell = xtgeo.well_from_file(testdata_path / WELL4)
    pline = mywell.get_fence_polyline(
        nextend=3, tvdmin=1000, sampling=20, asnumpy=False
    )

    assert isinstance(pline, Polygons)
    dfr = pline.get_dataframe()
    assert dfr.H_CUMLEN.max() == pytest.approx(62.858, abs=0.01)


def test_get_zonation_points(testdata_path):
    """Get zonations points (zone tops)"""

    mywell = xtgeo.well_from_file(testdata_path / WFILE, zonelogname="Zonelog")
    ppx = mywell.get_zonation_points()
    assert ppx.loc[2, "TopName"] == "TopBelow_TopLowerReek"


def test_get_zone_interval(testdata_path):
    """Get zonations points (zone tops)"""

    mywell = xtgeo.well_from_file(testdata_path / WFILE, zonelogname="Zonelog")
    line = mywell.get_zone_interval(3)

    assert line.iat[0, 0] == pytest.approx(462698.33299, abs=0.001)
    assert line.iat[-1, 2] == pytest.approx(1643.1618, abs=0.001)


def test_remove_parallel_parts(testdata_path):
    """Remove the part of the well thst is parallel with some other"""

    well1 = xtgeo.well_from_file(testdata_path / WELL1)
    well2 = xtgeo.well_from_file(testdata_path / WELL2)

    well1.truncate_parallel_path(well2)

    assert well1.nrow == 3290


def test_get_zonation_holes(testdata_path):
    """get a report of holes in the zonation, some samples with -999"""

    mywell = xtgeo.well_from_file(testdata_path / WFILE_HOLES, zonelogname="Zonelog")
    report = mywell.report_zonation_holes()

    assert report.iat[0, 0] == 4193  # first value for INDEX
    assert report.iat[1, 3] == 1609.5800  # second value for Z


def test_get_filled_dataframe(testdata_path):
    """Get a filled DataFrame"""

    mywell = xtgeo.well_from_file(testdata_path / WFILE)

    df1 = mywell.get_dataframe()

    df2 = mywell.get_filled_dataframe(fill_value=-999, fill_value_int=-888)

    logger.info(df1)
    logger.info(df2)

    assert np.isnan(df1.iat[4860, 6])
    assert df2.iat[4860, 6] == -888


def test_create_surf_distance_log(loadwell1, testdata_path):
    """Test making a log which is distance to a surface."""

    well = loadwell1

    surf1 = xtgeo.surface_from_file(testdata_path / SURF1)
    surf2 = xtgeo.surface_from_file(testdata_path / SURF2)

    well.create_surf_distance_log(surf1, name="DIST_TOP")
    well.create_surf_distance_log(surf2, name="DIST_BASE")

    assert well.get_dataframe().loc[0, "DIST_TOP"] == pytest.approx(1653.303263)
    assert well.get_dataframe().loc[0, "DIST_BASE"] == pytest.approx(1696.573171)

    # moving the surface so it is outside the well
    surf1.translate_coordinates((10000, 10000, 0))
    well.create_surf_distance_log(surf1, name="DIST_BASE_NEW")
    assert np.isnan(well.get_dataframe().loc[0, "DIST_BASE_NEW"])


def test_create_surf_distance_log_more(tmp_path, loadwell1, testdata_path):
    """Test making a log which is distance to a surface and do some operations.

    This is a prototype  exploring the possibility to run a check if zonelog is
    some % of being within surfaces. I.e. a surface version of:

    Grid().report_zone_mismatch() method

    When/if the prototype is incorporated this test should be removed.
    """

    well = loadwell1

    surf1 = xtgeo.surface_from_file(testdata_path / SURF1)
    surf2 = xtgeo.surface_from_file(testdata_path / SURF2)

    well.create_surf_distance_log(surf1, name="DIST_TOP")
    well.create_surf_distance_log(surf2, name="DIST_BASE")

    # for simplicity create one zone log
    lrec = {0: "ABOVE", 1: "IN", 2: "BELOW"}
    well.create_log("MEGAZONE1", logtype="DISC", logrecord=lrec)
    well.create_log("MEGAZONE2", logtype="DISC", logrecord=lrec)

    dfr = well.get_dataframe()
    zlog_df = well.get_dataframe()["Zonelog"]
    # well.get_dataframe()["MEGAZONE1"][(zl > 0) & (zl < 4)] = 1   # << get Pd warnings

    dfr.loc[(zlog_df > 0) & (zlog_df < 4), "MEGAZONE1"] = 1
    dfr.loc[zlog_df > 3, "MEGAZONE1"] = 2
    well.set_dataframe(dfr)

    _, counts = np.unique(well.get_dataframe()["MEGAZONE1"].values, return_counts=True)
    assert counts.tolist() == [4780, 75, 11]

    dfr.loc[np.isnan(zlog_df), "MEGAZONE1"] = np.nan
    well.set_dataframe(dfr)
    _, counts = np.unique(well.get_dataframe()["MEGAZONE1"].values, return_counts=True)
    assert counts.tolist() == [4779, 75, 11, 1]

    # derive from distance log:
    d1 = well.get_dataframe()["DIST_TOP"]
    d2 = well.get_dataframe()["DIST_BASE"]
    dfr = well.get_dataframe()
    dfr.loc[(d1 <= 0.0) & (d2 > 0), "MEGAZONE2"] = 1
    well.set_dataframe(dfr)

    _, counts = np.unique(well.get_dataframe()["MEGAZONE2"].values, return_counts=True)
    assert counts.tolist() == [4788, 78]

    # now use logics from Grid() report_zone_mismatch()...
    # much coding pasting vvvvvv =======================================================

    zname = "MEGAZONE1"
    zmodel = "MEGAZONE2"
    zonelogname = "MEGAZONE1"
    depthrange = [1200, 3000]
    zonelogrange = [1, 1]

    well.to_file(tmp_path / "well_surf_dist.w")

    # get the IJK along the well as logs; use a copy of the well instance
    wll = well.copy()

    if depthrange:
        d1, d2 = depthrange
        dfr = wll.get_dataframe()
        wll.set_dataframe(dfr[(d1 < dfr.Z_TVDSS) & (d2 > dfr.Z_TVDSS)])

    # from here, work with the dataframe only
    dfr = wll.get_dataframe().copy()

    # zonelogrange
    z1, z2 = zonelogrange
    zmin = int(dfr[zonelogname].min())
    zmax = int(dfr[zonelogname].max())
    skiprange = list(range(zmin, z1)) + list(range(z2 + 1, zmax + 1))

    for zname in (zonelogname, zmodel):
        if skiprange:  # needed check; du to a bug in pandas version 0.21 .. 0.23
            dfr[zname] = dfr[zname].replace(skiprange, -888)
        dfr[zname] = dfr[zname].fillna(-999)
    # now there are various variotions on how to count mismatch:
    # dfuse 1: count matches when zonelogname is valid (exclude -888)
    # dfuse 2: count matches when zonelogname OR zmodel are valid (exclude < -888
    # or -999)
    # The first one is the original approach

    dfuse1 = dfr.copy(deep=True)
    dfuse1 = dfuse1.loc[dfuse1[zonelogname] > -888]

    dfuse1["zmatch1"] = np.where(dfuse1[zmodel] == dfuse1[zonelogname], 1, 0)
    mcount1 = dfuse1["zmatch1"].sum()
    tcount1 = dfuse1["zmatch1"].count()
    if not np.isnan(mcount1):
        mcount1 = int(mcount1)
    if not np.isnan(tcount1):
        tcount1 = int(tcount1)

    res1 = dfuse1["zmatch1"].mean() * 100

    dfuse2 = dfr.copy(deep=True)
    dfuse2 = dfuse2.loc[(dfr[zmodel] > -888) | (dfr[zonelogname] > -888)]
    dfuse2["zmatch2"] = np.where(dfuse2[zmodel] == dfuse2[zonelogname], 1, 0)
    mcount2 = dfuse2["zmatch2"].sum()
    tcount2 = dfuse2["zmatch2"].count()
    if not np.isnan(mcount2):
        mcount2 = int(mcount2)
    if not np.isnan(tcount2):
        tcount2 = int(tcount2)

    res2 = dfuse2["zmatch2"].mean() * 100

    # update Well() copy (segment only)
    wll.set_dataframe(dfuse2)

    res = {
        "MATCH1": res1,
        "MCOUNT1": mcount1,
        "TCOUNT1": tcount1,
        "MATCH2": res2,
        "MCOUNT2": mcount2,
        "TCOUNT2": tcount2,
        "WELLINTV": wll,
    }

    assert res["MATCH2"] == pytest.approx(93.67, abs=0.03)


def test_copy(string_to_well):
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
    13 14 15 3"""
    well = string_to_well(wellstring)
    well_copy = well.copy()
    assert well.get_dataframe().equals(well_copy.get_dataframe())
    assert well.lognames == well_copy.lognames
    assert well.name == well_copy.name
    assert well.wname == well_copy.wname
    assert well.rkb == well_copy.rkb
    assert (well.xpos, well.ypos) == (well_copy.xpos, well_copy.ypos)
    assert well.lognames_all == well_copy.lognames_all
    assert well.lognames == well_copy.lognames


@pytest.mark.parametrize(
    "well_definition, expected_hlen",
    [
        (
            """1.01
    Unknown
    name 0 0 0
    1
    Zonelog DISC 1 zone1 2 zone2 3 zone3
    1 1 1 nan
    1 2 1 1
    1 3 1 1
    1 4 1 2
    1 5 1 2
    1 6 1 3""",
            [0, 1, 2, 3, 4, 5],
        ),
        (
            """1.01
        Unknown
        name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3
        1 1 1 nan
        2 1 1 1
        3 1 1 1
        4 1 1 2
        5 1 1 2
        6 1 1 3""",
            [0, 1, 2, 3, 4, 5],
        ),
        (
            """1.01
        Unknown
        name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3
        1 1 1 nan
        2 1 2 1
        3 1 4 1
        4 1 4 2
        5 1 5 2
        6 1 6 3""",
            [0, 1, 2, 3, 4, 5],
        ),
        (
            """1.01
        Unknown
        name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3
        1 1 1 nan
        2 2 1 1
        3 3 1 1
        4 4 1 2
        5 5 1 2
        6 6 1 3""",
            [
                0.0,
                1.4142135623730951,
                2.8284271247461903,
                4.242640687119286,
                5.656854249492381,
                7.0710678118654755,
            ],
        ),
        (
            """1.01
        Unknown
        name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3
        1 1 1 nan
        2 3 1 1
        3 5 1 1
        4 9 1 2
        5 50 1 2
        50 180 1 3""",
            [
                0.0,
                2.23606797749979,
                4.47213595499958,
                8.595241580617241,
                49.607434889436995,
                187.17559981141304,
            ],
        ),
    ],
)
def test_create_relative_hlen(string_to_well, well_definition, expected_hlen):
    well = string_to_well(well_definition)
    well.create_relative_hlen()
    assert well.get_dataframe()["R_HLEN"].to_list() == pytest.approx(expected_hlen)


def test_truncate_parallel_path_too_short(string_to_well):
    well_1 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 2 1 1
3 3 1 1
4 4 1 1"""
    )
    well_2 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
3 3 1 1
4 4 1 1"""
    )
    with pytest.raises(ValueError, match="Too few points to truncate parallel path"):
        well_1.truncate_parallel_path(well_2)

    with pytest.raises(ValueError, match="Too few points to truncate parallel path"):
        well_2.truncate_parallel_path(well_1)


def test_truncate_parallel_path(string_to_well):
    well_1 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 1
2 2 1 1
3 3 1 1
4 4 1 1
5 5 1 1"""
    )
    well_2 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
2 2 1 1
3 3 1 1
4 4 1 1"""
    )
    well_1.truncate_parallel_path(well_2)
    assert well_1.get_dataframe().to_dict() == {
        "X_UTME": {0: 1.0, 1: 5.0},
        "Y_UTMN": {0: 1.0, 1: 5.0},
        "Z_TVDSS": {0: 1.0, 1: 1.0},
        "Zonelog": {0: 1.0, 1: 1.0},
    }


@pytest.mark.parametrize(
    "x1, x2, y1, y2",
    [
        pytest.param(2, 1, 1, 1, id="xmin1 > xmax2"),
        pytest.param(1, 1, 2, 1, id="ymin1 > ymax2"),
        pytest.param(1, 2, 1, 1, id="xmin2 > xmax1"),
        pytest.param(1, 1, 1, 2, id="ymin2 > ymax1"),
    ],
)
def test_may_overlap_no_overlap(string_to_well, x1, x2, y1, y2):
    well_1 = string_to_well(
        f"""1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
{x1} {y1} 1 nan"""
    )
    well_2 = string_to_well(
        f"""1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
{x2} {y2} 1 nan"""
    )
    assert not well_1.may_overlap(well_2)


def test_may_overlap(string_to_well):
    well_1 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 2 1 1
3 3 1 1"""
    )
    well_2 = string_to_well(
        """1.01
Unknown
name 0 0 0
1
Zonelog DISC 1 zone1 2 zone2 3 zone3
1 1 1 nan
2 2 1 1
3 3 1 1"""
    )
    assert well_1.may_overlap(well_2)


@pytest.mark.parametrize(
    "lower_limit, upper_limit, expected_result",
    [
        (0, 1000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, 9, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (1, 8, [1, 2, 3, 4, 5, 6, 7, 8]),
        (2, 5, [2, 3, 4, 5]),
        (10, 0, []),
    ],
)
def test_limit_tvd(string_to_well, upper_limit, lower_limit, expected_result):
    well_definition = """1.01
            Unknown
            custom_name 0 0 0
            1
            Zonelog DISC 1 zone1 2 zone2 3 zone3"""

    for i in range(10):
        well_definition += f"\n        {i} {i} {i} 1"

    well = string_to_well(well_definition)
    well.limit_tvd(lower_limit, upper_limit)
    assert well.get_dataframe()["Z_TVDSS"].to_list() == expected_result


@pytest.mark.parametrize(
    "input_points, expected_points",
    [(range(10), [0, 4, 8, 9]), ([1, 10, 11, 12, 13, 14, 100, 10000], [1, 13, 10000])],
)
def test_downsample(string_to_well, input_points, expected_points):
    well_definition = """1.01
            Unknown
            custom_name 0 0 0
            1
            Zonelog DISC 1 zone1 2 zone2 3 zone3"""

    for i in input_points:
        well_definition += f"\n        {i} {i} {i} 1"

    well = string_to_well(well_definition)
    well.downsample()
    dataframe = well.get_dataframe()
    assert {
        "X_UTME": dataframe["X_UTME"].values.tolist(),
        "Y_UTMN": dataframe["Y_UTMN"].values.tolist(),
        "Z_TVDSS": dataframe["Z_TVDSS"].values.tolist(),
    } == {
        "X_UTME": expected_points,
        "Y_UTMN": expected_points,
        "Z_TVDSS": expected_points,
    }


@pytest.mark.parametrize(
    "input_points, expected_points",
    [(range(10), [0, 4, 8]), ([1, 10, 11, 12, 13, 14, 100, 10000], [1, 13])],
)
def test_downsample_not_keeplast(string_to_well, input_points, expected_points):
    well_definition = """1.01
            Unknown
            custom_name 0 0 0
            1
            Zonelog DISC 1 zone1 2 zone2 3 zone3"""

    for i in input_points:
        well_definition += f"\n        {i} {i} {i} 1"

    well = string_to_well(well_definition)
    well.downsample(keeplast=False)
    dataframe = well.get_dataframe()
    assert {
        "X_UTME": dataframe["X_UTME"].to_list(),
        "Y_UTMN": dataframe["Y_UTMN"].to_list(),
        "Z_TVDSS": dataframe["Z_TVDSS"].to_list(),
    } == {
        "X_UTME": expected_points,
        "Y_UTMN": expected_points,
        "Z_TVDSS": expected_points,
    }


def test_get_polygons(string_to_well):
    well_definition = """1.01
        Unknown
        custom_name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3"""

    for x, y, z in zip(
        np.random.random(10), np.random.random(10), np.random.random(10)
    ):
        well_definition += f"\n        {x} {y} {z} 1"

    well = string_to_well(well_definition)
    polygons = well.get_polygons()
    dataframe = well.get_dataframe()
    poly_dataframe = polygons.get_dataframe()

    assert dataframe["X_UTME"].to_list() == pytest.approx(
        poly_dataframe["X_UTME"].to_list()
    )
    assert dataframe["Y_UTMN"].to_list() == pytest.approx(
        poly_dataframe["Y_UTMN"].to_list()
    )
    assert dataframe["X_UTME"].to_list() == pytest.approx(
        poly_dataframe["X_UTME"].to_list()
    )
    assert "Zonelog" not in poly_dataframe.columns
    assert "NAME" in poly_dataframe.columns
    assert polygons.name == "custom_name"


def test_get_polygons_skipname(string_to_well):
    well_definition = """1.01
        Unknown
        custom_name 0 0 0
        1
        Zonelog DISC 1 zone1 2 zone2 3 zone3
                1 1 1 1"""

    well = string_to_well(well_definition)
    polygons = well.get_polygons(skipname=True)
    assert "NAME" not in polygons.get_dataframe().columns
    assert polygons.name == "custom_name"


def test_welldf_view_vs_copy(loadwell1):
    well = loadwell1

    df1 = well.get_dataframe(copy=False)
    df2 = well.get_dataframe(copy=False)
    assert id(df1) == id(df2)

    df1 = well.get_dataframe(copy=True)
    df2 = well.get_dataframe(copy=True)
    assert id(df1) != id(df2)


def test_well_dataframe_deprecation(loadwell1):
    well = loadwell1

    with pytest.warns(PendingDeprecationWarning):
        well.dataframe
