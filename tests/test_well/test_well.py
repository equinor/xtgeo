import glob
import sys
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================


def test_import():
    """Import well from file."""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-E-1_H.w"

    mywell = Well(wfile)

    logger.debug("True well name:", mywell.truewellname)
    tsetup.assert_equal(mywell.xpos, 524139.420, 'XPOS')
    tsetup.assert_equal(mywell.ypos, 6740790.41, 'YPOS')
    tsetup.assert_equal(mywell.wellname, '31/2-E-1_H', 'YPOS')

    logger.info(mywell.get_logtype('ZONELOG'))
    logger.info(mywell.get_logrecord('ZONELOG'))
    logger.info(mywell.lognames_all)
    logger.info(mywell.dataframe)

    # logger.info the numpy string of GR...
    logger.info(type(mywell.dataframe['GR'].values))


def test_import_export_many():
    """ Import many wells (test speed)"""

    wfiles = "../xtgeo-testdata/wells/tro/1/*"
    logger.debug(wfiles)

    for filename in glob.glob(wfiles):
        logger.info("Importing " + filename)
        mywell = Well(filename)
        logger.info(mywell.nrow)
        logger.info(mywell.ncol)
        logger.info(mywell.lognames)

        wname = td + "/" + mywell.xwellname + ".w"
        logger.info("Exporting " + wname)
        mywell.to_file(wname)

# def test_import_export_many2():
#     """ Import many wells (test speed) GULLFAKS"""
#     wfiles = "/project/gullfaks/resmod/gfmain_brent/2015a/" +\
#         "r003/rms/output/tmp/etc/data/wells/geomodel/*.w"

#     start = timer()
#     for filename in glob.glob(wfiles):
#         logger.info("Importing "+filename)
#         mywell = Well()
#         mywell.from_file(filename)
#         # logger.info(mywell.nrow)
#         # logger.info(mywell.ncol)
#         # logger.info(mywell.lognames)

#         # wname = path + "/" + mywell.xwellname + ".w"
#         # logger.info("Exporting "+wname)
#         # mywell.to_file(wname)

#     end = timer()
#     diff = end - start
#     logger.info("\nImporten many gullfaks wells using {} seconds\n".format(diff))

# def test_operations1():
#     """Operation on a log."""

#     wfile = "../../testdata/Well/T/a/31_2-1.w"

#     mywell = Well()

#     mywell.from_file(wfile)

#     df = mywell.dataframe
#     logger.info(df.head())

#     # make GR = GR+100 if not -999 ...

#     df['GR'].fillna(value=100, inplace=True)

#     logger.info(df.head())

#     # set zone 21 to -999

#     df['ZONELOG'].loc[df['ZONELOG']==21] = np.nan

#     # set GR to undef if ZONELOG is undef
#     df.GR = df.GR.where(df.ZONELOG, np.nan)

# mywell.to_file("TMP/x.w")


def test_get_carr():
    """Get a C array pointer"""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-1.w"

    mywell = Well(wfile)

    dummy = mywell.get_carray("NOSUCH")

    tsetup.assert_equal(dummy, None, 'Wrong log name')

    cref = mywell.get_carray("X_UTME")

    xref = str(cref)
    swig = False
    if "Swig" in xref and "double" in xref:
        swig = True

    tsetup.assert_equal(swig, True, 'carray from log name, double')

    cref = mywell.get_carray("ZONELOG")

    xref = str(cref)
    swig = False
    if "Swig" in xref and "int" in xref:
        swig = True

    tsetup.assert_equal(swig, True, 'carray from log name, int')


def test_make_hlen():
    """Create a hlen log."""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-1.w"

    mywell = Well(wfile)
    mywell.create_relative_hlen()

    logger.debug(mywell.dataframe)


def test_fence():
    """Return a resampled fence."""

    wfile = "../xtgeo-testdata/wells/gfb/1/34_10-A-42.w"

    mywell = Well(wfile)
    pline = mywell.get_fence_polyline(extend=10, tvdmin=1000)

    logger.debug(pline)


def test_get_zonation_points():
    """Get zonations points (zone tops)"""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-1.w"

    mywell = Well(wfile, zonelogname='ZONELOG')
    mywell.get_zonation_points()


@tsetup.skipifroxar  # fails if roxar version; wrong pandas?
def test_get_zone_interval():
    """Get zonations points (zone tops)"""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-E-3_Y1H.w"

    mywell = Well(wfile, zonelogname='ZONELOG')
    line = mywell.get_zone_interval(10)

    logger.info(type(line))

    tsetup.assert_equal(line.iat[0, 0], 524826.882)
    tsetup.assert_equal(line.iat[-1, 2], 1555.3452)


def test_get_zonation_holes():
    """get a report of holes in the zonation, some samples with -999 """

    wfile = "../xtgeo-testdata/wells/tro/3/31_2-G-4_BY1H_holes.w"

    mywell = Well(wfile, zonelogname='ZONELOG')
    report = mywell.report_zonation_holes()

    logger.info("\n{}".format(report))

    tsetup.assert_equal(report.iat[0, 0], 4166)  # first value for INDEX
    tsetup.assert_equal(report.iat[1, 3], 1570.3855)  # second value for Z

    # ----------------------------------------------------------

    wfile = "../xtgeo-testdata/wells/oea/1/w1_holes.w"

    mywell = Well(wfile, zonelogname='Z2002A', mdlogname='MDEPTH')
    report = mywell.report_zonation_holes()

    logger.info("\n{}".format(report))

    tsetup.assert_equal(report.iat[0, 6], 3823.4)  # value for MD

    # ----------------------------------------------------------

    wfile = "../xtgeo-testdata/wells/tro/3/31_2-1.w"

    mywell = Well(wfile, zonelogname='ZONELOG', mdlogname='MD')
    report = mywell.report_zonation_holes()

    logger.info("\n{}".format(report))
    logger.info("\n{}".format(len(report)))

    tsetup.assert_equal(len(report), 2)  # report length
    tsetup.assert_equal(report.iat[1, 4], 28)  # zone no.


@tsetup.skipifroxar
def test_get_filled_dataframe():
    """Get a filled DataFrame"""

    wfile = "../xtgeo-testdata/wells/tro/1/31_2-1.w"

    mywell = Well(wfile)

    df1 = mywell.dataframe

    df2 = mywell.get_filled_dataframe()

    logger.debug(df1)
    logger.debug(df2)
