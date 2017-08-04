import unittest
import os
import glob
import sys
import logging
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

# =========================================================================
# Do tests
# =========================================================================


class Test(unittest.TestCase):
    """Testing suite for wells"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_import(self):
        """
        Import well from file
        """
        self.getlogger('test_import')

        wfile = "../../testdata/Well/T/a/31_2-E-1_H.w"

        mywell = Well()

        mywell.from_file(wfile)

        print("True well name:", mywell.truewellname)
        self.assertEqual(mywell.xpos, 524139.420, 'XPOS')
        self.assertEqual(mywell.ypos, 6740790.41, 'YPOS')
        self.assertEqual(mywell.wellname, '31/2-E-1_H', 'YPOS')

        self.logger.info(mywell.get_logtype('ZONELOG'))
        self.logger.info(mywell.get_logrecord('ZONELOG'))
        self.logger.info(mywell.lognames_all)
        self.logger.info(mywell.dataframe)

        # self.logger.info the numpy string of GR...
        self.logger.info(type(mywell.dataframe['GR'].values))

    def test_import_export_many(self):
        """ Import many wells (test speed)"""

        self.getlogger('test_import_export_many')

        wfiles = "../../testdata/Well/T/a/*"

        for filename in glob.glob(wfiles):
            self.logger.info("Importing " + filename)
            mywell = Well()
            mywell.from_file(filename)
            self.logger.info(mywell.nrows)
            self.logger.info(mywell.ncolumns)
            self.logger.info(mywell.lognames)

            wname = path + "/" + mywell.xwellname + ".w"
            self.logger.info("Exporting " + wname)
            mywell.to_file(wname)

    # def test_import_export_many2(self):
    #     """ Import many wells (test speed) GULLFAKS"""
    #     wfiles = "/project/gullfaks/resmod/gfmain_brent/2015a/" +\
    #         "r003/rms/output/tmp/etc/data/wells/geomodel/*.w"

    #     start = timer()
    #     for filename in glob.glob(wfiles):
    #         self.logger.info("Importing "+filename)
    #         mywell = Well()
    #         mywell.from_file(filename)
    #         # self.logger.info(mywell.nrows)
    #         # self.logger.info(mywell.ncolumns)
    #         # self.logger.info(mywell.lognames)

    #         # wname = path + "/" + mywell.xwellname + ".w"
    #         # self.logger.info("Exporting "+wname)
    #         # mywell.to_file(wname)

    #     end = timer()
    #     diff = end - start
    #     self.logger.info("\nImporten many gullfaks wells using {} seconds\n".format(diff))

    # def test_operations1(self):
    #     """Operation on a log."""

    #     wfile = "../../testdata/Well/T/a/31_2-1.w"

    #     mywell = Well()

    #     mywell.from_file(wfile)

    #     df = mywell.dataframe
    #     self.logger.info(df.head())

    #     # make GR = GR+100 if not -999 ...

    #     df['GR'].fillna(value=100, inplace=True)

    #     self.logger.info(df.head())

    #     # set zone 21 to -999

    #     df['ZONELOG'].loc[df['ZONELOG']==21] = np.nan

    #     # set GR to undef if ZONELOG is undef
    #     df.GR = df.GR.where(df.ZONELOG, np.nan)

    # mywell.to_file("TMP/x.w")

    def test_get_carr(self):
        """Get a C array pointer"""

        wfile = "../../testdata/Well/T/a/31_2-1.w"

        mywell = Well()

        mywell.from_file(wfile)

        dummy = mywell.get_carray("NOSUCH")

        self.assertEqual(dummy, None, 'Wrong log name')

        cref = mywell.get_carray("X_UTME")

        xref = str(cref)
        swig = False
        if "Swig" in xref and "double" in xref:
            swig = True

        self.assertEqual(swig, True, 'carray from log name, double')

        cref = mywell.get_carray("ZONELOG")

        xref = str(cref)
        swig = False
        if "Swig" in xref and "int" in xref:
            swig = True

        self.assertEqual(swig, True, 'carray from log name, int')

    def test_make_hlen(self):
        """Create a hlen log"""

        self.getlogger('test_make_hlen')

        wfile = "../../testdata/Well/T/a/31_2-1.w"

        mywell = Well()
        mywell.from_file(wfile)
        mywell.create_relative_hlen()

        print(mywell.dataframe)

    def test_fence(self):
        """Return a resampled fence"""

        self.getlogger('test_fence')

        wfile = "../../testdata/Well/G/w1/34_10-A-42.w"

        mywell = Well()
        mywell.from_file(wfile)
        pline = mywell.get_fence_polyline(extend=10, tvdmin=1000)

        print(pline)

    def test_get_zonation_points(self):
        """Get zonations points (zone tops)"""

        self.getlogger('test_get_zonation_points')

        wfile = "../../testdata/Well/T/a/31_2-1.w"

        mywell = Well().from_file(wfile)
        mywell.get_zonation_points(zonelogname='ZONELOG')

    def test_get_zonation_holes(self):
        """get a report of holes in the zonation, some samples with -999 """

        self.getlogger('test_get_zonation_holes')

        wfile = "../../testdata/Well/T/c/31_2-G-4_BY1H_holes.w"

        mywell = Well().from_file(wfile)
        report = mywell.report_zonation_holes(zonelogname='ZONELOG')

        self.logger.info("\n{}".format(report))

        self.assertEqual(report.iat[0, 0], 4166)  # first value for INDEX
        self.assertEqual(report.iat[1, 3], 1570.3855)  # second value for Z

        # ----------------------------------------------------------

        wfile = "../../testdata/Well/O2/w1_holes.w"

        mywell = Well().from_file(wfile)
        report = mywell.report_zonation_holes(zonelogname='Z2002A',
                                              mdlogname="MDEPTH")

        self.logger.info("\n{}".format(report))

        self.assertEqual(report.iat[0, 6], 3823.4)  # value for MD

        # ----------------------------------------------------------

        wfile = "../../testdata/Well/T/c/31_2-1.w"

        mywell = Well().from_file(wfile)
        report = mywell.report_zonation_holes(zonelogname='ZONELOG',
                                              mdlogname="MD")

        self.logger.info("\n{}".format(report))
        self.logger.info("\n{}".format(len(report)))

        self.assertEqual(len(report), 2)  # report length
        self.assertEqual(report.iat[1, 4], 28)  # zone no.


if __name__ == '__main__':

    unittest.main()
