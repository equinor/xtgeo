"""Importing grid props from ECL runs, e,g, INIT, UNRST"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import xtgeo

from . import _grid_eclbin_record as _eclbin
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

XTGDEBUG = 0

# cf metadata["IPHS"]:
PHASES = {
    1: "oil",
    2: "water",
    3: "oil/water",
    4: "gas",
    5: "oil/gas",
    6: "gas/water",
    7: "oil/water/gas",
}


def import_eclbinary(
    self, pfile, name=None, etype=1, date=None, grid=None, fracture=False, _kwlist=None
):

    # if pfile is a file, then the file is opened/closed here; otherwise, the
    # "outer" routine must handle that

    local_fhandle = False
    fhandle = pfile
    if isinstance(pfile, str):
        local_fhandle = True
        pfile = xtgeo._XTGeoCFile(pfile)
        fhandle = pfile.fhandle

    status = 0

    logger.info("Import ECL binary, name requested is %s", name)

    # scan file for properties byte positions etc
    if _kwlist is None:
        logger.info("Make kwlist, scan keywords")
        kwlist = utils.scan_keywords(
            fhandle, fformat="xecl", maxkeys=100000, dataframe=True, dates=True
        )
    else:
        kwlist = _kwlist

    metadata = _import_eclbinary_meta(self, fhandle, kwlist, etype, date, grid)
    date = metadata["DATE"]

    if name == "SGAS":
        status = _import_sgas(self, fhandle, kwlist, metadata, grid, date, fracture)

    elif name == "SOIL":
        status = _import_soil(self, fhandle, kwlist, metadata, grid, date, fracture)

    elif name == "SWAT":
        status = _import_swat(self, fhandle, kwlist, metadata, grid, date, fracture)

    if status == 0:
        name = name.replace("{__}", "")

        logger.info("Importing %s", name)

        _import_eclbinary_checks1(self, grid)

        kwname, kwlen, kwtype, kwbyte = _import_eclbinary_checks2(
            kwlist, name, etype, date
        )

        if grid._dualporo:  # _dualporo shall always be True if _dualperm is True
            _import_eclbinary_dualporo(
                self,
                grid,
                fhandle,
                kwname,
                kwlen,
                kwtype,
                kwbyte,
                name,
                date,
                etype,
                fracture,
            )
        else:
            _import_eclbinary_prop(
                self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype
            )

    if local_fhandle and not pfile.close(cond=local_fhandle):
        raise RuntimeError("Error in closing file handle for binary Eclipse file")


def _import_swat(self, fhandle, kwlist, metadata, grid, date, fracture):
    """Import SWAT; this may lack in very special cases"""

    if metadata["IPHS"] in (3, 6, 7):
        import_eclbinary(
            self,
            fhandle,
            name="SWAT{__}",
            etype=5,
            grid=grid,
            date=date,
            fracture=fracture,
            _kwlist=kwlist,
        )

    else:

        self.name = "SWAT" + "_" + str(date)
        self._nrow = grid.nrow
        self._ncol = grid.ncol
        self._nlay = grid.nlay
        self._date = date
        if metadata["IPHS"] == 2:
            self._values = np.ones(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )
        else:
            self._values = np.zeros(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )

        if grid.dualporo:
            if fracture:
                self.name = "SWATF" + "_" + str(date)
                self._values[grid._dualactnum.values == 1] = 0.0
            else:
                self.name = "SWATM" + "_" + str(date)
                self._values[grid._dualactnum.values == 2] = 0.0

    gactnum = grid.get_actnum().values
    self._values = ma.masked_where(gactnum < 1, self._values)
    return 3


def _import_sgas(self, fhandle, kwlist, metadata, grid, date, fracture):
    """Import SGAS; this may be lack of oil/water (need to verify)"""

    flag = 0
    if metadata["IPHS"] in (5, 7):
        import_eclbinary(
            self,
            fhandle,
            name="SGAS{__}",
            etype=5,
            grid=grid,
            date=date,
            fracture=fracture,
            _kwlist=kwlist,
        )

    elif metadata["IPHS"] == 6:
        flag = 1
        logger.info("SGAS: ask for SWAT")
        swat = self.__class__()
        import_eclbinary(
            swat,
            fhandle,
            name="SWAT{__}",
            etype=5,
            grid=grid,
            date=date,
            fracture=fracture,
            _kwlist=kwlist,
        )

        self.name = "SGAS" + "_" + str(date)
        self._nrow = grid.nrow
        self._ncol = grid.ncol
        self._nlay = grid.nlay
        self._date = date
        self._values = swat._values * -1 + 1.0
        del swat

    elif metadata["IPHS"] in (1, 2, 3, 4):
        flag = 1
        logger.info("SGAS: asked for but 0% or 100%")
        self.name = "SGAS" + "_" + str(date)
        self._nrow = grid.nrow
        self._ncol = grid.ncol
        self._nlay = grid.nlay
        self._date = date
        if metadata["IPHS"] == 4:
            self._values = np.ones(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )
        else:
            self._values = np.zeros(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )

    if grid.dualporo and flag:
        if fracture:
            self.name = "SGASF" + "_" + str(date)
            self._values[grid._dualactnum.values == 1] = 0.0
        else:
            self.name = "SGASM" + "_" + str(date)
            self._values[grid._dualactnum.values == 2] = 0.0

    gactnum = grid.get_actnum().values
    self._values = ma.masked_where(gactnum < 1, self._values)
    return 1


def _import_soil(self, fhandle, kwlist, metadata, grid, date, fracture):

    flag = 0
    if metadata["IPHS"] in (3, 5, 7):
        sgas = None
        swat = None
        flag = 1
        logger.info("Making SOIL from SWAT and/or SGAS ...")
        if metadata["IPHS"] in (3, 7):
            swat = self.__class__()
            import_eclbinary(
                swat,
                fhandle,
                name="SWAT{__}",
                etype=5,
                grid=grid,
                date=date,
                fracture=fracture,
                _kwlist=kwlist,
            )

        if metadata["IPHS"] in (5, 7):  # og, owg
            sgas = self.__class__()
            import_eclbinary(
                sgas,
                fhandle,
                name="SGAS{__}",
                etype=5,
                grid=grid,
                date=date,
                fracture=fracture,
                _kwlist=kwlist,
            )

        self.name = "SOIL" + "_" + str(date)
        self._nrow = grid.nrow
        self._ncol = grid.ncol
        self._nlay = grid.nlay
        self._date = date
        if metadata["IPHS"] == 7:  # owg
            self._values = swat._values * -1 - sgas._values + 1.0
        elif metadata["IPHS"] == 5:  # og
            self._values = sgas._values * -1 + 1.0
        elif metadata["IPHS"] == 3:  # ow
            self._values = swat._values * -1 + 1.0

        if swat:
            del swat
        if sgas:
            del sgas

    elif metadata["IPHS"] in (1, 2, 4, 6):
        flag = 1
        logger.info("SOIL: asked for but 0% or 100%")
        self.name = "SOIL" + "_" + str(date)
        self._nrow = grid.nrow
        self._ncol = grid.ncol
        self._nlay = grid.nlay
        self._date = date
        if metadata["IPHS"] == 1:
            self._values = np.ones(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )
        else:
            self._values = np.zeros(
                (self._ncol, self._nrow, self.nlay), dtype=np.float64
            )

    if grid.dualporo and flag:
        if fracture:
            self.name = "SOILF" + "_" + str(date)
            self._values[grid._dualactnum.values == 1] = 0.0
        else:
            self.name = "SOILM" + "_" + str(date)
            self._values[grid._dualactnum.values == 2] = 0.0

    gactnum = grid.get_actnum().values
    self._values = ma.masked_where(gactnum < 1, self._values)

    return 2


def _import_eclbinary_meta(self, fhandle, kwlist, etype, date, grid):
    """Find settings and metadata, private to this module.

    Returns:
        A dictionary of metadata

    """
    nentry = 0
    metadata = {}

    datefound = True
    if etype == 5:
        datefound = False
        logger.info("Look for date %s", date)

        # scan for date and find SEQNUM entry number; also potentially update date!
        dtlist = kwlist["DATE"].values.tolist()
        if date == 0:
            date = dtlist[0]
        elif date == 9:
            date = dtlist[-1]

        logger.info("Redefined date is %s", date)

        for ientry, dtentry in enumerate(dtlist):
            if str(dtentry) == str(date):
                datefound = True
                nentry = ientry
                break

        if not datefound:
            msg = "Date {} not found, nentry={}".format(date, nentry)
            xtg.warn(msg)
            raise xtgeo.DateNotFoundError(msg)

    kwxlist = list(kwlist.itertuples(index=False, name=None))
    # INTEHEAD is needed to verify grid dimensions:
    for kwitem in kwxlist:
        if kwitem[0] == "INTEHEAD":
            kwname, kwtype, kwlen, kwbyte, _kwdate = kwitem
            break

    # read INTEHEAD record:
    intehead = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte).tolist()
    ncol, nrow, nlay = intehead[8:11]
    logger.info("Dimensions detected %s %s %s", ncol, nrow, nlay)

    metadata["IPHS"] = intehead[14]  # phase indicator 1:o 2:w 3:ow 4:g 5:og 6:gw 7:owg
    logger.info("Phase system is %s", PHASES[metadata["IPHS"]])

    # LOGIHEAD item [14] in restart should be True, if dualporo model...
    # LOGIHEAD item [15] in restart should be True, if dualperm (+ dualporo) model.
    for kwitem in kwxlist:
        if kwitem[0] == "LOGIHEAD":
            kwname, kwtype, kwlen, kwbyte, _kwdate = kwitem
            break

    # read INTEHEAD record:
    logihead = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte).tolist()

    # DUAL; which kind if doubles (not exact!) the layers when reading,
    # and assign first half* to Matrix (M) and second half to Fractures (F).
    # *half: The number of active cells per half may NOT be equal

    if grid.dualporo:
        logger.info("Dual poro system")
        self._dualporo = True

    if grid.dualperm:
        logger.info("Dual poro + dual perm system")
        self._dualporo = True
        self._dualperm = True

    if etype == 5 and (logihead[13] or logihead[14]) and not grid.dualporo:
        raise RuntimeError("Some inconsistentcy wrt dual porosity model. Bug?")

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    metadata["DATE"] = date
    return metadata


def _import_eclbinary_checks1(self, grid):
    """Do some validations/checks"""

    if self._dualporo:
        logger.info(
            "Grid dimensions in INIT or RESTART file seems: %s %s %s",
            self._ncol,
            self._nrow,
            self._nlay,
        )
        logger.info(
            "Note! This is DUAL POROSITY grid, so actual size: %s %s %s",
            self._ncol,
            self._nrow,
            grid.nlay,
        )
    else:
        logger.info(
            "Grid dimensions in INIT or RESTART file: %s %s %s",
            self._ncol,
            self._nrow,
            self._nlay,
        )

    logger.info(
        "Grid dimensions from GRID file: %s %s %s", grid.ncol, grid.nrow, grid.nlay
    )

    if not grid.dualporo and (
        grid.ncol != self._ncol or grid.nrow != self._nrow or grid.nlay != self._nlay
    ):
        msg = "Errors in dimensions prop: {} {} {} vs grid: {} {} {} ".format(
            self._ncol, self._nrow, self._nlay, grid.ncol, grid.ncol, grid.nlay
        )
        raise RuntimeError(msg)

    if grid.dualporo and (
        grid.ncol != self._ncol
        or grid.nrow != self._nrow
        or grid.nlay * 2 != self._nlay
    ):
        msg = "Errors in dimensions prop: {} {} {} vs grid: {} {} {} ".format(
            self._ncol, self._nrow, self._nlay, grid.ncol, grid.ncol, 2 * grid.nlay
        )
        raise RuntimeError(msg)

    # reset nlay for property when dualporo
    if grid.dualporo:
        self._nlay = grid.nlay


def _import_eclbinary_checks2(kwlist, name, etype, date):
    """More checks, and returns what's needed for actual import"""

    datefound = True

    kwfound = False
    datefoundhere = False
    usedate = "0"
    restart = False

    if etype == 5:
        usedate = str(date)
        restart = True

    kwxlist = list(kwlist.itertuples(index=False, name=None))
    for kwitem in kwxlist:
        kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
        logger.debug("Keyword %s -  date: %s usedate: %s", kwname, kwdate, usedate)
        if name == kwname:
            kwfound = True

        if name == kwname and usedate == str(kwdate):
            logger.info("Keyword %s ok at date %s", name, usedate)
            kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
            datefoundhere = True
            break

    if restart:
        if datefound and not kwfound:
            msg = "Date <{}> is found, but not keyword <{}>".format(date, name)
            xtg.warn(msg)
            raise xtgeo.KeywordNotFoundError(msg)

        if not datefoundhere and kwfound:
            msg = "The keyword <{}> exists but not for " "date <{}>".format(name, date)
            xtg.warn(msg)
            raise xtgeo.KeywordFoundNoDateError(msg)
    else:
        if not kwfound:
            msg = "The keyword <{}> is not found".format(name)
            xtg.warn(msg)
            raise xtgeo.KeywordNotFoundError(msg)

    return kwname, kwlen, kwtype, kwbyte


def _import_eclbinary_prop(
    self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype
):
    """Import the actual record"""

    values = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    self._isdiscrete = False
    use_undef = xtgeo.UNDEF
    self.codes = {}

    if kwtype == "INTE":
        self._isdiscrete = True
        use_undef = xtgeo.UNDEF_INT

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        values = values.astype(np.float64)  # cast REAL (float32) to float64

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping (need both C
    # and F order)
    gactnum = grid.get_actnum().values
    gactindc = grid.actnum_indices
    gactindf = grid.get_actnum_indices(order="F")

    allvalues = (
        np.zeros((self._ncol * self._nrow * self._nlay), dtype=values.dtype) + use_undef
    )

    msg = "\n"
    msg = msg + "grid.actnum_indices.shape[0] = {}\n".format(
        grid.actnum_indices.shape[0]
    )
    msg = msg + "values.shape[0] = {}\n".format(values.shape[0])
    msg = msg + "ncol nrow nlay {} {} {}, nrow*nrow*nlay = {}\n".format(
        self._ncol, self._nrow, self._nlay, self._ncol * self._nrow * self._nlay
    )

    logger.info(msg)

    if gactindc.shape[0] == values.shape[0]:
        allvalues[gactindf] = values
    elif values.shape[0] == self._ncol * self._nrow * self._nlay:  # often case for PORV
        allvalues = values.copy()
    else:

        msg = (
            "BUG somehow... Is the file corrupt? If not contact "
            "the library developer(s)!\n" + msg
        )
        raise SystemExit(msg)

    allvalues = allvalues.reshape((self._ncol, self._nrow, self._nlay), order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(gactnum < 1, allvalues)

    self._values = allvalues

    if etype == 1:
        self._name = name
    else:
        self._name = name + "_" + str(date)
        self._date = date


def _import_eclbinary_dualporo(
    self, grid, fhandle, kwname, kwlen, kwtype, kwbyte, name, date, etype, fracture
):
    """Import the actual record for dual poro scheme"""

    #
    # TODO, merge this with routine above!
    # A lot of code duplication here, as this is under testing
    #

    values = _eclbin.eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping (need both C
    # and F order)

    gactnum = grid.get_actnum().values
    gactindc = grid.get_dualactnum_indices(order="C", fracture=fracture)
    gactindf = grid.get_dualactnum_indices(order="F", fracture=fracture)

    indsize = gactindc.size
    if kwlen == 2 * grid.ntotal:
        indsize = grid.ntotal  # in case of e.g. PORV which is for all cells

    if not fracture:
        values = values[:indsize]
    else:
        values = values[values.size - indsize :]

    self._isdiscrete = False
    self.codes = {}

    if kwtype == "INTE":
        self._isdiscrete = True

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        values = values.astype(np.float64)  # cast REAL (float32) to float64

    allvalues = (
        np.zeros((self._ncol * self._nrow * self._nlay), dtype=values.dtype)
        + self.undef
    )

    if gactindc.shape[0] == values.shape[0]:
        allvalues[gactindf] = values
    elif values.shape[0] == self._ncol * self._nrow * self._nlay:  # often case for PORV
        allvalues = values.copy()
    else:

        msg = (
            "BUG somehow reading binary Eclipse! Is the file corrupt? If not contact "
            "the library developer(s)!\n"
        )
        raise SystemExit(msg)

    allvalues = allvalues.reshape((self._ncol, self._nrow, self.nlay), order="F")
    allvalues = np.asanyarray(allvalues, order="C")

    if self._dualporo:
        if fracture:
            allvalues[grid._dualactnum.values == 1] = 0.0
        else:
            allvalues[grid._dualactnum.values == 2] = 0.0

    allvalues = ma.masked_where(gactnum < 1, allvalues)
    self._values = allvalues

    append = ""
    if self._dualporo:
        append = "M"
        if fracture:
            append = "F"

    logger.info("Dual status is %s, and append is %s", self._dualporo, append)
    if etype == 1:
        self._name = name + append
    else:
        self._name = name + append + "_" + str(date)
        self._date = date
