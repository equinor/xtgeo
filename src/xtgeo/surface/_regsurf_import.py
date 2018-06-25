"""Import RegularSurface data."""
import numpy as np
import numpy.ma as ma
import pandas as pd

import xtgeo.common.calc as xcalc
import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')


def import_irap_binary(self, mfile):
    # using swig type mapping

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    # read with mode 0, to get mx my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 0, 1, 0, xtg_verbose_level)

    nval = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 1, nval, 0, xtg_verbose_level)

    ier, ncol, nrow, ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        raise RuntimeError('Problem in {}, code {}'.format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order='C')

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info('NaN values are found, will mask...')
        val = ma.masked_invalid(val)

    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
    self._xinc = xinc
    self._yinc = yinc
    self._rotation = rot
    self._values = val
    self._filesrc = mfile


def import_irap_ascii(self, mfile):
    # version using swig type mapping

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    # read with mode 0, scan to get mx my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 0, 1, 0, xtg_verbose_level)

    nv = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 1, nv, 0, xtg_verbose_level)

    ier, ncol, nrow, ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        raise RuntimeError('Problem in {}, code {}'.format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order='C')

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info('NaN values are found, will mask...')
        val = ma.masked_invalid(val)

    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
    self._xinc = xinc
    self._yinc = yinc
    self._rotation = rot
    self._values = val
    self._filesrc = mfile


def import_ijxyz_ascii(self, mfile):
    # import of seismic column system on the form:
    # 2588	1179	476782.2897888889	6564025.6954	1000.0
    # 2588	1180	476776.7181777778	6564014.5058	1000.0

    # read in as pandas dataframe, and extract info
    heading = ['ILINE', 'XLINE', 'X_UTME', 'Y_UTMN', 'MAPVALUES']

    df = pd.read_csv(mfile, delim_whitespace=True,
                     header=None, names=heading)

    # compute geometrics
    xori = df.X_UTME.iloc[0]
    yori = df.Y_UTMN.iloc[0]

    dfifilter = df[df.ILINE == df.ILINE[0]]
    xtmp1 = dfifilter.X_UTME.iloc[1]
    ytmp1 = dfifilter.Y_UTMN.iloc[1]

    dfxfilter = df[df.XLINE == df.XLINE[0]]
    xtmp2 = dfxfilter.X_UTME.iloc[1]
    ytmp2 = dfxfilter.Y_UTMN.iloc[1]

    yinc, xa_radian, xa_degrees = xcalc.vectorinfo2(xori, xtmp1, yori, ytmp1)

    xinc, xa_radian, rot = xcalc.vectorinfo2(xori, xtmp2, yori, ytmp2)

    flip = xcalc.find_flip((xtmp2 - xori, ytmp2 - yori, 0),
                           (xtmp1 - xori, ytmp1 - xori, 0),
                           (0, 0, -1))
    logger.info('Flip is {}'.format(flip))

    self._xori = xori
    self._xinc = xinc
    self._yori = yori
    self._yinc = yinc
    self._nrow = len(dfifilter)
    self._ncol = len(dfxfilter)
    self._rotation = rot
    self._yflip = flip

    self._ilines = dfxfilter.ILINE.values.astype(np.int32)
    self._xlines = dfifilter.XLINE.values.astype(np.int32)

    val = df.MAPVALUES.values.copy()

    self._values = val.reshape((self.nrow, self.ncol))
    self._filesrc = mfile

    if self._yflip == -1:
        self.swapaxes()

    del df
    del dfifilter
    del dfxfilter
