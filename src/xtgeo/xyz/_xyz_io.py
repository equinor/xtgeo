# -*- coding: utf-8 -*-
"""Private import and export routines"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)

# -------------------------------------------------------------------------
# Import/Export methods for various formats
# -------------------------------------------------------------------------


def import_xyz(xyz, pfile):

    # Simple X Y Z file. All points as Pandas framework

    xyz._df = pd.read_csv(pfile, delim_whitespace=True, skiprows=0,
                          header=None, names=['X_UTME', 'Y_UTMN', 'Z_TVDSS'],
                          dtype=np.float64, na_values=999.00)

    xyz.logger.debug(xyz._df.head())


def import_zmap(xyz, pfile):

    # the zmap ascii polygon format; not sure about all details;
    # seems that I just
    # take in the columns starting from @<blank> line as is.
    # Potential here to improve...

    #
    # !
    # ! File exported from RMS.
    # !
    # ! Project:
    # ! Date:          2017-11-07T17:22:30
    # !
    # ! Polygons/points Z-MAP file generated for ''.
    # ! Coordinate system is ''.
    # !
    # !------------------------------------------------------------------
    # @FREE POINT        , DATA, 80, 1
    # X (EASTING)        , 1, 1,  1,      1, 20,,    1.0E+30,,,   4, 0
    # Y (NORTHING)       , 2, 2,  1,     21, 40,,    1.0E+30,,,   4, 0
    # Z VALUE            , 3, 3,  1,     41, 60,,    1.0E+30,,,   4, 0
    # SEG I.D.           , 4, 35, 1,     61, 70,,    1.0E+30,,,   0, 0
    # @
    #    457357.781250      6782685.500000      1744.463379         0
    #    457359.343750      6782676.000000      1744.482056         0
    #    457370.906250      6782606.000000      1744.619507         0
    #    457370.468750      6782568.500000      1745.868286         0

    dtype = {'X_UTME': np.float64, 'Y_UTMN': np.float64, 'Z_TVDSS': np.float64,
             'ID': np.int32}

    xyz._df = pd.read_csv(pfile, delim_whitespace=True, skiprows=16,
                          header=None,
                          names=['X_UTME', 'Y_UTMN', 'Z_TVDSS', 'ID'],
                          dtype=dtype, na_values=1.0E+30)

    logger.debug(xyz._df.head())


def export_rms_attr(xyz, pfile, attributes=None, filter=None):
    """Export til RMS attribute, also called RMS extended set.

    If attributes is None, then it will be a simple XYZ file.

    Filter is on the form {TopName: ['Name1', 'Name2']}

    Returns:
        The number of values exported. If value is 0; then no file
        is made.
    """

    df = xyz.dataframe.copy()
    columns = ['X_UTME', 'Y_UTMN', 'Z_TVDSS']
    df.fillna(value=999.0, inplace=True)

    mode = 'w'

    # apply filter if any
    if filter:
        for key, val in filter.items():
            if key in df.columns:
                df = df.loc[df[key].isin(val)]
            else:
                raise KeyError('The requested filter key {} was not '
                               'found in dataframe. Valid keys are '
                               '{}'.format(key, df.columns))

    if len(df.index) < 1:
        logger.warning('Nothing to export')
        return 0

    if attributes is None and 'ID' in df.columns and xyz._ispolygons:
        # need to convert the dataframe
        df = _convert_idbased_xyz(df)

    if attributes is not None:
        mode = 'a'
        columns += attributes
        with open(pfile, 'w') as fout:
            for col in attributes:
                if col in df.columns:
                    fout.write('String ' + col + '\n')

    with open(pfile, mode) as f:
        df.to_csv(f, sep=' ', header=None,
                  columns=columns, index=False, float_format='%.3f')

    return len(df.index)


def _convert_idbased_xyz(df):

    # If polygons, there is a 4th column with ID. This needs
    # to replaced by adding 999 line instead (for polygons)
    # prior to XYZ export or when interactions in CXTGEO

    idgroups = df.groupby('ID')

    newdf = pd.DataFrame(columns=['X_UTME', 'Y_UTMN', 'Z_TVDSS'])
    udef = pd.DataFrame([[999.0, 999.0, 999.0]], columns=['X_UTME',
                                                          'Y_UTMN',
                                                          'Z_TVDSS'])

    for id_, gr in idgroups:
        dfx = gr.drop('ID', axis=1)
        newdf = newdf.append([dfx, udef], ignore_index=True)

    return newdf


def export_rms_wpicks(xyz, pfile, hcolumn, wcolumn, mdcolumn=None):
    """Export til RMS wellpicks

    If a MD column (mdcolumn) exists, it will use the MD

    Args:
        pfile (str): File to export to
        hcolumn (str): Name of horizon/zone column in the point set
        wcolumn (str): Name of well column in the point set
        mdcolumn (str): Name of measured depht column (if any)
    Returns:
        The number of values exported. If value is 0; then no file
        is made.

    """

    df = xyz.dataframe.copy()

    print(df)

    columns = []

    if hcolumn in df.columns:
        columns.append(hcolumn)
    else:
        raise ValueError('Column for horizons/zones <{}> '
                         'not present'.format(hcolumn))

    if wcolumn in df.columns:
        columns.append(wcolumn)
    else:
        raise ValueError('Column for wells <{}> '
                         'not present'.format(wcolumn))

    if mdcolumn in df.columns:
        columns.append(mdcolumn)
    else:
        columns += ['X_UTME', 'Y_UTMN', 'Z_TVDSS']

    print(df)
    print(columns)

    if len(df.index) < 1:
        logger.warning('Nothing to export')
        return 0

    with open(pfile, 'w') as f:
        df.to_csv(f, sep=' ', header=None,
                  columns=columns, index=False)

    return len(df.index)
