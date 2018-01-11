"""Private module, Grid ETC 1 methods, info/modify/report ...."""

from __future__ import print_function, absolute_import

import inspect
import warnings
import numpy as np

import cxtgeo.cxtgeo as _cxtgeo
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get dZ as propertiy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dz(grid, name='dZ', flip=True, mask=True):

    ntot = grid._ncol * grid._nrow * grid._nlay

    dz = xtgeo.grid3d.GridProperty(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=name,
        discrete=False)

    ptr_dz_v = _cxtgeo.new_doublearray(grid.ntotal)

    nflip = 1
    if not flip:
        nflip = -1

    option = 0
    if mask:
        option = 1

    _cxtgeo.grd3d_calc_dz(grid._ncol, grid._nrow, grid._nlay, grid._p_zcorn_v,
                          grid._p_actnum_v, ptr_dz_v, nflip, option,
                          xtg_verbose_level)

    dz._cvalues = ptr_dz_v
    dz._update_values()

    # return the property object
    return dz


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get dX, dY as properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dxdy(grid, names=('dX', 'dY')):

    ntot = grid._ncol * grid._nrow * grid._nlay
    dx = xtgeo.grid3d.GridProperty(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False)
    dy = xtgeo.grid3d.GridProperty(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False)

    ptr_dx_v = _cxtgeo.new_doublearray(grid.ntotal)
    ptr_dy_v = _cxtgeo.new_doublearray(grid.ntotal)

    option1 = 0
    option2 = 0

    _cxtgeo.grd3d_calc_dxdy(grid._ncol, grid._nrow, grid._nlay,
                            grid._p_coord_v, grid._p_zcorn_v, grid._p_actnum_v,
                            ptr_dx_v, ptr_dy_v, option1, option2,
                            xtg_verbose_level)

    dx._cvalues = ptr_dx_v
    dx._update_values()

    dy._cvalues = ptr_dy_v
    dy._update_values()

    # return the property objects
    return dx, dy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z as properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz(grid, names=['X', 'Y', 'Z'], mask=True):

    ntot = grid.ntotal

    GrProp = xtgeo.grid3d.GridProperty

    x = GrProp(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False)

    y = GrProp(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False)

    z = GrProp(
        ncol=grid._ncol,
        nrow=grid._nrow,
        nlay=grid._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[2],
        discrete=False)

    ptr_x_v = _cxtgeo.new_doublearray(grid.ntotal)
    ptr_y_v = _cxtgeo.new_doublearray(grid.ntotal)
    ptr_z_v = _cxtgeo.new_doublearray(grid.ntotal)

    option = 0
    if mask:
        option = 1

    _cxtgeo.grd3d_calc_xyz(grid._ncol, grid._nrow, grid._nlay, grid._p_coord_v,
                           grid._p_zcorn_v, grid._p_actnum_v, ptr_x_v, ptr_y_v,
                           ptr_z_v, option, xtg_verbose_level)

    x._cvalues = ptr_x_v
    y._cvalues = ptr_y_v
    z._cvalues = ptr_z_v

    x._update_values()
    y._update_values()
    z._update_values()

    # return the objects
    return x, y, z


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z cell corners for one cell
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz_cell_corners(grid, ijk=(1, 1, 1), mask=True, zerobased=False):

    i, j, k = ijk

    shift = 0
    if zerobased:
        shift = 1

    if mask is True:
        actnum = grid.get_actnum()
        iact = actnum.values3d[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if iact == 0:
            return None

    pcorners = _cxtgeo.new_doublearray(24)

    _cxtgeo.grd3d_corners(i + shift, j + shift, k + shift, grid.ncol,
                          grid.nrow, grid.nlay, grid._p_coord_v,
                          grid._p_zcorn_v, pcorners, xtg_verbose_level)

    cornerlist = []
    for i in range(24):
        cornerlist.append(_cxtgeo.doublearray_getitem(pcorners, i))

    clist = tuple(cornerlist)
    return clist


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z cell corners for all cells (as 24 GridProperty objects)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz_corners(grid, names=('X', 'Y', 'Z')):

    ntot = grid.ntotal

    grid_props = []

    GrProp = xtgeo.grid3d.GridProperty

    for i in range(0, 8):
        xname = names[0] + str(i)
        yname = names[1] + str(i)
        zname = names[2] + str(i)
        x = GrProp(
            ncol=grid._ncol,
            nrow=grid._nrow,
            nlay=grid._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=xname,
            discrete=False)

        y = GrProp(
            ncol=grid._ncol,
            nrow=grid._nrow,
            nlay=grid._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=yname,
            discrete=False)

        z = GrProp(
            ncol=grid._ncol,
            nrow=grid._nrow,
            nlay=grid._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=zname,
            discrete=False)

        grid_props.append(x)
        grid_props.append(y)
        grid_props.append(z)

    ptr_coord = []
    for i in range(24):
        some = _cxtgeo.new_doublearray(grid.ntotal)
        ptr_coord.append(some)

    for i, v in enumerate(ptr_coord):
        grid.logger.debug('SWIG object {}   {}'.format(i, v))

    option = 0

    # note, fool the argument list to unpack ptr_coord with * ...
    _cxtgeo.grd3d_get_all_corners(
        grid._ncol, grid._nrow, grid._nlay, grid._p_coord_v, grid._p_zcorn_v,
        grid._p_actnum_v, *(ptr_coord + [option] + [xtg_verbose_level]))

    for i in range(0, 24, 3):
        grid_props[i]._cvalues = ptr_coord[i]
        grid_props[i + 1]._cvalues = ptr_coord[i + 1]
        grid_props[i + 2]._cvalues = ptr_coord[i + 2]

        grid_props[i]._update_values()
        grid_props[i + 1]._update_values()
        grid_props[i + 2]._update_values()

    # return the 24 objects (x1, y1, z1, ... x8, y8, z8)
    return tuple(grid_props)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get grid geometrics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_geometrics(self, allcells=False, cellcenter=True, return_dict=False):

    ptr_x = []
    for i in range(13):
        ptr_x.append(_cxtgeo.new_doublepointer())

    option1 = 1
    if allcells:
        option1 = 0

    option2 = 1
    if not cellcenter:
        option2 = 0

    quality = _cxtgeo.grd3d_geometrics(
        self._ncol, self._nrow, self._nlay, self._p_coord_v,
        self._p_zcorn_v, self._p_actnum_v, ptr_x[0], ptr_x[1], ptr_x[2],
        ptr_x[3], ptr_x[4], ptr_x[5], ptr_x[6], ptr_x[7], ptr_x[8],
        ptr_x[9], ptr_x[10], ptr_x[11], ptr_x[12], option1, option2,
        xtg_verbose_level)

    glist = []
    for i in range(13):
        glist.append(_cxtgeo.doublepointer_value(ptr_x[i]))

    glist.append(quality)

    self.logger.info('Cell geometrics done')

    if return_dict:
        gdict = {}
        gkeys = ['xori', 'yori', 'zori', 'xmin', 'xmax', 'ymin', 'ymax',
                 'zmin', 'zmax', 'avg_rotation', 'avg_dx', 'avg_dy', 'avg_dz',
                 'grid_regularity_flag']

        for i, key in enumerate(gkeys):
            gdict[key] = glist[i]

        return gdict
    else:
        return tuple(glist)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Inactivate by DZ
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def inactivate_by_dz(grid, threshold):

    if isinstance(threshold, int):
        threshold = float(threshold)

    if not isinstance(threshold, float):
        raise ValueError('The threshold is not a float or int')

    # assumption (unless somebody finds a Petrel made grid):
    nflip = 1

    _cxtgeo.grd3d_inact_by_dz(grid.ncol, grid.nrow, grid.nlay,
                              grid._p_zcorn_v, grid._p_actnum_v,
                              threshold, nflip, xtg_verbose_level)

    return grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Inactivate inside a polygon (or outside)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def inactivate_inside(grid, poly, layer_range=None, inside=True,
                      force_close=False):

    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError('Input polygon not a XTGeo Polygons instance')

    if layer_range is not None:
        k1, k2 = layer_range
    else:
        k1 = 1
        k2 = grid.nlay

    method = 0
    if not inside:
        method = 1

    iforce = 0
    if force_close:
        iforce = 1

    # get dataframe where each polygon is ended by a 999 value
    dfxyz = poly.get_xyz_dataframe()

    xc = dfxyz['X'].values
    yc = dfxyz['Y'].values

    ier = _cxtgeo.grd3d_inact_outside_pol(xc, yc, grid.ncol,
                                          grid.nrow,
                                          grid.nlay, grid._p_coord_v,
                                          grid._p_zcorn_v,
                                          grid._p_actnum_v, k1, k2,
                                          iforce, method,
                                          xtg_verbose_level)

    if ier == 1:
        raise RuntimeError('Problems with one or more polygons. '
                           'Not closed?')

    return grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collapse inactive cells
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def collapse_inactive_cells(grid):

    _cxtgeo.grd3d_collapse_inact(grid.ncol, grid.nrow, grid.nlay,
                                 grid._p_zcorn_v, grid._p_actnum_v,
                                 xtg_verbose_level)
    return grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reduce grid to one layer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reduce_to_one_layer(grid):
    """Reduce the grid to one single single layer.

    Example::

        >>> from xtgeo.grid3d import Grid
        >>> gf = Grid('gullfaks2.roff')
        >>> gf.nlay
        47
        >>> gf.reduce_to_one_layer()
        >>> gf.nlay
        1

    """

    # need new pointers in C (not for coord)

    ptr_new_num_act = _cxtgeo.new_intpointer()

    nnum = (1 + 1) * 4
    ptr_new_zcorn_v = _cxtgeo.new_doublearray(grid.ncol * grid.nrow * nnum)

    ptr_new_actnum_v = _cxtgeo.new_intarray(grid.ncol * grid.nrow * 1)

    _cxtgeo.grd3d_reduce_onelayer(grid.ncol, grid.nrow, grid.nlay,
                                  grid._p_zcorn_v,
                                  ptr_new_zcorn_v,
                                  grid._p_actnum_v,
                                  ptr_new_actnum_v,
                                  ptr_new_num_act,
                                  0,
                                  xtg_verbose_level)

    grid._nlay = 1
    grid._p_zcorn_v = ptr_new_zcorn_v
    grid._p_actnum_v = ptr_new_actnum_v
    grid._nactive = _cxtgeo.intpointer_value(ptr_new_num_act)
    grid._nsubs = 0
    grid._props = []

    return grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Translate coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def translate_coordinates(grid, translate=(0, 0, 0), flip=(1, 1, 1)):

    tx, ty, tz = translate
    fx, fy, fz = flip

    ier = _cxtgeo.grd3d_translate(grid._ncol, grid._nrow, grid._nlay,
                                  fx, fy, fz, tx, ty, tz,
                                  grid._p_coord_v, grid._p_zcorn_v,
                                  xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Something went wrong in translate, code: {}'
                           .format(ier))

    logger.info('Translation of coords done')

    return grid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reports well to zone mismatch
# This works together with a Well object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def report_zone_mismatch(grid, well=None, zonelogname='ZONELOG',
                         mode=0, zoneprop=None, onelayergrid=None,
                         zonelogrange=(0, 9999), zonelogshift=0,
                         depthrange=None, option=0, perflogname=None):

    this = inspect.currentframe().f_code.co_name

    # first do some trimming of the well dataframe
    if not well or not isinstance(well, xtgeo.well.Well):
        msg = ('No well object in <{}> or invalid object; '
               'returns no result'.format(this))
        warnings.warn(UserWarning(msg))
        return None

    # qperf = True
    if perflogname == 'None' or perflogname is None:
        # qperf = False
        pass
    else:
        if perflogname not in well.lognames:
            msg = ('Asked for perf log <{}> but no such in <{}> for well {}; '
                   'return None'.format(perflogname, this, well.wellname))
            warnings.warn(UserWarning(msg))
            return None

    logger.info('Process well object for {}...'.format(well.wellname))
    df = well.dataframe.copy()

    if depthrange:
        logger.info('Filter depth...')
        df = df[df.Z_TVDSS > depthrange[0]]
        df = df[df.Z_TVDSS < depthrange[1]]
        df = df.copy()
        logger.debug(df)

    logger.info('Adding zoneshift {}'.format(zonelogshift))
    if zonelogshift != 0:
        df[zonelogname] += zonelogshift

    logger.info('Filter ZONELOG...')
    df = df[df[zonelogname] > zonelogrange[0]]
    df = df[df[zonelogname] < zonelogrange[1]]
    df = df.copy()

    if perflogname:
        logger.info('Filter PERF...')
        df[perflogname].fillna(-999, inplace=True)
        df = df[df[perflogname] > 0]
        df = df.copy()

    df.reset_index(drop=True, inplace=True)
    well.dataframe = df

    logger.debug(df)

    # get the relevant well log C arrays...
    ptr_xc = well.get_carray('X_UTME')
    ptr_yc = well.get_carray('Y_UTMN')
    ptr_zc = well.get_carray('Z_TVDSS')
    ptr_zo = well.get_carray(zonelogname)

    nval = well.nrow

    ptr_results = _cxtgeo.new_doublearray(10)

    ptr_zprop = zoneprop.cvalues

    cstatus = _cxtgeo.grd3d_rpt_zlog_vs_zon(grid._ncol, grid._nrow,
                                            grid._nlay, grid._p_coord_v,
                                            grid._p_zcorn_v,
                                            grid._p_actnum_v, ptr_zprop,
                                            nval, ptr_xc, ptr_yc, ptr_zc,
                                            ptr_zo, zonelogrange[0],
                                            zonelogrange[1],
                                            onelayergrid._p_zcorn_v,
                                            onelayergrid._p_actnum_v,
                                            ptr_results, option,
                                            xtg_verbose_level)

    if cstatus == 0:
        logger.debug('OK well')
    elif cstatus == 2:
        msg = ('Well {} have no zonation?'.format(well.wellname))
        warnings.warn(msg, UserWarning)
    else:
        msg = ('Something is rotten with {}'.format(well.wellname))
        raise SystemExit(msg)

    # extract the report
    perc = _cxtgeo.doublearray_getitem(ptr_results, 0)
    tpoi = _cxtgeo.doublearray_getitem(ptr_results, 1)
    mpoi = _cxtgeo.doublearray_getitem(ptr_results, 2)

    return (perc, tpoi, mpoi)
