"""Private module, Grid ETC 1 methods, info/modify/report ...."""

from __future__ import print_function, absolute_import

import inspect
import warnings
import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()

# Note that "self" is the grid instance


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get dZ as propertiy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dz(self, name='dZ', flip=True, mask=True):

    ntot = (self._ncol, self._nrow, self._nlay)

    dz = xtgeo.grid3d.GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=name,
        discrete=False)

    ptr_dz_v = _cxtgeo.new_doublearray(self.ntotal)

    nflip = 1
    if not flip:
        nflip = -1

    option = 0
    if mask:
        option = 1

    _cxtgeo.grd3d_calc_dz(self._ncol, self._nrow, self._nlay, self._p_zcorn_v,
                          self._p_actnum_v, ptr_dz_v, nflip, option,
                          xtg_verbose_level)

    _gridprop_lowlevel.update_values_from_carray(dz, ptr_dz_v, np.float64,
                                                 delete=True)
    # return the property object

    logger.info('DZ mean value: {}'.format(dz.values.mean()))

    return dz


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get dX, dY as properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dxdy(self, names=('dX', 'dY')):

    ntot = self._ncol * self._nrow * self._nlay
    dx = xtgeo.grid3d.GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False)
    dy = xtgeo.grid3d.GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False)

    ptr_dx_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_dy_v = _cxtgeo.new_doublearray(self.ntotal)

    option1 = 0
    option2 = 0

    _cxtgeo.grd3d_calc_dxdy(self._ncol, self._nrow, self._nlay,
                            self._p_coord_v, self._p_zcorn_v, self._p_actnum_v,
                            ptr_dx_v, ptr_dy_v, option1, option2,
                            xtg_verbose_level)

    _gridprop_lowlevel.update_values_from_carray(dx, ptr_dx_v, np.float64,
                                                 delete=True)
    _gridprop_lowlevel.update_values_from_carray(dy, ptr_dy_v, np.float64,
                                                 delete=True)

    # return the property objects
    return dx, dy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get I J K as properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_ijk(self, names=['IX', 'JY', 'KZ'], mask=True, zero_base=False):

    GrProp = xtgeo.grid3d.GridProperty

    actnum = self.get_actnum()

    ashape = (self._ncol, self._nrow, self._nlay)

    ix, jy, kz = np.indices(ashape)

    ix = ix.ravel()
    jy = jy.ravel()
    kz = kz.ravel()

    if mask:
        ix = ma.masked_where(actnum.values1d == 0, ix)
        jy = ma.masked_where(actnum.values1d == 0, jy)
        kz = ma.masked_where(actnum.values1d == 0, kz)

    if not zero_base:
        ix += 1
        jy += 1
        kz += 1

    ix = GrProp(ncol=self._ncol,
                nrow=self._nrow,
                nlay=self._nlay,
                values=ix.reshape(ashape),
                name=names[0],
                discrete=True)
    jy = GrProp(ncol=self._ncol,
                nrow=self._nrow,
                nlay=self._nlay,
                values=jy.reshape(ashape),
                name=names[1],
                discrete=True)
    kz = GrProp(ncol=self._ncol,
                nrow=self._nrow,
                nlay=self._nlay,
                values=kz.reshape(ashape),
                name=names[2],
                discrete=True)

    # return the objects
    return ix, jy, kz


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z as properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz(self, names=['X_UTME', 'Y_UTMN', 'Z_TVDSS'], mask=True):

    ntot = self.ntotal

    GrProp = xtgeo.grid3d.GridProperty

    x = GrProp(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False)

    y = GrProp(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False)

    z = GrProp(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[2],
        discrete=False)

    ptr_x_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_y_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_z_v = _cxtgeo.new_doublearray(self.ntotal)

    option = 0
    if mask:
        option = 1

    _cxtgeo.grd3d_calc_xyz(self._ncol, self._nrow, self._nlay, self._p_coord_v,
                           self._p_zcorn_v, self._p_actnum_v, ptr_x_v, ptr_y_v,
                           ptr_z_v, option, xtg_verbose_level)

    _gridprop_lowlevel.update_values_from_carray(x, ptr_x_v, np.float64,
                                                 delete=True)
    _gridprop_lowlevel.update_values_from_carray(y, ptr_y_v, np.float64,
                                                 delete=True)
    _gridprop_lowlevel.update_values_from_carray(z, ptr_z_v, np.float64,
                                                 delete=True)

    # Note: C arrays are deleted in the update_values_from_carray()

    return x, y, z


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z cell corners for one cell
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz_cell_corners(self, ijk=(1, 1, 1), mask=True, zerobased=False):

    i, j, k = ijk

    shift = 0
    if zerobased:
        shift = 1

    if mask is True:
        actnum = self.get_actnum()
        iact = actnum.values3d[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if iact == 0:
            return None

    pcorners = _cxtgeo.new_doublearray(24)

    _cxtgeo.grd3d_corners(i + shift, j + shift, k + shift, self.ncol,
                          self.nrow, self.nlay, self._p_coord_v,
                          self._p_zcorn_v, pcorners, xtg_verbose_level)

    cornerlist = []
    for i in range(24):
        cornerlist.append(_cxtgeo.doublearray_getitem(pcorners, i))

    clist = tuple(cornerlist)
    return clist


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get X Y Z cell corners for all cells (as 24 GridProperty objects)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_xyz_corners(self, names=('X_UTME', 'Y_UTMN', 'Z_TVDSS')):

    ntot = (self._ncol, self._nrow, self._nlay)

    grid_props = []

    GrProp = xtgeo.grid3d.GridProperty

    for i in range(0, 8):
        xname = names[0] + str(i)
        yname = names[1] + str(i)
        zname = names[2] + str(i)
        x = GrProp(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=xname,
            discrete=False)

        y = GrProp(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=yname,
            discrete=False)

        z = GrProp(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=zname,
            discrete=False)

        grid_props.append(x)
        grid_props.append(y)
        grid_props.append(z)

    ptr_coord = []
    for i in range(24):
        some = _cxtgeo.new_doublearray(self.ntotal)
        ptr_coord.append(some)

    for i, v in enumerate(ptr_coord):
        logger.debug('SWIG object {}   {}'.format(i, v))

    option = 0

    # note, fool the argument list to unpack ptr_coord with * ...
    _cxtgeo.grd3d_get_all_corners(
        self._ncol, self._nrow, self._nlay, self._p_coord_v, self._p_zcorn_v,
        self._p_actnum_v, *(ptr_coord + [option] + [xtg_verbose_level]))

    for i in range(0, 24, 3):

        _gridprop_lowlevel.update_values_from_carray(grid_props[i],
                                                     ptr_coord[i],
                                                     np.float64,
                                                     delete=True)

        _gridprop_lowlevel.update_values_from_carray(grid_props[i + 1],
                                                     ptr_coord[i + 1],
                                                     np.float64,
                                                     delete=True)

        _gridprop_lowlevel.update_values_from_carray(grid_props[i + 2],
                                                     ptr_coord[i + 2],
                                                     np.float64,
                                                     delete=True)

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

    logger.info('Cell geometrics done')

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
def inactivate_by_dz(self, threshold):

    if isinstance(threshold, int):
        threshold = float(threshold)

    if not isinstance(threshold, float):
        raise ValueError('The threshold is not a float or int')

    # assumption (unless somebody finds a Petrel made grid):
    nflip = 1

    _cxtgeo.grd3d_inact_by_dz(self.ncol, self.nrow, self.nlay,
                              self._p_zcorn_v, self._p_actnum_v,
                              threshold, nflip, xtg_verbose_level)

    return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Inactivate inside a polygon (or outside)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def inactivate_inside(self, poly, layer_range=None, inside=True,
                      force_close=False):

    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError('Input polygon not a XTGeo Polygons instance')

    if layer_range is not None:
        k1, k2 = layer_range
    else:
        k1 = 1
        k2 = self.nlay

    method = 0
    if not inside:
        method = 1

    iforce = 0
    if force_close:
        iforce = 1

    # get dataframe where each polygon is ended by a 999 value
    dfxyz = poly.get_xyz_dataframe()

    xc = dfxyz['X_UTME'].values.copy()
    yc = dfxyz['Y_UTMN'].values.copy()

    ier = _cxtgeo.grd3d_inact_outside_pol(xc, yc, self.ncol,
                                          self.nrow,
                                          self.nlay, self._p_coord_v,
                                          self._p_zcorn_v,
                                          self._p_actnum_v, k1, k2,
                                          iforce, method,
                                          xtg_verbose_level)

    if ier == 1:
        raise RuntimeError('Problems with one or more polygons. '
                           'Not closed?')

    return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collapse inactive cells
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def collapse_inactive_cells(self):

    _cxtgeo.grd3d_collapse_inact(self.ncol, self.nrow, self.nlay,
                                 self._p_zcorn_v, self._p_actnum_v,
                                 xtg_verbose_level)
    return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Do cropping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def do_cropping(self, spec):
    """Do cropping of geometry (and params)."""

    (ic1, ic2), (jc1, jc2), (kc1, kc2) = spec

    if ic1 < 1 or ic2 > self.ncol or jc1 < 1 or jc2 > self.nrow or \
       kc1 < 1 or kc2 > self.nlay:

        raise ValueError('Boundary for tuples not matching grid'
                         'NCOL, NROW, NLAY')

    # compute size of new cropped grid
    nncol = ic2 - ic1 + 1
    nnrow = jc2 - jc1 + 1
    nnlay = kc2 - kc1 + 1

    ntot = nncol * nnrow * nnlay
    ncoord = (nncol + 1) * (nnrow + 1) * 2 * 3
    nzcorn = nncol * nnrow * (nnlay + 1) * 4

    new_num_act = _cxtgeo.new_intpointer()
    new_p_coord_v = _cxtgeo.new_doublearray(ncoord)
    new_p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    new_p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.grd3d_crop_geometry(self.ncol, self.nrow, self.nlay,
                                self._p_coord_v,
                                self._p_zcorn_v,
                                self._p_actnum_v,
                                new_p_coord_v,
                                new_p_zcorn_v,
                                new_p_actnum_v,
                                ic1, ic2, jc1, jc2, kc1, kc2,
                                new_num_act,
                                0,
                                xtg_verbose_level)

    self._p_coord_v = new_p_coord_v
    self._p_zcorn_v = new_p_zcorn_v
    self._p_actnum_v = new_p_actnum_v

    self._nactive = _cxtgeo.intpointer_value(new_num_act)
    self._ncol = nncol
    self._nrow = nnrow
    self._nlay = nnlay

    # TODO: subgrid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reduce grid to one layer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reduce_to_one_layer(self):
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
    ptr_new_zcorn_v = _cxtgeo.new_doublearray(self.ncol * self.nrow * nnum)

    ptr_new_actnum_v = _cxtgeo.new_intarray(self.ncol * self.nrow * 1)

    _cxtgeo.grd3d_reduce_onelayer(self.ncol, self.nrow, self.nlay,
                                  self._p_zcorn_v,
                                  ptr_new_zcorn_v,
                                  self._p_actnum_v,
                                  ptr_new_actnum_v,
                                  ptr_new_num_act,
                                  0,
                                  xtg_verbose_level)

    self._nlay = 1
    self._p_zcorn_v = ptr_new_zcorn_v
    self._p_actnum_v = ptr_new_actnum_v
    self._nactive = _cxtgeo.intpointer_value(ptr_new_num_act)
    self._props = []
    self._subgrids = None

    return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Translate coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):

    tx, ty, tz = translate
    fx, fy, fz = flip

    ier = _cxtgeo.grd3d_translate(self._ncol, self._nrow, self._nlay,
                                  fx, fy, fz, tx, ty, tz,
                                  self._p_coord_v, self._p_zcorn_v,
                                  xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Something went wrong in translate, code: {}'
                           .format(ier))

    logger.info('Translation of coords done')

    return self


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reports well to zone mismatch
# This works together with a Well object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def report_zone_mismatch(self, well=None, zonelogname='ZONELOG',
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

    # get the relevant well log C arrays...
    ptr_xc = well.get_carray('X_UTME')
    ptr_yc = well.get_carray('Y_UTMN')
    ptr_zc = well.get_carray('Z_TVDSS')
    ptr_zo = well.get_carray(zonelogname)

    nval = well.nrow

    ptr_results = _cxtgeo.new_doublearray(10)

    ptr_zprop = _gridprop_lowlevel.update_carray(zoneprop)

    cstatus = _cxtgeo.grd3d_rpt_zlog_vs_zon(self._ncol, self._nrow,
                                            self._nlay, self._p_coord_v,
                                            self._p_zcorn_v,
                                            self._p_actnum_v, ptr_zprop,
                                            nval, ptr_xc, ptr_yc, ptr_zc,
                                            ptr_zo, zonelogrange[0],
                                            zonelogrange[1],
                                            onelayergrid._p_zcorn_v,
                                            onelayergrid._p_actnum_v,
                                            ptr_results, option,
                                            xtg_verbose_level)

    _gridprop_lowlevel.delete_carray(zoneprop, ptr_zprop)

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

    # returns percent match, then total numbers of well counts for zone,
    # then match count. perc = mpoi/tpoi
    return (perc, int(tpoi), int(mpoi))
