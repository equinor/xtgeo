from os.path import join as ojn

import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================

rpath1 = '../xtgeo-testdata/surfaces/reek'
rpath2 = '../xtgeo-testdata/3dgrids/reek'

rtop1 = ojn(rpath1, '1/topreek_rota.gri')
rgrd1 = ojn(rpath2, 'REEK.EGRID')
rprop1 = ojn(rpath2, 'REEK.INIT')
rgrd2 = ojn(rpath2, 'reek_sim_grid.roff')
rprop2 = ojn(rpath2, 'reek_sim_zone.roff')


@tsetup.plotskipifroxar
def test_get_surface_from_grd3d_porosity():
    """Sample a surface from a 3D grid"""

    surf = xtgeo.surface.RegularSurface(rtop1)
    print(surf.values.min(), surf.values.max())
    grd = xtgeo.grid3d.Grid(rgrd1, fformat='egrid')
    surf.values = 1700
    zsurf = surf.copy()
    surfr = surf.copy()
    surf2 = surf.copy()
    phi = xtgeo.grid3d.GridProperty(rprop1, fformat='init', name='PORO',
                                    grid=grd)

    # slice grd3d
    surf.slice_grid3d(grd, phi)

    surf.to_file(ojn(td, 'surf_slice_grd3d_reek.gri'))
    surf.quickplot(filename=ojn(td, 'surf_slice_grd3d_reek.png'))

    # refined version:
    surfr.refine(2)
    surfr.slice_grid3d(grd, phi)

    surfr.to_file(ojn(td, 'surf_slice_grd3d_reek_refined.gri'))
    surfr.quickplot(filename=ojn(td, 'surf_slice_grd3d_reek_refined.png'))

    # use zsurf:
    surf2.slice_grid3d(grd, phi, zsurf=zsurf)

    surf2.to_file(ojn(td, 'surf_slice_grd3d_reek_zslice.gri'))
    surf2.quickplot(filename=ojn(td, 'surf_slice_grd3d_reek_zslice.png'))

    assert np.allclose(surf.values, surf2.values)

    tsetup.assert_almostequal(surf.values.mean(), 0.1667, 0.01)
    tsetup.assert_almostequal(surfr.values.mean(), 0.1667, 0.01)


@tsetup.plotskipifroxar
def test_get_surface_from_grd3d_zones():
    """Sample a surface from a 3D grid, using zones"""

    surf = xtgeo.surface.RegularSurface(rtop1)
    grd = xtgeo.grid3d.Grid(rgrd2, fformat='roff')
    surf.values = 1700
    zone = xtgeo.grid3d.GridProperty(rprop2, fformat='roff', name='Zone',
                                     grid=grd)

    # slice grd3d
    surf.slice_grid3d(grd, zone, sbuffer=1)

    surf.to_file(ojn(td, 'surf_slice_grd3d_reek_zone.gri'))
    surf.quickplot(filename=ojn(td, 'surf_slice_grd3d_reek_zone.png'))
