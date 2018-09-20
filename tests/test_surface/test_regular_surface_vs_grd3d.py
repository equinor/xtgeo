from os.path import join as ojn

import xtgeo
from xtgeo.common import XTGeoDialog

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


def test_get_surface_from_grd3d():
    """Construct a constant surface from cube."""

    surf = xtgeo.surface.RegularSurface(rtop1)
    print(surf.values.min(), surf.values.max())
    grd = xtgeo.grid3d.Grid(rgrd1, fformat='egrid')
    surf.values = 1700
    phi = xtgeo.grid3d.GridProperty(rprop1, fformat='init', name='PORO',
                                    grid=grd)

    # slice grd3d
    surf.slice_grid3d(phi)

    surf.to_file(ojn(td, 'surf_slice_grd3d_reek.gri'))
    surf.quickplot(filename=ojn(td, 'surf_slice_grd3d_reek.png'))
