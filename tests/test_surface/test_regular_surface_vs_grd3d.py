from os.path import join

import numpy as np

import tests.test_common.test_xtg as tsetup
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =============================================================================
# Do tests
# =============================================================================

RPATH1 = TPATH / "surfaces/reek"
RPATH2 = TPATH / "3dgrids/reek"

RTOP1 = RPATH1 / "1/topreek_rota.gri"
RGRD1 = RPATH2 / "REEK.EGRID"
RPROP1 = RPATH2 / "REEK.INIT"
RGRD2 = RPATH2 / "reek_sim_grid.roff"
RPROP2 = RPATH2 / "reek_sim_zone.roff"


@tsetup.plotskipifroxar
@tsetup.skipifmac  # segm fault; need to be investigated
def test_get_surface_from_grd3d_porosity(tmpdir):
    """Sample a surface from a 3D grid"""

    surf = xtgeo.surface.RegularSurface(RTOP1)
    print(surf.values.min(), surf.values.max())
    grd = xtgeo.grid3d.Grid(RGRD1, fformat="egrid")
    surf.values = 1700
    zsurf = surf.copy()
    surfr = surf.copy()
    surf2 = surf.copy()
    phi = xtgeo.grid3d.GridProperty(RPROP1, fformat="init", name="PORO", grid=grd)

    # slice grd3d
    surf.slice_grid3d(grd, phi)

    surf.to_file(join(tmpdir, "surf_slice_grd3d_reek.gri"))
    surf.quickplot(filename=join(tmpdir, "surf_slice_grd3d_reek.png"))

    # refined version:
    surfr.refine(2)
    surfr.slice_grid3d(grd, phi)

    surfr.to_file(join(tmpdir, "surf_slice_grd3d_reek_refined.gri"))
    surfr.quickplot(filename=join(tmpdir, "surf_slice_grd3d_reek_refined.png"))

    # use zsurf:
    surf2.slice_grid3d(grd, phi, zsurf=zsurf)

    surf2.to_file(join(tmpdir, "surf_slice_grd3d_reek_zslice.gri"))
    surf2.quickplot(filename=join(tmpdir, "surf_slice_grd3d_reek_zslice.png"))

    assert np.allclose(surf.values, surf2.values)

    tsetup.assert_almostequal(surf.values.mean(), 0.1667, 0.01)
    tsetup.assert_almostequal(surfr.values.mean(), 0.1667, 0.01)


@tsetup.plotskipifroxar
def test_get_surface_from_grd3d_zones(tmpdir):
    """Sample a surface from a 3D grid, using zones"""

    surf = xtgeo.surface.RegularSurface(RTOP1)
    grd = xtgeo.grid3d.Grid(RGRD2, fformat="roff")
    surf.values = 1700
    zone = xtgeo.grid3d.GridProperty(RPROP2, fformat="roff", name="Zone", grid=grd)

    # slice grd3d
    surf.slice_grid3d(grd, zone, sbuffer=1)

    surf.to_file(join(tmpdir, "surf_slice_grd3d_reek_zone.gri"))
    surf.quickplot(filename=join(tmpdir, "surf_slice_grd3d_reek_zone.png"))


@tsetup.plotskipifroxar
def test_surface_from_grd3d_layer(tmpdir):
    """Create a surface from a 3D grid layer"""

    surf = xtgeo.surface.RegularSurface()
    grd = xtgeo.grid3d.Grid(RGRD2, fformat="roff")
    # grd = xtgeo.Grid("../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg.roff")
    surf.from_grid3d(grd)

    surf.fill()
    surf.to_file(join(tmpdir, "surf_from_grid3d_top.gri"))
    tmp = surf.copy()
    surf.quickplot(filename=join(tmpdir, "surf_from_grid3d_top.png"))

    surf.from_grid3d(grd, template=tmp, mode="i")

    surf.to_file(join(tmpdir, "surf_from_grid3d_top_icell.gri"))
    surf.quickplot(filename=join(tmpdir, "surf_from_grid3d_top_icell.png"))

    surf.from_grid3d(grd, template=tmp, mode="j")
    surf.fill()
    surf.to_file(join(tmpdir, "surf_from_grid3d_top_jcell.gri"))
    surf.quickplot(filename=join(tmpdir, "surf_from_grid3d_top_jcell.png"))

    surf.from_grid3d(grd, template=tmp, mode="depth", where="3_base")
    surf.to_file(join(tmpdir, "surf_from_grid3d_3base.gri"))
    surf.quickplot(filename=join(tmpdir, "surf_from_grid3d_3base.png"))
