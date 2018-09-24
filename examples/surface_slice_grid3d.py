"""
Slice a 3Grid property with a surface, e.g. a FLW map.

In this case 3 maps with constant depth are applied. The maps are refined
for smoother result, and output is exported as Roxar binary *.gri and
quickplots (png)

JRIV
"""

from os.path import join as ojn

import xtgeo

expath1 = '../../xtgeo-testdata/3dgrids/reek'
expath2 = '../../xtgeo-testdata/surfaces/reek/1'

gridfileroot = ojn(expath1, 'REEK')
surfacefile = ojn(expath2, 'midreek_rota.gri')

initprops = ['PORO', 'PERMX']

grd = xtgeo.grid.Grid()
grd.from_file(gridfileroot, fformat='eclipserun', initprops=initprops)

# read a surface, which is used for "template"
surf = xtgeo.surface_from_file(surfacefile)
surf.refine(4)  # make finer for nicer sampling (NB takes time then)

slices = [1700, 1720, 1740]

for sl in slices:

    print('Slice is {}'.format(sl))

    for prp in grd.props:
        sconst = surf.copy()
        sconst.values = sl  # set constant value for surface
        print('Work with {}, slice at {}'.format(prp.name, sl))
        sconst.slice_grid3d(prp)
        fname = '{}_{}.gri'.format(prp.name, sl)
        sconst.to_file(fname)
        fname = '{}_{}.png'.format(prp.name, sl)
        sconst.quickplot(filename=fname)
