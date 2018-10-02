"""
Compute statistics of N realisations. In this the realisatiosn are "faked" by
just adding a constant to each loop. It provides and insight on memomery
handling and speed.
"""

from os.path import join as ojn
import numpy as np
import numpy.ma as ma
import xtgeo
import time

# from memory_profiler import profile

EXPATH1 = '../../xtgeo-testdata/3dgrids/reek'
MAP = '../../xtgeo-testdata/surfaces/reek/1/basereek_rota.gri'
ROFFG = '../../xtgeo-testdata/3dgrids/reek/reek_geo_grid.roff'
ROFFP = '../../xtgeo-testdata/3dgrids/reek/reek_geo_stooip.roff'

GRIDFILEROOT = ojn(EXPATH1, 'REEK')

INITPROPS = ['PORO', 'PERMX']
RESTARTPROPS = ['PRESSURE', 'SWAT', 'SOIL']
RDATES = [20001101, 20030101]

NRUN = 30

INCREMENTAL = True


def sum_stats():
    """Accumulate numpies for all realisations and then do stats.

    This will be quite memory intensive, and memory conmsumption will
    increase linearly.
    """

    propsd = {}

    for irel in range(NRUN):
        # load as Eclipse run; this will look for EGRID, INIT, UNRST

        print('Loading realization no {}'.format(irel))
        grd = xtgeo.grid3d.Grid()
        grd.from_file(GRIDFILEROOT, fformat='eclipserun', initprops=INITPROPS,
                      restartprops=RESTARTPROPS, restartdates=RDATES)

        for prop in grd.props:
            if prop.name not in propsd:
                propsd[prop.name] = []
            if prop.name == 'PORO':
                prop.values += irel * 0.001  # mimic variability aka ensembles
            else:
                prop.values += irel * 1  # just to mimic variability

            propsd[prop.name].append(prop.values1d)

    # find the averages:
    porovalues = ma.stack(propsd['PORO'])
    poromeanarray = porovalues.mean(axis=0)
    porostdarray = porovalues.std(axis=0)
    print(poromeanarray)
    print(poromeanarray.mean())
    print(porostdarray)
    print(porostdarray.mean())


def sum_running_stats():
    """Find avg per realisation and do a running mean.

    This will be much less memory intensive, and memory conmsumption shall
    be relatively low.
    """


    for irel in range(NRUN):
        # load as Eclipse run; this will look for EGRID, INIT, UNRST

        print('Loading realization no {}'.format(irel))
        #xtgeo.surface_from_file(MAP)
        # xtgeo.gridproperty_from_file(GRIDFILEROOT + '.INIT', name='PORO')
        time.sleep(0.1)
        grd = xtgeo.grid3d.Grid(ROFFG)



        gprop = xtgeo.grid3d.GridProperty(ROFFP, grid=grd)

        gprop.__del__()
        grd.__del__()

        # grdprop = xtgeo.gridproperty_from_file(ROFFP)

        # for prop in grd.props:
        #     if prop.name == 'PORO':
        #         prop.values += irel * 0.001  # mimic variability aka ensembles
        #     else:
        #         prop.values += irel * 1  # just to mimic variability

    #         if prop.name == 'PORO':
    #             if irel == 0:
    #                 psumm = prop.values1d.copy()
    #             else:
    #                 pmean = prop.values1d.copy() / NRUN
    #                 psumm = psumm * (NRUN - 1) / NRUN
    #                 psumm = ma.stack([pmean, psumm])
    #                 psumm = psumm.sum(axis=0)
    #                 del pmean

    # # find the averages:
    # poromeanarray = psumm
    # print(poromeanarray)
    # print(poromeanarray.mean())


if __name__ == '__main__':

    # sum_stats()
    sum_running_stats()
