"""
Compute statistics of N realisations. In this the realisations are "faked" by
just adding a constant to each loop. It provides and insight on memory
handling and speed.
"""

from os.path import join as ojn
import numpy.ma as npma
import xtgeo

# from memory_profiler import profile

EXPATH1 = '../../xtgeo-testdata/3dgrids/reek'

GRIDFILEROOT = ojn(EXPATH1, 'REEK')

INITPROPS = ['PORO', 'PERMX']
RESTARTPROPS = ['PRESSURE', 'SWAT', 'SOIL']
RDATES = [20001101, 20030101]

NRUN = 10


def sum_stats():
    """Accumulate numpies for all realisations and then do stats.

    This will be quite memory intensive, and memory consumption will
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
    porovalues = npma.vstack(propsd['PORO'])
    poromeanarray = porovalues.mean(axis=0)
    porostdarray = porovalues.std(axis=0)
    print(poromeanarray)
    print(poromeanarray.mean())
    print(porostdarray)
    print(porostdarray.mean())
    return poromeanarray.mean()


def sum_running_stats():
    """Find avg per realisation and do a cumulative rolling mean.

    Memory consumption shall be very low.
    """

    for irel in range(NRUN):
        # load as Eclipse run; this will look for EGRID, INIT, UNRST

        print('Loading realization no {}'.format(irel))

        grd = xtgeo.grid3d.Grid()
        grd.from_file(GRIDFILEROOT, fformat='eclipserun',
                      restartprops=RESTARTPROPS, restartdates=RDATES,
                      initprops=INITPROPS)

        nnum = float(irel + 1)
        for prop in grd.props:
            if prop.name == 'PORO':
                prop.values += irel * 0.001  # mimic variability aka ensembles
            else:
                prop.values += irel * 1  # just to mimic variability

            if prop.name == 'PORO':
                if irel == 0:
                    pcum = prop.values1d
                else:
                    pavg = prop.values1d / nnum
                    pcum = pcum * (nnum - 1) / nnum
                    pcum = npma.vstack([pcum, pavg])
                    pcum = pcum.sum(axis=0)

    # find the averages:
    print(pcum)
    print(pcum.mean())
    return pcum.mean()


if __name__ == '__main__':

    AVG1 = sum_stats()
    AVG2 = sum_running_stats()

    if AVG1 == AVG2:
        print('Same result, OK!')
