"""
Compute statistics of N realisations. In this the realisations are "faked" by
just adding a constant to each loop. It provides and insight on memory
handling and speed.
"""

import io
from os.path import join as ojn
import numpy.ma as npma
import xtgeo

# from memory_profiler import profile

EXPATH1 = "../../xtgeo-testdata/surfaces/reek/2/01_topreek_rota.gri"

GRIDFILEROOT = ojn(EXPATH1, "REEK")


NRUN = 1000


def sum_running_stats():
    """Find avg per realisation and do a cumulative rolling mean.

    Memory consumption shall be very low.
    """

    for irel in range(NRUN):
        # load as Eclipse run; this will look for EGRID, INIT, UNRST

        print("Loading realization no {}".format(irel))

        srf = xtgeo.RegularSurface(EXPATH1)

        nnum = float(irel + 1)
        srf.values += irel * 1  # just to mimic variability

        if irel == 0:
            pcum = srf.values1d
        else:
            pavg = srf.values1d / nnum
            pcum = pcum * (nnum - 1) / nnum
            pcum = npma.vstack([pcum, pavg])
            pcum = pcum.sum(axis=0)

    # find the averages:
    print(pcum)
    print(pcum.mean())
    return pcum.mean()


def sum_running_stats_bytestream():
    """Find avg per realisation and do a cumulative rolling mean.

    Memory consumption shall be very low.
    """

    for irel in range(NRUN):
        # load as Eclipse run; this will look for EGRID, INIT, UNRST

        print("Loading realization no {}".format(irel))

        with open(EXPATH1, "rb") as myfile:
            stream = io.BytesIO(myfile.read())

        srf = xtgeo.RegularSurface(stream, fformat="irap_binary")

        nnum = float(irel + 1)
        srf.values += irel * 1  # just to mimic variability

        if irel == 0:
            pcum = srf.values1d
        else:
            pavg = srf.values1d / nnum
            pcum = pcum * (nnum - 1) / nnum
            pcum = npma.vstack([pcum, pavg])
            pcum = pcum.sum(axis=0)

    # find the averages:
    print(pcum)
    print(pcum.mean())
    return pcum.mean()


if __name__ == "__main__":

    AVG1 = sum_running_stats()

    print(AVG1)

    print("Now as bytestream")
    AVG2 = sum_running_stats_bytestream()

    print(AVG2)
