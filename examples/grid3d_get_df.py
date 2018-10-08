"""
Example on how to retrieve a dataframe (Pandas) from a 3D grid.

Explanation:

Both a GridProperties and a Grid instance can return a dataframe.
The `grd.gridprops` attribute below is the GridProperties, and
this will return a a dataframe by default which does not include
XYZ and ACTNUM, as this information is only from the Grid (geometry).

The grid itself can also return a dataframe, and in this case
XYZ and ACNUM will be returned by default. Also properties that
are "attached" to the Grid via a GridProperties attribute will
be shown.

"""

from os.path import join as ojn
import xtgeo

EXPATH1 = '../../xtgeo-testdata/3dgrids/reek'

GRIDFILEROOT = ojn(EXPATH1, 'REEK')

INITPROPS = ['PORO', 'PERMX']
RESTARTPROPS = ['PRESSURE', 'SWAT', 'SOIL']
RDATES = [20001101, 20030101]


def extractdf():
    """Extract dataframe from Eclipse case"""

    # gete dataframe from the grid only
    grd = xtgeo.grid3d.Grid(GRIDFILEROOT + '.EGRID')
    dataframe = grd.dataframe()  # will not have any grid props
    print(dataframe)

    # # load as Eclipse run; this will automatically look for EGRID, INIT, UNRST
    # grd = xtgeo.grid3d.Grid()
    # grd.from_file(GRIDFILEROOT, fformat='eclipserun', initprops=INITPROPS,
    #               restartprops=RESTARTPROPS, restartdates=RDATES)

    # # dataframe from a GridProperties instance, in this case grd.gridprops
    # dataframe = grd.gridprops.dataframe()  # properties for all cells

    # print(dataframe)

    # # Get a dataframe for all cells, with ijk and xyz. In this case
    # # a grid key input is required:
    # dataframe = grd.dataframe()

    # print(dataframe)  # default is for all cells

    # # For active cells only:
    # dataframe = grd.dataframe(active=True)

    # print(dataframe)

    # dataframe.to_csv('reek_sim.csv')


if __name__ == '__main__':
    extractdf()
