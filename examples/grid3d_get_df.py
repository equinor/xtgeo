"""
Example on how to retrieve dataframe (Pandas) from a 3D grid.
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
    # load as Eclipse run; this will look for EGRID, INIT, UNRST
    grd = xtgeo.grid3d.Grid()
    grd.from_file(GRIDFILEROOT, fformat='eclipserun', initprops=INITPROPS,
                  restartprops=RESTARTPROPS, restartdates=RDATES)

    grdprops = grd.get_gridproperties()  # get a GridProperties instance

    dataframe = grdprops.dataframe()

    print(dataframe)

    # to get a dataframe for alle cells, with ijk and xyz:
    dataframe = grdprops.dataframe(activeonly=False, ijk=True, xyz=True)

    print(dataframe)

    dataframe.to_csv('reek_sim.csv')


if __name__ == '__main__':
    extractdf()
