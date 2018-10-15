"""
Print a CSV from all INIT vectors
"""

from os.path import join as ojn
import numpy as np
import xtgeo

EXPATH1 = '../../xtgeo-testdata/3dgrids/reek'
GRIDFILEROOT = ojn(EXPATH1, 'REEK')

# EXPATH1 = '/scratch/troll_fmu/rnyb/10_troll_r003/realization-0/iter' +\
#           '-0/eclipse/model'
# GRIDFILEROOT = ojn(EXPATH1, 'ECLIPSE')

INITPROPS = 'all'  # will look for all vectors that looks "gridvalid"


def all_init_as_csv():
    """Get dataframes, print as CSV."""

    print('Loading Eclipse data {}'.format(GRIDFILEROOT))
    grd = xtgeo.grid3d.Grid()
    grd.from_file(GRIDFILEROOT, fformat='eclipserun', initprops=INITPROPS)
    print('Get dataframes...')
    dfr = grd.dataframe(activeonly=True)

    print(dfr.head())
    print('Filter out columns with constant values...')
    dfr = dfr.iloc[:, ~np.isclose(0, dfr.var())]
    print(dfr.head())
    print('Write to file...')
    dfr.to_csv('mycsvdump.csv', index=False)


if __name__ == '__main__':

    all_init_as_csv()
