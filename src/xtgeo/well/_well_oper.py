# -*- coding: utf-8 -*-
"""Operations along a well, private module"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def rescale(self, delta=0.15):
    """Rescale by using a new MD increment

    The rescaling is technically done by interpolation in the Pandas dataframe
    """

    pdrows = pd.options.display.max_rows
    pd.options.display.max_rows = 999

    if self.mdlogname is None:
        self.geometrics()

    dfr = self._df.copy().set_index(self.mdlogname)

    logger.debug('Initial dataframe\n %s', dfr)

    start = dfr.index[0]
    stop = dfr.index[-1]

    nentry = int(round((stop - start) / delta))

    dfr = dfr.reindex(dfr.index.union(np.linspace(start, stop, num=nentry)))
    dfr = dfr.interpolate('index', limit_area='inside').loc[
        np.linspace(start, stop, num=nentry)]

    dfr[self.mdlogname] = dfr.index
    dfr.reset_index(inplace=True, drop=True)

    for lname in dfr.columns:
        if lname in self._wlogtype:
            ltype = self._wlogtype[lname]
            if ltype == 'DISC':
                dfr = dfr.round({lname: 0})

    logger.debug('Updated dataframe:\n%s', dfr)

    pd.options.display.max_rows = pdrows  # reset

    self._df = dfr
