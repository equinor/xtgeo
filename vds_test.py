import faulthandler
import sys
from pathlib import Path

faulthandler.enable()


import openvds  # noqa

print("Running...", file=sys.stderr)
import xtgeo  # noqa

try:
    grid = xtgeo.grid_from_file(
        Path("../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff")
    )
except Exception as e:
    print(e)
    raise e
print("Done")
