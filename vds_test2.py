import faulthandler
import sys
from pathlib import Path

faulthandler.enable()


print("Running...", file=sys.stderr)
import xtgeo  # noqa
import openvds  # noqa


try:
    grid = xtgeo.grid_from_file(
        Path("../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff")
    )
except Exception as e:
    print(e)
    raise e
print("Done")
