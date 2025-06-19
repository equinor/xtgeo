import sys
from pathlib import Path

import openvds  # noqa

print("Running...", file=sys.stderr)
import xtgeo  # noqa

grid = xtgeo.grid_from_file(
    Path("../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff")
)
print("Done")
