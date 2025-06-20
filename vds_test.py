import faulthandler
import subprocess
import sys
from pathlib import Path

faulthandler.enable()

print("Initial modules:", len(sys.modules), file=sys.stderr)

import openvds  # noqa

print("After openvds modules:", len(sys.modules), file=sys.stderr)

print("OpenVDS location:", openvds.__file__, file=sys.stderr)
result = subprocess.run(
    ["dumpbin", "/dependents", openvds.__file__],
    capture_output=True,
    text=True,
    shell=True,
)
print("OpenVDS dependencies:", file=sys.stderr)
print(result.stdout, file=sys.stderr)

print("Running...", file=sys.stderr)
import xtgeo  # noqa

print("After xtgeo modules:", len(sys.modules), file=sys.stderr)

try:
    grid = xtgeo.grid_from_file(
        Path("../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff")
    )
except Exception as e:
    print(e, file=sys.stderr)
    raise e
print("Done")
