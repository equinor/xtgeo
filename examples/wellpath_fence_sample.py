import os
from pathlib import Path

import matplotlib.pyplot as plt
import xtgeo

TPATH = Path("../xtgeo-testdata")

x = xtgeo.polygons_from_file(TPATH / "polygons/etc/well16.pol")
y = x.copy()

y.rescale(10)

IDGROUPSX = x.get_dataframe().groupby(x.pname)
IDGROUPSY = y.get_dataframe().groupby(y.pname)

plt.figure(figsize=(7, 7))
for idx, grp in IDGROUPSX:
    plt.plot(grp[x.xname].values, grp[x.yname].values, label=str(idx))

for idx, grp in IDGROUPSY:
    plt.plot(grp[y.xname].values, grp[y.yname].values, label=str(idx))

if "SKIP_PLOT" in os.environ:
    print("Plotting skipped")
else:
    plt.show()


print(y.get_dataframe())
