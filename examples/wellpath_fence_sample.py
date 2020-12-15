from pathlib import Path
import xtgeo
import matplotlib.pyplot as plt

TPATH = Path("../xtgeo-testdata")

x = xtgeo.Polygons(TPATH / "polygons/etc/well16.pol")
y = x.copy()

y.rescale(10)

IDGROUPSX = x.dataframe.groupby(x.pname)
IDGROUPSY = y.dataframe.groupby(y.pname)

plt.figure(figsize=(7, 7))
for idx, grp in IDGROUPSX:
    plt.plot(grp[x.xname].values, grp[x.yname].values, label=str(idx))

for idx, grp in IDGROUPSY:
    plt.plot(grp[y.xname].values, grp[y.yname].values, label=str(idx))
plt.show()

print(grp[y.dhname].min(), grp[y.dhname].max())
print(y.dataframe)
