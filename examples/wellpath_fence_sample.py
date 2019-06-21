import xtgeo
import matplotlib.pyplot as plt

x = xtgeo.Polygons("../xtgeo-testdata/polygons/etc/well16.pol")
y = x.copy()

y.rescale(10, constant=True)

idgroupsx = x.dataframe.groupby(x.pname)
idgroupsy = y.dataframe.groupby(y.pname)

plt.figure(figsize=(7, 7))
for id, grp in idgroupsx:
    plt.plot(grp[x.xname].values, grp[x.yname].values, label=str(id))

for id, grp in idgroupsy:
    plt.plot(grp[y.xname].values, grp[y.yname].values, label=str(id))
plt.show()

print(grp[y.dhname].min(), grp[y.dhname].max())
print(y.dataframe)
