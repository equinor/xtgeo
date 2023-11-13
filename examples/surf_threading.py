import io
import threading

import xtgeo


def test_xtgeo():
    stream = io.BytesIO()
    surface = xtgeo.RegularSurface(ncol=10, nrow=12, xinc=10, yinc=10)
    surface.to_file(stream)
    print("XTGeo succeeded")


threading.Timer(1.0, test_xtgeo).start()
