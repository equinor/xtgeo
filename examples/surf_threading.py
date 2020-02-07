import io
import threading
import xtgeo
def test_xtgeo():
    stream = io.BytesIO()
    surface = xtgeo.RegularSurface()
    surface.to_file(stream)
    print("XTGeo succeeded")
threading.Timer(1.0, test_xtgeo).start()
