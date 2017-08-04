python -m unittest -v tests.test_regular_surface
nosetests -s --nologcapture tests.test_regular_surface:TestSurface.test_create
nosetests --debug=test tests.test_regular_surface:TestSurface.test_create
nosetests tests.test_regular_surface:TestSurface.test_create
