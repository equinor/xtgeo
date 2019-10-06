REM Cleaning up stuff when on windows
@ECHO OFF
RMDIR /Q /S build 2>NUL
RMDIR /Q /S build 2>NUL
RMDIR /Q /S src\xtgeo\cxtgeo\clib\build 2>NUL
RMDIR /Q /S dist 2>NUL
RMDIR /Q /S .eggs 2>NUL
DEL /S /Q *.egg *.egg.info *.pyc *.pyo __pychache__ 2> NUL
DEL /S /Q *.dll cxtgeo.py cxtgeo_wrap.c 2> NUL
