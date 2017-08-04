#/bin/sh
\rm -fr build/ *tgeo.py* *.so *.c
\rm -fr TMP/ dist/ *.egg*
rm -f `find . -name "*.pyc"` 
rm -f `find . -name "*.so"` 
rm -fr `find . -name "__pycache__"` 



