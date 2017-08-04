#!/bin/sh
# running this for installing to VIRTUAL ENVs
echo "\nThis is for VENV install only ..."

# current directory is name of the module
mymodule=${PWD##*/}
echo ""
echo "Current module is $mymodule"
echo ""

# remove all earlier stuff
sh cleanup.sh

if [ $mymodule == "cxtgeo" ]; then
    #python setup.py build_ext -i
    echo "Running setup for for cxtgeo..."
    python setup.py bdist_wheel
    # trick to make unit testing working...
    /bin/cp build/lib*/cxtgeo/_cxtgeo*.so cxtgeo/.
else
    python setup.py bdist_wheel --universal
fi

echo "\nInstall to $INSTUSE\n"

if [ $mymodule == "cxtgeo" ]; then
    echo ""
    echo "Running install for cxtgeo..."
    pip uninstall --yes $mymodule
    pip install --upgrade ./dist/*
else
    pip uninstall --yes $mymodule
    pip install --upgrade ./dist/*
fi

echo ""
echo "Install in virtual env..."
echo ""

# install script if any:
# wrapper scripts:
if [ -d "bin" ]; then
    echo "Sync scripts to $INST /bin ..."
    rsync -v -L --chmod=a+rx --perms bin/* ${VIRTUAL_ENV}/bin/.
fi
# sphinx-apidoc -f -o docs xtgeo
