#!/bin/bash

function pre_build {
    SYS=$(uname)
    if [[ "$SYS" == "Linux" ]]; then
        echo "Prebuild stage, install swig..."
        # quite verbose...
        echo "Download and building swig...."
        build_swig > /dev/null
    else
        echo "OSX is running"
    fi
}

function run_tests {
    set -x
    SYS=$(uname)

    # need to install git on the test docker image to get the test data
    if [[ "$SYS" == "Linux" ]]; then
        apt-get -y update > /dev/null
        apt-get -y install git > /dev/null
    fi
    pip install pytest
    export TRAVISRUN=true
    python -c "import xtgeo; print(xtgeo.__version__)"
    pushd ..
    PYV=$(python --version | cut -d" " -f2  | cut -d. -f1,2)

    if [[ $SYS == "Linux" && $PYV == "3.6" ]]; then
        echo "For Python 3.6 / Linux, run pytest with coverage report in travis main"
    else
        git clone --depth 1 https://github.com/equinor/xtgeo-testdata ../xtgeo-testdata
        pytest tests --disable-warnings
    fi

}
