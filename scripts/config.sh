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
    git clone --depth 1 https://github.com/equinor/xtgeo-testdata ../../xtgeo-testdata
    pip install pytest
    pip install pytest-cov
    export TRAVISRUN=true
    echo $USER
    python -c "import xtgeo; print(xtgeo.__version__)"
    pushd ..
    pytest tests --disable-warnings
    # codecov
}
