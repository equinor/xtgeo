#!/bin/bash

function pre_build {
    echo "Prebuild stage, install swig..."
    # quite verbose...
    build_swig > /dev/null
}

function run_tests {
    set -x
    # need to install git on the test docker image to get the test data
    apt-get -y update > /dev/null
    apt-get -y install git > /dev/null
    git clone --depth 1 https://github.com/equinor/xtgeo-testdata ../../xtgeo-testdata
    pip install pytest
    export TRAVISRUN=true
    echo $USER
    python -c "import xtgeo; print(xtgeo.__version__)"
    pushd ..
    pytest tests
}
