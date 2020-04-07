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
    export TRAVISRUN=true
    echo $USER
    python -c "import xtgeo; print(xtgeo.__version__)"
    pushd ..
    # PYV=$(python -c 'import sys; v = sys.version_info; print("{}.{}".format(v[0], v[1]))')
    PYV=$(python --version | cut -d" " -f2  | cut -d. -f1,2)
    # codecov
    if [[ $SYS == "Linux" && $PYV == "3.8" ]]; then
        # apt-get -y install curl > /dev/null
        pip install pytest-cov
        pip install codecov
        COVERAGE_FNAME="/tmp/test_coverage.xml"
        pytest tests --disable-warnings --cov=xtgeo --cov-report=xml:$COVERAGE_FNAME
        # bash <(curl -s https://codecov.io/bash) -Z -c -f $COVERAGE_FNAME
    else
        pytest tests --disable-warnings
    fi
    pwd

}
