#!/bin/bash

function pre_build {
    echo "Prebuild stage, install swig..."
    # way too verbose...
    build_swig > /dev/null
    # rather:
    # SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
    # SWIG="swig-3.0.12"
    # yum install -y pcre-devel
    # curl -O ${SWIGURL}/${SWIG}.tar.gz
    # tar xzf ${SWIG}.tar.gz
    # cd $SWIG
    # sh ./configure \
    #    --with-python \
    #    --with-python3 \
    #    --without-perl5 \
    #    --without-ruby \
    #    --without-tcl \
    #    --without-maximum-compile-warnings \
    #    > /dev/null
    # make > /dev/null
    # make install > /dev/null
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
