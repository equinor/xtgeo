SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
SWIG="swig-3.0.12"

function pre_build {

    yum install -y pcre-devel
    curl -O ${SWIGURL}/${SWIG}.tar.gz
    tar xzf ${SWIG}.tar.gz
    pushd $SWIG
    sh ./configure > /dev/null
    make > /dev/null
    make install > /dev/null
    popd
}

function run_tests {
    set -x
    ls -l ..
    ls -l ../..
    pip install pytest
    python -c "import xtgeo; print(xtgeo.__version__)"
    ln -s ../tests tests
    pytest tests/test_simple
}
