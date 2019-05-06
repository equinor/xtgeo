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
    git clone https://git@github.com/equinor/xtgeo-testdata.git ../xtgeo-testdata
    python --version
    echo $PWD
}
