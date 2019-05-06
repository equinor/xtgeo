
function pre_build {
    $CPWD=$PWD
    ROOT="/"
    TMP="/tmp"
    IO="/io"

    SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
    SWIG="swig-3.0.12"

    # Install the SWIG package required by our library for install
    cd $TMP

    yum install -y pcre-devel
    curl -O ${SWIGURL}/${SWIG}.tar.gz
    tar xzf ${SWIG}.tar.gz
    cd $SWIG
    sh ./configure > /dev/null
    make > /dev/null
    make install > /dev/null
    cd $CPDW
    echo "PWD is $PWD"
}
