#!/bin/bash
set -e
# Try to install swig in cases it is missing on the system

rm -fr _tmp_swiginstall
mkdir -p _tmp_swiginstall

rm -fr _tmp_swig
mkdir -p _tmp_swig

INST=$(pwd)
SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
SWIG="swig-3.0.12"

pushd _tmp_swiginstall
curl -O ${SWIGURL}/${SWIG}.tar.gz
tar xf ${SWIG}.tar.gz
pushd ${SWIG}
curl -L https://sourceforge.net/projects/pcre/files/pcre/8.45/pcre-8.45.tar.gz > pcre-8.45.tar.gz
sh Tools/pcre-build.sh
echo "Running configure..."
./configure --prefix="$INST/_tmp_swig"  > swiginstall.log

echo "Running make..."
make >> swiginstall.log
echo "Running make install..."
make install >> swiginstall.log
