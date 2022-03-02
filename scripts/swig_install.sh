#!/bin/bash
set -e
# Try to install swig in cases it is missing on the system

rm -fr _tmp_swiginstall
mkdir -p _tmp_swiginstall

rm -fr _tmp_swig
mkdir -p _tmp_swig

INST=$(pwd)

SWIG="swig-3.0.12"
SWIG_ALT1="https://ftp.osuosl.org/pub/blfs/conglomeration/swig/${SWIG}.tar.gz"
SWIG_ALT2="/project/res/etc/swig_install/${SWIG}.tar.gz"  # fallback in Equinor

PCRE_ALT1="https://sourceforge.net/projects/pcre/files/pcre/8.45/pcre-8.45.tar.gz"
PCRE_ALT2="/project/res/etc/swig_install/pcre-8.45.tar.gz"  # fallback in Equinor

pushd _tmp_swiginstall
curl -O ${SWIG_ALT1} || cp -v ${SWIG_ALT2} .

tar xf ${SWIG}.tar.gz
pushd ${SWIG}

curl -L ${PCRE_ALT1} > pcre-8.45.tar.gz || cp -v ${PCRE_ALT2} .

sh Tools/pcre-build.sh
echo "Running configure..."
./configure --prefix="$INST/_tmp_swig"  > swiginstall.log

echo "Running make..."
make >> swiginstall.log
echo "Running make install..."
make install >> swiginstall.log
