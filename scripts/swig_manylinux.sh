#!/bin/sh
# For manylinux docker installs in CI

echo "Download swig..."
SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
SWIG="swig-4.0.1"
curl -O ${SWIGURL}/${SWIG}.tar.gz
tar xzf ${SWIG}.tar.gz
cd $SWIG
echo "Download swig... done"

echo "Download pcre..."
curl -O "https://ftp.pcre.org/pub/pcre/pcre-8.38.tar.gz"
sh Tools/pcre-build.sh > /dev/null
echo "PCRE is built locally"

sh ./configure \
   --with-python \
   --with-python3 \
   --without-perl5 \
   --without-ruby \
   --without-tcl \
   --without-maximum-compile-warnings \
   > /dev/null
make > /dev/null
make install > /dev/null
echo "SWIG installed"
