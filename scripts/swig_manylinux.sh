#!/bin/sh
# For manylinux docker installs in CI

SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
SWIG="swig-3.0.12"
yum install -y pcre-devel
curl -O ${SWIGURL}/${SWIG}.tar.gz
tar xzf ${SWIG}.tar.gz
cd $SWIG
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
