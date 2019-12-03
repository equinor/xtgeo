#!/bin/bash
set -e

rm -fr _tmp_swiginstall
mkdir -p _tmp_swiginstall

rm -fr _tmp_swig
mkdir -p _tmp_swig

INST=`pwd`

pushd _tmp_swiginstall
wget https://sourceforge.net/projects/swig/files/swig/swig-4.0.1/swig-4.0.1.tar.gz
tar xf swig-4.0.1.tar.gz
pushd swig-4.0.1
wget "https://ftp.pcre.org/pub/pcre/pcre-8.38.tar.gz"
./configure --prefix="$INST/_tmp_swig"

make
make install
