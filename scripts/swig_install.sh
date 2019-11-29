#!/bin/sh
set -e

mkdir _tmp_swig
pushd _tmp_swig
wget "https://sourceforge.net/projects/swig/files/swig/swig-4.0.1/swig-4.0.1.tar.gz"
tar xf swig-4.0.1.tar.gz
pushd swig-4.0.1
wget "https://ftp.pcre.org/pub/pcre/pcre-8.38.tar.gz"
