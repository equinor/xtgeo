#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y pcre-devel
cd tmp
curl -O https://ftp.osuosl.org/pub/blfs/conglomeration/swig/swig-3.0.12.tar.gz
tar xzvf swig-3.0.12.tar.gz
cd swig-3.0.12
sh ./configure; make; make install

# # Compile wheels
# for PYBIN in /opt/python/*/bin; do
#     "${PYBIN}/pip" install -r /io/dev-requirements.txt
#     "${PYBIN}/pip" wheel /io/ -w wheelhouse/
# done

# # Bundle external shared libraries into the wheels
# for whl in wheelhouse/*.whl; do
#     auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
# done

# # Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done
