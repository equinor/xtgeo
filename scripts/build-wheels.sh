#!/bin/bash
set -e -x

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

# ===========================

cd $ROOT

/opt/python/cp36-cp36m/bin/pip install twine cmake
ln -s /opt/python/cp36-cp36m/bin/cmake /usr/bin/cmake

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    echo $PYBIN
    if [[ $PYBIN == *"cp"* ]]; then
        echo "======================================="
        echo "Install for $PYBIN"
        echo "======================================="
        if [[ $PYBIN == "/opt/python/cp27-cp27m/bin" ]]; then
            echo "****************  Skip $PYBIN"
            continue
        fi
        "${PYBIN}/pip" install numpy
        "${PYBIN}/pip" wheel /io/ -w /io/wheelhouse/
        cd $IO
        "${PYBIN}/python" /io/setup.py sdist -d /io/wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
cd $IO
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# # # Install packages and test
# # for PYBIN in /opt/python/*/bin/; do
# #     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
# #     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# # done


# # Test
# # for PYBIN in /opt/python/cp3*/bin/; do
# #     "${PYBIN}/pip" install -r /io/requirements-test.txt
# #     "${PYBIN}/pip" install --no-index -f /io/wheelhouse nb-cpp
# #     (cd "$PYHOME"; "${PYBIN}/pytest" /io/test/)
# # done

# # #  Upload
# # for WHEEL in /io/wheelhouse/nb_cpp*; do
# #     /opt/python/cp37-cp37m/bin/twine upload \
# #         --skip-existing \
# #         -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" \
# #         "${WHEEL}"
# # done
