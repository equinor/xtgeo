SWIGURL="https://ftp.osuosl.org/pub/blfs/conglomeration/swig"
SWIG="swig-3.0.12"

function pre_build {

    echo "Prebuild stage, install swig..."
    build_swig
}

function run_tests {
    set -x
    apt-get -y update > /dev/null
    apt-get -y install git > /dev/null
    git clone --depth 1 https://github.com/equinor/xtgeo-testdata ../../xtgeo-testdata
    ls -l ..
    ls -l ../..
    pip install pytest
    python -c "import xtgeo; print(xtgeo.__version__)"
    pushd ..
    pytest tests/test_surface
}
