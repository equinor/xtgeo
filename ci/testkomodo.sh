XTG_TEST_RUNPATH=$CI_TEST_ROOT/run

RMS_VERSIONS=(
    12.1.4
    13.1.2
    14.1.0
)

run_tests() {
    copy_test_files
    start_tests
}

copy_test_files () {
    git clone --depth=1 https://github.com/equinor/xtgeo-testdata $CI_TEST_ROOT/xtgeo-testdata
    # xtgeo tests look two directories back for xtgeo-testdata
    mkdir -p $XTG_TEST_RUNPATH
    cp -r $CI_SOURCE_ROOT/tests $XTG_TEST_RUNPATH
    ln -s $CI_SOURCE_ROOT/conftest.py $XTG_TEST_RUNPATH/conftest.py
    ln -s $CI_SOURCE_ROOT/pyproject.toml $XTG_TEST_RUNPATH/pyproject.toml
}

start_tests () {
    pushd $CI_TEST_ROOT
    echo "Testing xtgeo against Komodo"
    install_and_test
    for version in ${RMS_VERSIONS[@]}; do
        test_in_roxenv $version
    done
    popd
}

install_and_test () {
    install_package
    run_pytest
}

install_package () {
    pushd $CI_SOURCE_ROOT
    pip install ".[dev]"
    popd
}

run_pytest () {
    pushd $XTG_TEST_RUNPATH
    pytest -n 4 -vv
    popd
}

test_in_roxenv () {
    set +e
    source /project/res/roxapi/aux/roxenvbash $1
    # Unsetting an empty PYTHONPATH after sourcing roxenvbash
    # may result in an error.
    unset PYTHONPATH
    set -e

    python -m venv roxenv --system-site-packages

    source roxenv/bin/activate
    echo "Testing xtgeo against RMS $1"
    install_and_test
    deactivate
    rm -rf roxenv
}
