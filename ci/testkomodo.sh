copy_test_files () {
    mkdir $CI_TEST_ROOT/testpath/

    pushd $CI_TEST_ROOT/testpath
    cp -r $CI_SOURCE_ROOT/tests tests
    cp -r $CI_SOURCE_ROOT/conftest.py conftest.py
    ln -s $CI_SOURCE_ROOT/examples
    ln -s $CI_SOURCE_ROOT/xtgeo-testdata
    git clone --depth=1 https://github.com/equinor/xtgeo-testdata ../xtgeo-testdata
    popd
}

install_package () {
    pip install ".[dev]"
}

start_tests () {
    pushd $CI_TEST_ROOT/testpath
    pytest -n auto -vv
    popd
}

cleanup () {
    rm -rf $CI_TEST_ROOT/testpath/../xtgeo-testdata
}

run_tests() {
    copy_test_files

    install_package

    pushd $CI_TEST_ROOT
    start_tests
    cleanup
}
