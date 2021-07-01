copy_test_files () {
    mkdir $CI_TEST_ROOT/testpath/

    pushd $CI_TEST_ROOT/testpath
    cp -r $CI_SOURCE_ROOT/tests tests
    ln -s $CI_SOURCE_ROOT/examples
    ln -s $CI_SOURCE_ROOT/xtgeo-testdata
    git clone --depth=1 https://github.com/equinor/xtgeo-testdata ../xtgeo-testdata
    popd
}

install_package () {
    pip install .
}

install_test_dependencies () {
    pip install -r requirements/requirements_test.txt
}

start_tests () {
    pushd $CI_TEST_ROOT/testpath
    pytest -vv --hypothesis-profile ci
    popd
}

cleanup () {
    rm -rf $CI_TEST_ROOT/testpath/../xtgeo-testdata
    rm -rf $CI_TEST_ROOT/testpath/TMP
}

run_tests() {
    copy_test_files

    if [ ! -z "${CI_PR_RUN:-}" ]
    then
        install_package
    fi

    install_test_dependencies

    pushd $CI_TEST_ROOT
    start_tests
    cleanup
}
