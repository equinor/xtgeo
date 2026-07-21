# This shell script is to be sourced and run from a github workflow
# when xtgeo is to be tested towards a new RMS Python enviroment

run_tests() {
    set_test_variables

    copy_test_files

    install_test_dependencies

    run_pytest
}

set_test_variables() {
    echo "Setting variables for xtgeo tests..."
    CI_TEST_ROOT=$CI_ROOT/xtgeo_test_root
}

copy_test_files () {
    echo "Copy xtgeo test files to $CI_TEST_ROOT..."
    mkdir -p $CI_TEST_ROOT
    cp -r $PROJECT_ROOT/tests $CI_TEST_ROOT
    
    echo "Create symlinks from $CI_TEST_ROOT to files in $PROJECT_ROOT..."    
    ln -s $PROJECT_ROOT/examples $CI_TEST_ROOT/examples
    ln -s $PROJECT_ROOT/conftest.py $CI_TEST_ROOT/conftest.py
    ln -s $PROJECT_ROOT/pyproject.toml $CI_TEST_ROOT/pyproject.toml
}

install_test_dependencies () {
    echo "Installing test dependencies..."

    # Cache CMake FetchContent deps (Eigen/fmt) per Python version so they are
    # fetched once instead of re-cloned on every run. Reuse the CI cache dir if
    # provided, otherwise fall back to a temp location for local runs.
    local cache_root="${CACHE_DIR:+$HOME/$CACHE_DIR}"
    cache_root="${cache_root:-${CI_ROOT:-/tmp}/cache}"
    local py_tag="${PYTHON_VERSION:-$(python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')}"
    export SKBUILD_CMAKE_DEFINE="FETCHCONTENT_BASE_DIR=${cache_root}/xtgeo-cmake-deps/${py_tag}"
    echo "SKBUILD_CMAKE_DEFINE=$SKBUILD_CMAKE_DEFINE"

    pip install ".[dev]"

    echo "Dependencies installed successfully. Listing installed dependencies..."
    pip list
}

run_pytest () {
    echo "Running xtgeo tests with pytest..."
    pushd $CI_TEST_ROOT
    pytest ./tests -n 4 -vv --testdatapath $XTGEO_TESTDATA_PATH
    popd
}