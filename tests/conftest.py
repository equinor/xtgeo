# coding: utf-8
# Setup common stuff for pytests...


def pytest_runtest_setup(item):
    # called for running each test in 'a' directory
    print("\nSetting up test\n", item)
