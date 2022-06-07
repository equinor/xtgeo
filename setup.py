#!/usr/bin/env python3
"""Setup for XTGeo - subsurface reservoir tool for maps, 3D grids etc."""
import os
import sys

try:
    import setuptools
    from setuptools import setup as setuptools_setup
except ImportError:
    print("\n*** Some requirements are missing, please run:")
    print("\n*** pip install -r requirements/requirements_setup.txt\n\n")
    raise

try:
    import skbuild
except ImportError:
    print("\n*** Some requirements are missing, please run:")
    print("\n*** pip install -r requirements/requirements_setup.txt\n")
    raise

from scripts import setup_utilities as setuputils

CMD = sys.argv[1]

README = setuputils.readmestuff("README.md")
HISTORY = setuputils.readmestuff("HISTORY.md")

setuputils.check_swig()  # Detect if swig is present and if case not, try a tmp install

REQUIREMENTS = setuputils.parse_requirements("requirements/requirements.txt")

TEST_REQUIREMENTS = setuputils.parse_requirements("requirements/requirements_test.txt")
SETUP_REQUIREMENTS = setuputils.parse_requirements(
    "requirements/requirements_setup.txt"
)
DOCS_REQUIREMENTS = setuputils.parse_requirements("requirements/requirements_docs.txt")
EXTRAS_REQUIRE = {"tests": TEST_REQUIREMENTS, "docs": DOCS_REQUIREMENTS}

CMDCLASS = {"clean": setuputils.CleanUp}


def src(anypath):
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root, anypath))


skbuild.setup(
    name="xtgeo",
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    use_scm_version={
        "root": src(""),
        "write_to": src("src/xtgeo/_theversion.py"),
    },
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    author="Equinor R&T",
    url="https://github.com/equinor/xtgeo",
    project_urls={
        "Documentation": "https://xtgeo.readthedocs.io/",
        "Issue Tracker": "https://github.com/equinor/xtgeo/issues",
    },
    license="LGPL-3.0",
    cmake_args=["-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass=CMDCLASS,
    zip_safe=False,
    keywords="xtgeo",
    command_options=setuputils.CMDSPHINX,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public "
        "License v3 or later (LGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    test_suite="tests",
    install_requires=REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
)

# Below is a hack to make "python setup.py develop" or "pip install -e ." to work.
# Without this, the xtgeo.egg-link file will be wrong, e.g.:
# /home/jan/work/git/xtg/xtgeo
# .
#
# instead of the correct:
# /home/jan/work/git/xtg/xtgeo/src
# ../
#
# The wrong egg-link comes when find_packages(where="src") finds a list of packages in
# scikit-build version of setup(). No clue why...

if CMD == "develop":
    print("Run in DEVELOP mode")
    setuptools_setup(  # use setuptools version of setup
        name="xtgeo",
        use_scm_version={
            "root": src(""),
            "write_to": src("src/xtgeo/_theversion.py"),
        },
        packages=setuptools.find_packages(where="src"),
        package_dir={"": "src"},
        zip_safe=False,
        test_suite="tests",
        install_requires=REQUIREMENTS,
        setup_requires=SETUP_REQUIREMENTS,
        tests_require=TEST_REQUIREMENTS,
        extras_require=EXTRAS_REQUIRE,
    )
