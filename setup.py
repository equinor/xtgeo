#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTGeo: Subsurface reservoir tool for maps, 3D grids etc."""

import os
import platform
import shutil
from os.path import splitext, exists, dirname, basename
from glob import glob
from shutil import rmtree
from distutils.command.clean import clean as _clean
from setuptools import find_packages

import skbuild

from skbuild.command import set_build_base_mixin
from skbuild.utils import new_style
from skbuild.constants import CMAKE_BUILD_DIR, CMAKE_INSTALL_DIR, SKBUILD_DIR

# ======================================================================================
# Overriding and extending setup commands
# ======================================================================================


class CleanUp(set_build_base_mixin, new_style(_clean)):
    """Custom implementation of ``clean`` setuptools command.

    Overriding clean in order to get rid if "dist" folder and etc
    """

    skroot = dirname(SKBUILD_DIR())
    CLEANFOLDERS = (
        CMAKE_INSTALL_DIR(),
        CMAKE_BUILD_DIR(),
        SKBUILD_DIR(),
        skroot,
        "TMP",
        "__pycache__",
        ".eggs",
        "dist",
        "sdist",
        "wheel",
        ".pytest_cache",
        "xtgeo.egg-info",
    )

    CLEANFILES = glob("src/xtgeo/cxtgeo/cxtgeo*")

    def run(self):
        """After calling the super class implementation, this function removes
        the directories specific to scikit-build ++."""
        super(CleanUp, self).run()

        for dir_ in CleanUp.CLEANFOLDERS:
            if exists(dir_):
                print("Removing: {}".format(dir_))
            if not self.dry_run and exists(dir_):
                rmtree(dir_)

        for fil_ in CleanUp.CLEANFILES:
            if exists(fil_):
                print("Removing: {}".format(fil_))
            if not self.dry_run and exists(fil_):
                os.remove(fil_)


# ======================================================================================
# README stuff
# ======================================================================================

try:
    with open("README.md") as readme_file:
        README = readme_file.read()
except IOError:
    README = "See README.md"


try:
    with open("HISTORY.md") as history_file:
        HISTORY = history_file.read()
except IOError:
    HISTORY = "See HISTORY.md"


# ======================================================================================
# Requirements:
# ======================================================================================


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except IOError:
        return []


REQUIREMENTS = parse_requirements("requirements.txt")

TEST_REQUIREMENTS = ["pytest"]

# ======================================================================================
# Special:
# ======================================================================================

# This is done for readthedocs purposes, which cannot deal with SWIG:
if "SWIG_FAKE" in os.environ:
    print("=================== FAKE SWIG SETUP ====================")
    shutil.copyfile("src/xtgeo/cxtgeo/cxtgeo_fake.py", "src/xtgeo/cxtgeo/cxtgeo.py")
    _EXT_MODULES = []


def src(x):
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root, x))


skbuild.setup(
    name="xtgeo",
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    use_scm_version={"root": src(""), "write_to": src("src/xtgeo/_theversion.py")},
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    author="Equinor R&T",
    url="https://github.com/equinor/xtgeo",
    license="LGPL-3.0",
    # cmake_with_sdist=False,
    include_package_data=True,
    cmake_args=[],
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    cmdclass={"clean": CleanUp},
    zip_safe=False,
    keywords="xtgeo",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public "
        "License v3 or later (LGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    test_suite="tests",
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    # setup_requires=["setuptools_scm"],
)
