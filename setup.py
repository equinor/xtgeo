#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTGeo: Subsurface reservoir tool for maps, 3D grids etc."""

import os
import sys
import shutil
import re
from os.path import exists, dirname
from glob import glob
from shutil import rmtree
import platform
from distutils.command.clean import clean as _clean
import fnmatch
from distutils.spawn import find_executable
from distutils.version import LooseVersion
import subprocess  # nosec

from setuptools import find_packages

import skbuild
from setuptools import setup as zetup

from skbuild.command import set_build_base_mixin
from skbuild.utils import new_style
from skbuild.constants import CMAKE_BUILD_DIR, CMAKE_INSTALL_DIR, SKBUILD_DIR

from sphinx.setup_command import BuildDoc as _BuildDoc
from setuptools_scm import get_version

CMD = sys.argv[1]


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
        "pip-wheel-metadata",
        ".eggs",
        "dist",
        "sdist",
        "wheel",
        ".pytest_cache",
        "xtgeo.egg-info",
        "docs/apiref",
        "docs/_build",
        "docs/_static",
        "docs/_templates",
    )

    CLEANFOLDERSRECURSIVE = ["__pycache__", "_tmp_*"]
    CLEANFILESRECURSIVE = ["*.pyc", "*.pyo"]

    CLEANFILES = glob("src/xtgeo/cxtgeo/cxtgeo*")

    @staticmethod
    def ffind(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    @staticmethod
    def dfind(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in dirs:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def run(self):
        """After calling the super class implementation, this function removes
        the directories specific to scikit-build ++."""
        super(CleanUp, self).run()

        for dir_ in CleanUp.CLEANFOLDERS:
            if exists(dir_):
                print("Removing: {}".format(dir_))
            if not self.dry_run and exists(dir_):
                rmtree(dir_)

        for dir_ in CleanUp.CLEANFOLDERSRECURSIVE:
            for pd in self.dfind(dir_, "."):
                print("Remove folder {}".format(pd))
                rmtree(pd)

        for fil_ in CleanUp.CLEANFILESRECURSIVE:
            for pf in self.ffind(fil_, "."):
                print("Remove file {}".format(pf))
                os.unlink(pf)

        for fil_ in CleanUp.CLEANFILES:
            if exists(fil_):
                print("Removing: {}".format(fil_))
            if not self.dry_run and exists(fil_):
                os.remove(fil_)


# ======================================================================================
# Sphinx
# ======================================================================================


class BuildDocCustom(_BuildDoc):
    """Trick issue with cxtgeo prior to docs are built """

    shutil.copyfile("src/xtgeo/clib/cxtgeo_fake.py", "src/xtgeo/cxtgeo/cxtgeo.py")

    def run(self):
        super(BuildDocCustom, self).run()


cmdclass = {"build_sphinx": BuildDocCustom}

CMDSPHINX = {
    "build_sphinx": {
        "project": ("setup.py", "xtgeo"),
        "version": ("setup.py", get_version()),
        "release": ("setup.py", ""),
        "source_dir": ("setup.py", "docs"),
    }
}


# ======================================================================================
# README stuff and Sphinx
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
# Detect if swig is present (and if case not, do a tmp install on some platforms)
# ======================================================================================

SWIGMINIMUM = "3.0.1"


def swigok():
    """Check swig version"""
    if CMD == "clean":
        return True
    swigexe = find_executable("swig")
    if not swigexe:
        print("Cannot find swig in system")
        return False
    sout = subprocess.check_output([swigexe, "-version"]).decode("utf-8")  # nosec
    swigver = re.findall(r"SWIG Version ([0-9.]+)", sout)[0]
    if LooseVersion(swigver) >= LooseVersion(SWIGMINIMUM):
        print("OK, found swig in system, version is >= ", SWIGMINIMUM)
        return True

    print("Found swig in system but version is < ", SWIGMINIMUM)
    return False


if not swigok():
    if "Linux" in platform.system():
        print("Installing swig from source (tmp) ...")
        subprocess.check_call(  # nosec
            ["bash", "swig_install.sh"],
            cwd="scripts",
        )
    else:
        raise SystemExit("Cannot find valid swig install")

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


def src(x):
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root, x))


# ======================================================================================
# Setup:
# ======================================================================================

skbuild.setup(
    name="xtgeo",
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    use_scm_version={"root": src(""), "write_to": src("src/xtgeo/_theversion.py")},
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    author="Equinor R&T",
    url="https://github.com/equinor/xtgeo",
    license="LGPL-3.0",
    # cmake_args=["-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"],
    packages=["xtgeo"],
    package_dir={"": "src"},
    cmdclass={"clean": CleanUp},
    zip_safe=False,
    keywords="xtgeo",
    command_options=CMDSPHINX,
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
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    test_suite="tests",
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
)
