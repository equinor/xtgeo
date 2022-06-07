"""Functions/classes for use in setup.py in order to make the latter clean and lean."""
import fnmatch
import os
import platform
import re
import subprocess  # nosec
import sys
from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable
from distutils.version import LooseVersion
from glob import glob
from os.path import dirname, exists
from pathlib import Path
from shutil import rmtree, which

from setuptools_scm import get_version
from skbuild.command import set_build_base_mixin
from skbuild.constants import CMAKE_BUILD_DIR, CMAKE_INSTALL_DIR, SKBUILD_DIR
from skbuild.utils import new_style

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
        "__pycache__",
        "pip-wheel-metadata",
        ".eggs",
        "dist",
        "sdist",
        "wheel",
        ".pytest_cache",
        "docs/_apiref",
        "docs/_build",
        "htmlcov",
    )

    CLEANFOLDERSRECURSIVE = ["__pycache__", "_tmp_*", "xtgeo.egg-info"]
    CLEANFILESRECURSIVE = ["*.pyc", "*.pyo", ".coverage", "coverage.xml"]

    CLEANFILES = glob("src/xtgeo/cxtgeo/cxtgeo*")
    CLEANFILES.extend(glob("src/xtgeo/cxtgeo/_cxtgeo*"))

    @staticmethod
    def ffind(pattern, path):
        """Find files."""
        result = []
        for root, _, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    @staticmethod
    def dfind(pattern, path):
        """Find folders."""
        result = []
        for root, dirs, _ in os.walk(path):
            for name in dirs:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def run(self):
        """Execute run.

        After calling the super class implementation, this function removes
        the directories specific to scikit-build ++.
        """
        # super().run()

        for dir_ in CleanUp.CLEANFOLDERS:
            if exists(dir_):
                print("Removing: {}".format(dir_))
            if exists(dir_):
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
            if exists(fil_):
                os.remove(fil_)


# ======================================================================================
# Sphinx
# ======================================================================================

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


def readmestuff(filename):
    """For README, HISTORY etc."""
    response = "See " + filename
    try:
        with open(filename) as some_file:
            response = some_file.read()
    except OSError:
        pass
    return response


# ======================================================================================
# Other helpers
# ======================================================================================


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except OSError:
        return []
