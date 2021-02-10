"""Functions/classes for use in setup.py in order to make the latter clean and lean."""
import os
import sys
import re
from os.path import exists, dirname
from glob import glob
from shutil import rmtree
import fnmatch
import platform
from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable
from distutils.version import LooseVersion
import subprocess  # nosec

from skbuild.command import set_build_base_mixin
from skbuild.utils import new_style
from skbuild.constants import CMAKE_BUILD_DIR, CMAKE_INSTALL_DIR, SKBUILD_DIR

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
        "docs/apiref",
        "docs/_build",
        "docs/_static",
        "docs/_templates",
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
# Detect if swig is present (and if case not, do a tmp install on some platforms)
# ======================================================================================

SWIGMINIMUM = "3.0.1"


def check_swig():
    """Check if swig is installed; if not try a tmp install if linux."""

    def swigok():
        """Check swig version."""
        if CMD == "clean":
            return True
        swigexe = find_executable("swig")
        if not swigexe:
            print("Cannot find swig in system")
            return False
        sout = subprocess.check_output([swigexe, "-version"]).decode("utf-8")  # nosec
        swigver = re.findall(r"SWIG Version ([0-9.]+)", sout)[0]
        if LooseVersion(swigver) >= LooseVersion(SWIGMINIMUM):
            print(
                "OK, found swig in system, version is >= {} ({})".format(
                    SWIGMINIMUM, swigexe
                )
            )
            return True

        print(
            "Found swig in system but version is < {} ({})".format(SWIGMINIMUM, swigexe)
        )
        return False

    if not swigok():
        if "SWIG_INSTALL_KOMODO" in os.environ:
            print("Hmm KOMODO setup but still cannot find swig... workaround required!")
            with open(".swigtmp", "w") as tmpfile:
                tmpfile.write("SWIG")

        elif "Linux" in platform.system():
            print("Installing swig from source (tmp workaround) ...")
            print("It is strongly recommended that SWIG>=3 is installed permanent!")
            subprocess.check_call(  # nosec
                ["bash", "swig_install.sh"],
                cwd="scripts",
            )
        else:
            raise SystemExit("Cannot find valid swig install")


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
