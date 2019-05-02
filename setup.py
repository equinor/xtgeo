#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

import subprocess
from glob import glob
import os
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages, Extension
from distutils.command.build import build as _build

import numpy
# import versioneer
from setuptools_scm import get_version

def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = parse_requirements("requirements.txt")

setup_requirements = [
    "numpy",
    "pytest-runner",
    "cmake",
    "wheel",
    "setuptools_scm>=3.2.0",
]

test_requirements = ["pytest"]

# -----------------------------------------------------------------------------
# Explaining versions:
# As system the PEP 440 major.minor.micro is used:
# - major: API or any very larger changes
# - minor: Functionality added, mostly backward compatibility but some
#          functions may change. Also includes larger refactoring of code.
# - micro: Added functionality and bug fixes with no expected side effects
# - Provide a tag on the form 3.4.0 for each release!
#
# -----------------------------------------------------------------------------


# def the_version():
#     """Process the version, to avoid non-pythonic version schemes.

#     Means that e.g. 1.5.12+2.g191571d.dirty is turned to 1.5.12.dev2.precommit

#     This function must be ~identical to xtgeo._theversion.py
#     """

#     version = versioneer.get_version()
#     sver = version.split(".")
#     print("\nFrom TAG description: {}".format(sver))

#     useversion = "UNSET"
#     if len(sver) == 3:
#         useversion = version
#     else:
#         print("SVER", sver)
#         bugv = sver[2].replace("+", ".dev")

#         if "dirty" in version:
#             ext = ".precommit"
#         else:
#             ext = ""
#         useversion = "{}.{}.{}{}".format(sver[0], sver[1], bugv, ext)

#     print("Using version {}\n".format(useversion))
#     return useversion


class build(_build):
    # different order: build_ext *before* build_py
    sub_commands = [
        ("build_ext", _build.has_ext_modules),
        ("build_py", _build.has_pure_modules),
        ("build_clib", _build.has_c_libraries),
        ("build_scripts", _build.has_scripts),
    ]


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=".", sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        print(self.cmake_lists_dir)
        self.build_temp = os.path.join(self.cmake_lists_dir, "build")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            subprocess.check_call(["cmake", ".."], cwd=self.build_temp)

        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp
        )


# get all C swig sources

sources = ["src/xtgeo/cxtgeo/cxtgeo.i"]

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# cxtgeo extension module
_cxtgeo = CMakeExtension(
    "xtgeo.cxtgeo._cxtgeo",
    cmake_lists_dir="src/xtgeo/cxtgeo/clib",
    sources=sources,
    extra_compile_args=["-Wno-uninitialized", "-Wno-strict-prototypes"],
    include_dirs=["src/xtgeo/cxtgeo/clib/src", numpy_include],
    library_dirs=["src/xtgeo/cxtgeo/clib/lib"],
    libraries=["cxtgeo"],
    swig_opts=["-modern"],
)

_cmdclass = {"build": build}
# _cmdclass.update(versioneer.get_cmdclass())

setup(
    name="xtgeo",
    version=get_version(root='.', relative_to=__file__),
    cmdclass=_cmdclass,
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    long_description=readme + "\n\n" + history,
    author="R&T Equinor",
    url="https://github.com/equinor/xtgeo",
    license="LGPL-3.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    ext_modules=[_cxtgeo],
    include_package_data=True,
    use_scm_version=True,
    zip_safe=False,
    keywords="xtgeo",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers :: Science/Research",
        "License :: OSI Approved",
        "License :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    test_suite="tests",
    install_requires=requirements,
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
