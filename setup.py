#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTGeo: Subsurface reservoir tool for maps, 3D grids etc."""
import platform
import subprocess
from glob import glob
import shutil
import os
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages, Extension
from distutils.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext

WINDOWS = False
CMAKECMD = ["cmake", ".."]
if "Windows" in platform.system():
    WINDOWS = True
    CMAKECMD = ["cmake", "..", "-DCMAKE_GENERATOR_PLATFORM=x64"]

def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except IOError:
        return []


try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except IOError:
    readme = "See README.md"


try:
    with open("HISTORY.md") as history_file:
        history = history_file.read()
except IOError:
    history = "See HISTORY.md"

requirements = parse_requirements("requirements.txt")

NUMPYVER = "numpy==1.13.3"
if platform.python_version_tuple() < ("3", "4", "99"):
    NUMPYVER = "numpy==1.10.4"
if platform.python_version_tuple() > ("3", "7", "0"):
    NUMPYVER = "numpy==1.16.3"

setup_requirements = [
    NUMPYVER,
    "pytest-runner",
    "cmake==3.13.3",
    "wheel",
    "setuptools_scm>=3.2.0",
]

test_requirements = ["pytest"]


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def src(x):
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root, x))


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

        if WINDOWS:
            print("******** REMOVE BUILD {}".format(self.build_temp))
            shutil.rmtree(self.build_temp)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            subprocess.check_call(CMAKECMD, cwd=self.build_temp)

        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp
        )


# get all C swig sources

sources = ["src/xtgeo/cxtgeo/cxtgeo.i"]

COMPILE_ARGS = ["-Wno-uninitialized", "-Wno-strict-prototypes"]
if WINDOWS:
    COMPILE_ARGS = ["/wd4267", "/wd4244"]

# cxtgeo extension module
_cxtgeo = CMakeExtension(
    "xtgeo.cxtgeo._cxtgeo",
    cmake_lists_dir="src/xtgeo/cxtgeo/clib",
    sources=sources,
    extra_compile_args=COMPILE_ARGS,
    include_dirs=["src/xtgeo/cxtgeo/clib/src"],
    library_dirs=["src/xtgeo/cxtgeo/clib/lib"],
    libraries=["cxtgeo"],
    swig_opts=["-modern"],
)

_CMDCLASS = {"build": build, "build_ext": build_ext}

_EXT_MODULES = [_cxtgeo]

# This is done for readthedocs purposes, which cannot deal with SWIG:
if "SWIG_FAKE" in os.environ:
    print("=================== FAKE SWIG SETUP ====================")
    shutil.copyfile("src/xtgeo/cxtgeo/cxtgeo_fake.py", "src/xtgeo/cxtgeo/cxtgeo.py")
    _EXT_MODULES = []

setup(
    name="xtgeo",
    cmdclass=_CMDCLASS,
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    author="Equinor R&T",
    url="https://github.com/equinor/xtgeo",
    license="LGPL-3.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    ext_modules=_EXT_MODULES,
    include_package_data=True,
    use_scm_version={"root": src(""), "write_to": src("src/xtgeo/_theversion.py")},
    zip_safe=False,
    keywords="xtgeo",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
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
    install_requires=requirements,
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
