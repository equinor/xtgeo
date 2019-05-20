#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTGeo: Subsurface reservoir tool for maps, 3D grids etc."""
import platform
import subprocess
from glob import glob
import os
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages, Extension
from distutils.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext

# import numpy


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except FileNotFoundError:
        return []


try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except FileNotFoundError:
    readme = "See README.md"


try:
    with open("HISTORY.md") as history_file:
        history = history_file.read()
except FileNotFoundError:
    history = "See HISTORY.md"

requirements = parse_requirements("requirements.txt")

numpyver = "numpy==1.13.3"
if platform.python_version_tuple() < ("3", "4", "99"):
    numpyver = "numpy==1.10.4"
if platform.python_version_tuple() > ("3", "7", "0"):
    numpyver = "numpy==1.16.3"

setup_requirements = [
    numpyver,
    "pytest-runner",
    "cmake",
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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            subprocess.check_call(["cmake", ".."], cwd=self.build_temp)

        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp
        )


# get all C swig sources

sources = ["src/xtgeo/cxtgeo/cxtgeo.i"]

# # Obtain the numpy include directory. This logic works across numpy versions.
# try:
#     numpy_include = numpy.get_include()
# except AttributeError:
#     numpy_include = numpy.get_numpy_include()

# cxtgeo extension module
_cxtgeo = CMakeExtension(
    "xtgeo.cxtgeo._cxtgeo",
    cmake_lists_dir="src/xtgeo/cxtgeo/clib",
    sources=sources,
    extra_compile_args=["-Wno-uninitialized", "-Wno-strict-prototypes"],
    include_dirs=["src/xtgeo/cxtgeo/clib/src"],
    library_dirs=["src/xtgeo/cxtgeo/clib/lib"],
    libraries=["cxtgeo"],
    swig_opts=["-modern"],
)

_cmdclass = {"build": build, "build_ext": build_ext}

setup(
    name="xtgeo",
    cmdclass=_cmdclass,
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    author="Equinor R&T",
    url="https://github.com/equinor/xtgeo",
    license="LGPL-3.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    ext_modules=[_cxtgeo],
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
