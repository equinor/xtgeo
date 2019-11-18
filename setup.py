#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTGeo: Subsurface reservoir tool for maps, 3D grids etc."""
import platform
import subprocess  # nosec
from glob import glob
import shutil
import os
import re

from distutils.command.build import build as _build
from distutils.spawn import find_executable
from distutils.version import LooseVersion

from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages, Extension

from setuptools.command.build_ext import build_ext as _build_ext

SWIGMINIMUM = "4.0.1"
SWIGDL = "https://sourceforge.net/projects/swig/files/swig/swig-XX/swig-XX.tar.gz"
PCREDL = "https://ftp.pcre.org/pub/pcre/pcre-8.38.tar.gz"


def cmakeexe():
    """Check cmakeexe version"""
    cmake = find_executable("cmake3")
    if not cmake:
        cmake = "cmake"  # assume the version from "pip install cmake"?
    return cmake


WINDOWS = False
if "Windows" in platform.system():
    WINDOWS = True
    CMAKECMD = [
        cmakeexe(),
        "..",
        "-DCMAKE_GENERATOR_PLATFORM=x64",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
else:
    CMAKECMD = [
        cmakeexe(),
        "..",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

print("CMAKE COMMAND: {}".format(CMAKECMD))


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except IOError:
        return []


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

REQUIREMENTS = parse_requirements("requirements.txt")

NUMPYVER = "numpy==1.13.3"
if platform.python_version_tuple() < ("3", "4", "99"):
    NUMPYVER = "numpy==1.10.4"
if platform.python_version_tuple() > ("3", "7", "0"):
    NUMPYVER = "numpy==1.16.3"

if WINDOWS:
    NUMPYVER = "numpy==1.17.3"

SETUP_REQUIREMENTS = [
    NUMPYVER,
    "pytest-runner",
    "cmake==3.13.3",
    "wheel",
    "setuptools_scm>=3.2.0",
]

TEST_REQUIREMENTS = ["pytest"]


class build_ext(_build_ext):  # pylint: disable=invalid-name
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


def src(x):
    root = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root, x))


class build(_build):  # pylint: disable=invalid-name
    # different order: build_ext *before* build_py
    sub_commands = [
        ("build_ext", _build.has_ext_modules),
        ("build_py", _build.has_pure_modules),
        ("build_clib", _build.has_c_libraries),
        ("build_scripts", _build.has_scripts),
    ]


class CMakeExtension(Extension):
    # pylint: disable=dangerous-default-value
    def __init__(self, name, cmake_lists_dir=".", sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        print(self.cmake_lists_dir)
        self.build = os.path.join(".", "build")
        self.build_temp = os.path.join(self.cmake_lists_dir, "build")
        self.lib = os.path.join(self.cmake_lists_dir, "lib")
        self.buildswig = os.path.abspath(os.path.join(".", "tmp_buildswig"))

        if WINDOWS:
            print("******** REMOVE BUILD {}".format(self.build_temp))
            if os.path.exists(self.build_temp):
                shutil.rmtree(self.build_temp)
            if os.path.exists(self.lib):
                shutil.rmtree(self.lib)
            if os.path.exists(self.build):
                shutil.rmtree(self.build)

        if not os.path.exists(self.lib):
            os.makedirs(self.lib)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            subprocess.check_call(CMAKECMD, cwd=self.build_temp)  # nosec

        subprocess.check_call(  # nosec
            [cmakeexe(), "--build", ".", "--target", "install", "--config", "Release"],
            cwd=self.build_temp,
        )

        if not self.swigok():
            if WINDOWS:
                raise SystemExit("SWIG is missing or wrong version in Windows")
            self.swiginstall()

    @staticmethod
    def swigok():
        """Check swig version"""
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

    def swiginstall(self):
        """Install correct SWIG unless it is available on system (which is preferred)"""
        print("Installing swig... {}".format(SWIGMINIMUM))
        if os.path.exists(self.buildswig):
            shutil.rmtree(self.buildswig)
        os.makedirs(self.buildswig)

        swigdownload = SWIGDL.replace("XX", SWIGMINIMUM)
        swigver = "swig-" + SWIGMINIMUM
        swigtargz = swigver + ".tar.gz"
        swigdir = os.path.join(self.buildswig, swigver)

        with open(os.path.join(self.buildswig, "swigdownload.log"), "w") as logfile:
            print("Download swig and pcre...")
            subprocess.check_call(  # nosec
                ["wget", swigdownload],
                cwd=self.buildswig,
                stdout=logfile,
                stderr=logfile,
            )
            subprocess.check_call(  # nosec
                ["tar", "xf", swigtargz],
                cwd=self.buildswig,
                stdout=logfile,
                stderr=logfile,
            )

        with open(os.path.join(swigdir, "pcredownload.log"), "w") as logfile:
            subprocess.check_call(  # nosec
                ["wget", PCREDL], cwd=swigdir, stdout=logfile, stderr=logfile
            )

        with open(os.path.join(swigdir, "pcre_build.log"), "w") as logfile:
            print("Compile and install pcre and swig...")
            subprocess.check_call(  # nosec
                ["Tools/pcre-build.sh"], cwd=swigdir, stdout=logfile, stderr=logfile
            )

        with open(os.path.join(swigdir, "swig_conf.log"), "w") as logfile:
            subprocess.check_call(  # nosec
                ["./configure", "--prefix=" + os.path.abspath(swigdir)],
                cwd=swigdir,
                stdout=logfile,
                stderr=logfile,
            )
        with open(os.path.join(swigdir, "swig_make.log"), "w") as logfile:
            subprocess.check_call(["make"], cwd=swigdir, stdout=logfile)  # nosec

        with open(os.path.join(swigdir, "swig_makeinstall.log"), "w") as logfile:
            subprocess.check_call(  # nosec
                ["make", "install"], cwd=swigdir, stdout=logfile
            )

        os.environ["PATH"] = swigdir + os.pathsep + os.environ["PATH"]
        print("Installing pcre and swig... DONE")


# get all C swig sources

SOURCES = ["src/xtgeo/cxtgeo/cxtgeo.i"]

COMPILE_ARGS = ["-Wno-uninitialized", "-Wno-strict-prototypes"]
if WINDOWS:
    COMPILE_ARGS = ["/wd4267", "/wd4244"]

# cxtgeo extension module
_CXTGEO = CMakeExtension(
    "xtgeo.cxtgeo._cxtgeo",
    cmake_lists_dir="src/xtgeo/cxtgeo/clib",
    sources=SOURCES,
    extra_compile_args=COMPILE_ARGS,
    include_dirs=["src/xtgeo/cxtgeo/clib/src"],
    library_dirs=["src/xtgeo/cxtgeo/clib/lib"],
    libraries=["cxtgeo"],
    swig_opts=[],
)

_CMDCLASS = {"build": build, "build_ext": build_ext}

_EXT_MODULES = [_CXTGEO]

# This is done for readthedocs purposes, which cannot deal with SWIG:
if "SWIG_FAKE" in os.environ:
    print("=================== FAKE SWIG SETUP ====================")
    shutil.copyfile("src/xtgeo/cxtgeo/cxtgeo_fake.py", "src/xtgeo/cxtgeo/cxtgeo.py")
    _EXT_MODULES = []

setup(
    name="xtgeo",
    cmdclass=_CMDCLASS,
    description="XTGeo is a Python library for 3D grids, surfaces, wells, etc",
    long_description=README + "\n\n" + HISTORY,
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
    setup_requires=SETUP_REQUIREMENTS,
)
