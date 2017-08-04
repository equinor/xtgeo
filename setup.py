#!/usr/bin/env python
# =============================================================================
# A "makefile" setup file for xtgeo library
# By Jan C. Rivenaes
#
# Commands: (use python (for 2.7) or python3 as 'python' below
#    -- make tests:
#    python setup.py tests
#    -- install as private user:
#    python setup.py install --user
#
# =============================================================================

# import os
import subprocess
import re
# from distutils.core import setup, Command
# from distutils.core import Command
# from distutils.command.sdist import sdist as _sdist
from setuptools import find_packages, setup
from setuptools import Command
from setuptools.command.sdist import sdist as _sdist

VERSION_PY = """
# This file is originally generated from Git information by running:
# 'setup.py version'. Distribution tarballs contain a pre-generated
# copy of this file.
__version__ = '{}'
_xtgeo_build = '{}'
"""


def update_version_py():
    try:
        p = subprocess.Popen(["git", "describe",
                              "--tags", "--always"],
                             stdout=subprocess.PIPE)
    except EnvironmentError:
        print("unable to run git, leaving xtgeo/_version.py alone")
        return
    stdout = p.communicate()[0]
    if p.returncode != 0:
        print("unable to run git, leaving xtgeo/_version.py alone")
        return

#    ver = stdout.rstrip(b'\0')
    ver = stdout.decode('UTF-8')
    ver = ver.replace("\n", "")
    # ensure a 3digit tag:
    mver = ver[1:6]
    f = open("xtgeo/_version.py", "w")
    f.write(VERSION_PY.format(mver, ver))
    f.close()
    print("set xtgeo/_version.py to build ID <{}>".format(ver))


def get_version():
    update_version_py()
    try:
        f = open("xtgeo/_version.py")
        print("Opening _version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            print("Version is {}".format(ver))
            return ver
    return None


class Version(Command):
    description = "update _version.py from Git repo"
    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        update_version_py()
        print("Version build ID is now", get_version())


class sdist(_sdist):
    def run(self):
        update_version_py()
        # unless we update this, the sdist command will keep using the old
        # version
        self.distribution.metadata.version = get_version()
        return _sdist.run(self)


setup(name="xtgeo",
      version=get_version(),
      description="XTGeo library for Python",
      author="Jan C. Rivenaes",
      author_email="JRIV@statoil.com",
      url="http://git.statoil.no.com/xtgeo",
      packages=find_packages(exclude=['tests']),
      license="Statoil",
      cmdclass={"version": Version, "sdist": sdist},
      test_suite="tests",
      )
