#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

import numpy

from glob import glob
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages, Extension
from distutils.command.build import build as _build
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    # 'segyio',
]


def the_version():
    """Process the version, to avoid non-pythonic version schemes.

    Means that e.g. 1.5.12+2.g191571d.dirty is turned to 1.5.12.2.dev0
    """

    version = versioneer.get_version()
    sver = version.split('.')
    print('\nFrom TAG description: {}'.format(sver))

    useversion = 'UNSET'
    if len(sver) == 3:
        useversion = version
    else:
        bugv = sver[2].replace('+', '.')

        if 'dirty' in version:
            ext = '.dev0'
        else:
            ext = ''
        useversion = '{}.{}.{}{}'.format(sver[0], sver[1], bugv, ext)

    print('Using version {}\n'.format(useversion))
    return useversion


class build(_build):
    # different order: build_ext *before* build_py
    sub_commands = [('build_ext', _build.has_ext_modules),
                    ('build_py', _build.has_pure_modules),
                    ('build_clib', _build.has_c_libraries),
                    ('build_scripts', _build.has_scripts)]


# get all C sources
sources = ['src/xtgeo/cxtgeo/cxtgeo.i']

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# cxtgeo extension module
_cxtgeo = Extension('xtgeo.cxtgeo._cxtgeo',
                    sources=sources,
                    extra_compile_args=['-Wno-uninitialized',
                                        '-Wno-strict-prototypes'],
                    include_dirs=['src/xtgeo/cxtgeo/clib/src', numpy_include],
                    library_dirs=['src/xtgeo/cxtgeo/clib/lib'],
                    libraries=['cxtgeo'],
                    swig_opts=['-modern'])

_cmdclass = {'build': build}
_cmdclass.update(versioneer.get_cmdclass())

setup(
    name='xtgeo',
    version=the_version(),
    cmdclass=_cmdclass,
    description="XTGeo Python library for grids, surfaces, wells, etc",
    long_description=readme + '\n\n' + history,
    author="Jan C. Rivenaes",
    author_email='jriv@equinor.com',
    url='https://github.com/Statoil/xtgeo-python',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    ext_modules=[_cxtgeo],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='xtgeo',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers :: Science/Research',
        'License :: OSI Approved',
        'License :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
        'Programming Language :: C',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
