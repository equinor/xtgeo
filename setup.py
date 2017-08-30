#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pytest'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    name='xtgeo',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="XTGeo Python library for grids, surfaces, wells, etc",
    long_description=readme + '\n\n' + history,
    author="Jan C. Rivenaes",
    author_email='jriv@statoil.com',
    url='https://git.statoil.no/jriv/pyxtgeo',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='xtgeo',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
