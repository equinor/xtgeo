# -*- coding: utf-8 -*-
"""Wrapper on _version.
Process the version, to avoid non-pythonic version schemes.
Means that e.g. 1.5.12+2.g191571d.dirty is turned to 1.5.12.dev2.precommit
"""

import xtgeo._version


def theversion():

    versions = xtgeo._version.get_versions()
    version = versions['version']
    sver = version.split('.')

    useversion = 'UNSET'
    if len(sver) == 3:
        useversion = version
    else:
        bugv = sver[2].replace('+', '.dev')

        if 'dirty' in version:
            ext = '.precommit'
        else:
            ext = ''
        useversion = '{}.{}.{}{}'.format(sver[0], sver[1], bugv, ext)

    return useversion
