# -*- coding: utf-8 -*-
"""Wrapper on _version.
Process the version, to avoid non-pythonic version schemes.
Means that e.g. 1.5.12+2.g191571d.dirty is turned to 1.5.12.2.dev0
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
        bugv = sver[2].replace('+', '.')

        if 'dirty' in version:
            ext = '.dev0'
        else:
            ext = ''
        useversion = '{}.{}.{}{}'.format(sver[0], sver[1], bugv, ext)

    return useversion
