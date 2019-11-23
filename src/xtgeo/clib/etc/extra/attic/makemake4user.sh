#!/usr/bin/sh
# Note, to have an local $MY_BINDIST (aka SDP_BINDIST) variable, 
# you need to have in .cshrc
# -JRIV

echo "Install path is <$MY_BINDIST>"

date=`date +%G%m%d_%H%M_`
sha=`git rev-parse --short HEAD`
echo '$VERSION="'$date$sha'_'$USER'";' > cxtgeo_version.pl
echo $date$sha'_'$USER > cxtgeo_version.txt

#perl Makefile.PL LIB=$MY_BINDIST/lib/perl5 INSTALL_BASE=$MY_BINDIST
perl Makefile.PL INSTALL_BASE=$MY_BINDIST

