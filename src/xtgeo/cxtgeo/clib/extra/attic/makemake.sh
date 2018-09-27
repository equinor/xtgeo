#!/usr/bin/sh
# Overriding settings in Makefile.PL
echo "Make a Makefile for project/res..."

# update version
echo "Version update from date and GIT version..."
date=`date +%G%m%d_%H%M_`
sha=`git rev-parse --short HEAD`
echo '$VERSION="'$date$sha'";' > cxtgeo_version.pl
echo $date$sha > cxtgeo_version.txt

perl Makefile.PL

